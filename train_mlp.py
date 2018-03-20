# coding: utf-8
import pandas as pd
import numpy as np
import os
import tensorflow as tf
from repeat_mlp import RepeatMLP
from data_supply import DataSupplierMLP
from constants import constants as const

#n_window = 3
#tr_ratio = 0.8
#init_lr = 1.e-4
#layers = [128, 128, 128, 2]

#log_directory = './summary'
#max_iteration = 100
#model_file_path = 'predict_mlp.ckpt'
#predict_path = 'predicted_result.csv'

def split_data(data, **split):
    """
    split data

    :param data:    data set (pandas.DataFrame)
    :param split:   split parameters (dictionary)
    :return:        train data and test data (tuple of pandas.DataFrame)
    """
    
    # case: split by data
    if 'date' in split.keys() and 'key' in split.keys():
        data_train = data.query('{} < {}'.format(split['key'], split['date']))
        data_test = data.query('{} >= {}'.format(split['key'], split['date']))
        
    # case: split as detremined number
    elif 'num_train' in split.keys():
        idx = data.index.values
        if 'random' in split.keys():
            np.random.shuffle(idx)
        data_train = data.loc[idx[:split['num_train']]]
        data_test = data.loc[idx[split['num_train']:]]
        
    else:
        data_train = data
        data_test = data
    
    return data_train, data_test

class TrainerMLP(object):
    def __init__(self):
        # get data
        dtype_dict = {'sales_date_1': np.int32, 'member_id': str, 'gender': str, 'pref_cd': str, 'bought': np.int32}
        for i in range(1, const.n_window + 1):
            dtype_dict['days_delta_{}'.format(i)] = np.float32
            dtype_dict['shop_id_{}'.format(i)] = str
            dtype_dict['age_{}'.format(i)] = np.float32
            dtype_dict['item_num_{}'.format(i)] = np.float32
            dtype_dict['list_price_sum_{}'.format(i)] = np.float32
        
        self.data = pd.read_csv(const.data_path, dtype=dtype_dict).iloc[:,1:].dropna()
        self.n_data = self.data.shape[0] # データの数
        self.n_user = self.data.member_id.unique().shape[0] # ユーザ数
        
        # split by random 
        #self._data_tr, self._data_te = split_data(self.data, num_train=int(self.n_data * const.tr_ratio), random=True)
        
        
        # split by date
        self._data_tr, self._data_te = split_data(self.data, date=const.split_date, key='sales_date_1')
        
        # 2017/04/01 - 2017/12/20
        self._data_te = self._data_te.query('sales_date_1 <= {}'.format(const.valid_latest_date))
        # 評価データを境界日付のみかつユーザーユニークに変更
        # 具体的には、2017/05/01のデータを
        # member_idとbought_at_1（最新購入日）でソートし
        # member_idが重複するものを除く（→ユーザーユニーク）
        #self._data_te = self._data_te.query('sales_date_1 == "{}"'.format(const.split_date)).sort_values(['member_id', 'sales_date_1']).drop_duplicates(subset=['member_id'])
                
        # 対象ユーザ数と全件数
        tf.logging.info("users: {} records, logged data: {} records".format(self.n_user, self.n_data))
        # train/testそれぞれの データ数
        tf.logging.info("train data: {} records, test data: {} records".format(self._data_tr.shape[0],
                                                                               self._data_te.shape[0]))
        # train/testそれぞれの 購買確率(0~1)
        # positiveデータがどれだけあるか
        tf.logging.info("bought ratio:: train data: {}, test data: {}".format(np.average(self._data_tr.bought.values),
                                                                              np.average(self._data_te.bought.values)))
        
        # labels
        shop_id = []
        for i in range(1, const.n_window + 1):
            shop_id.append(self.data['shop_id_{}'.format(i)].values)
            
        # shop_idのユニークな値をarrayとして返す
        shop_id = np.unique(np.concatenate(shop_id, axis=0))
        
        gender = self.data['gender'].drop_duplicates().values 
        pref_cd = self.data['pref_cd'].drop_duplicates().values
        
        # data supplier
        self.train_supplier = DataSupplierMLP(name='supplier_tr', data=self._data_tr)
        self.test_supplier = DataSupplierMLP(name='supplier_te', data=self._data_te)
        
        # indices init op
        self.train_init_op = None
        self.test_init_op = None
        
        # MLP model
        self.model = RepeatMLP(name='repeat_mlp',
                               shop_id={'labels': shop_id, 'dim': 16},
                               gender={'labels': gender, 'dim': 2},
                               pref_cd={'labels': pref_cd, 'dim': 6})
        self.model.setup_hash_tables()
        
        # operation
        self.x_tr = self.t_tr = self.y_tr = None
        self.x_te = self.t_te = self.y_te = None
        self.next_idx = None
        self.total_loss = self.evals = self.ops = None
        self.accumulated_loss = self.accumulate_op = self.reset_loss_op = None
        #self.total_loss_test = self.evals = self.ops = None
        #self.accumulated_loss_test = self.accumulate_op_test = self.reset_loss_op_test = None
        self._oplimizer = self.global_step = self.learning_rate = self.train_step = None
        self.epochs = self.epoch_count_op = None
        
    def build_graphs(self):
        # indices
        iterator = tf.data.Iterator.from_structure(self.train_supplier.data_idx.output_types,
                                                  self.train_supplier.data_idx.output_shapes)
        
        self.next_idx = iterator.get_next()
        with tf.name_scope('initialize_train_iterator'):
            self.train_init_op = iterator.make_initializer(self.train_supplier.data_idx)
        with tf.name_scope('initilize_test_iteratior'):
            self.test_init_op = iterator.make_initializer(self.test_supplier.data_idx)
            
        # train
        self.x_tr, self.t_tr = self.train_supplier(self.next_idx)
        self.y_tr = self.model(self.x_tr, True)
        
        # accumulation losses
        with tf.name_scope('loss'):
            self.total_loss = self.train_supplier.loss(self.y_tr, self.t_tr, output_size=const.layers[-1])
            self.accumulated_loss = tf.Variable(0., trainable=False, name='accumulative_loss')
            self.accumulate_op = tf.assign_add(self.accumulated_loss, self.total_loss, name='accumulate_loss')
            self.reset_loss_op = tf.assign(self.accumulated_loss, 0., name='reset_loss')
            
        # test
        self.x_te, self.t_te = self.test_supplier(self.next_idx)
        self.y_te = self.model(self.x_te, False)
        with tf.name_scope('evaluation'):
            self.evals, self.ops = self.test_supplier.evaluation(self.y_te, self.t_te)
            
        # test accumulation losses
        #with tf.name_scope('test_loss'):
        #    self.total_loss_test = self.test_supplier.loss(self.y_te, self.t_te, output_size=const.layers[-1])
        #    self.accumulated_loss_test = tf.Variable(0., trainable=False, name='test_accumulative_loss')
        #    self.accumulate_op_test = tf.assign_add(self.accumulated_loss_test, self.total_loss_test, name='test_accumulate_loss')
        #    self.reset_loss_op_test = tf.assign(self.accumulated_loss_test, 0., name='test_reset_loss')
        
        # optimizer
        with tf.name_scope('optimizer'):
            self.global_step = tf.Variable(0., trainable=False, name='global_step')
            self._optimizer = tf.train.AdamOptimizer(const.init_lr)
            self.train_step = self._optimizer.minimize(self.total_loss, global_step=self.global_step)
            
        # global epoch
        with tf.name_scope('epoch_counter'):
            self.epochs = tf.Variable(0, trainable=False, name='epoch')
            self.epoch_count_op = tf.assign_add(self.epochs, 1, name='epoch_count')
            
    def set_summary(self):
        # loss
        tf.summary.scalar('loss', self.accumulated_loss)
        
        # test_loss
        #tf.summary.scalar('test_loss', self.accumulated_loss_test)
        # accuracy
        tf.summary.scalar('accuracy', self.evals[0])
        # precision
        tf.summary.scalar('precision', self.evals[1])
        # recall
        tf.summary.scalar('recall', self.evals[2])
        
    def initialize(self, sess):
        # initialize
        sess.run(tf.global_variables_initializer())
        self.model.init_hash_tables(sess)
        
        # restore trained model
        if os.path.exists(const.log_dir):
            saver = tf.train.Saver()
            saver.restore(sess=sess, save_path=os.path.join(const.log_dir, const.model_file_path))
        
    def train(self, sess):
        # writer
        writer = tf.summary.FileWriter(const.log_dir, sess.graph)
        
        merged = tf.summary.merge_all()
        loss = None
        summary = None
        #loss_test = None
        
        for e in range(const.max_iter):
            sess.run(tf.variables_initializer(tf.local_variables(), name='init_local'))
            
            # train
            sess.run(self.train_init_op)
            sess.run(self.reset_loss_op)
            while True:
                try:
                    ops = [self.accumulated_loss, self.accumulate_op, self.train_step]
                    loss, _, _ = sess.run(ops)
                    summary = sess.run(merged)
                    
                except tf.errors.OutOfRangeError:
                    break
                
            # evaluation
            sess.run(self.test_init_op)
            #sess.run(self.reset_loss_op_test)
            while True:
                try:
                    #ops = [self.accumulated_loss_test, self.accumulate_op_test]
                    #loss = 
                    sess.run(self.ops)
                    summary = sess.run(merged)
                except tf.errors.OutOfRangeError:
                    evaluations = sess.run(self.evals)
                    break

            epc, _ = sess.run([self.epochs, self.epoch_count_op])
            writer.add_summary(summary, epc)

            # set logging
            logs = " epoch {}: [train] loss = {:.6f}".format(epc, np.average(loss))
            logs += ",\t[test] accuracy = {:.6f}, precision = {:.6f}, recall = {:.6f}".format(*evaluations)
            tf.logging.info(logs)
            
            # save trained model
            saver = tf.train.Saver()
            saver.save(sess, os.path.join(const.log_dir, const.model_file_path))
            
    def evaluation(self, sess):
        sess.run(tf.variables_initializer(tf.local_variables(), name='init_local'))
        
        # evaluation
        sess.run(self.test_init_op)
        with open(const.predicted_path, 'w') as fout:
            fout.write("member_id, gender, pref_cd, predict, actual, score\n")
            
            while True:
                try:
                    indices, t_test, y_test, _ = sess.run([self.next_idx, self.t_te, self.y_te, self.ops])
                    data = self._data_te.iloc[indices].reindex(columns=['member_id', 'gender', 'pref_cd'])
                    data = data.assign(predict=np.argmax(y_test, axis=1), actual=t_test, score=y_test[:, 1])
                    for idx, row in data.iterrows():
                        fout.write('{}\n'.format(','.join(map(str, row.values))))
                        fout.flush()
                except tf.errors.OutOfRangeError:
                    break
                    
if __name__ == '__main__': 
    # 当スクリプトが実行されたときに以下が実行される
    tf.logging.set_verbosity(tf.logging.INFO)

    trainer = TrainerMLP()
    trainer.build_graphs()
    trainer.set_summary()

    # if os.path.exists(const.log_dir):
    #     for f in os.listdir(const.log_dir):
    #         if f.startswith('events.out.tfevents'):
    #             os.remove(os.path.join(const.log_dir, f))

    # 以下を入れないとGPU全て使ってしまうので必ず入れる
    # GPU config
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(visible_device_list="0", allow_growth=True))
    
    with tf.Session(config=config) as sess:
        trainer.initialize(sess)
        trainer.train(sess)
        trainer.evaluation(sess)