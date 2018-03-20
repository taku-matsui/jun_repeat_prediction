# coding: utf-8
#import pandas as pd
import tensorflow as tf
import sonnet as snt
from constants import constants

#batch_size = 10000
#n_window = 3 # 3日間とる
#data_path = 'jun_train_data_6months.csv'
#data = pd.read_csv(data_path).iloc[:,1:]
#data_num = data.shape[0]
#split_date = '2017-05-01'

class DataSupplierMLP(snt.AbstractModule):
    def __init__(self, name, data):
        
        self._batch_size = constants.batch_size
        self._n_window = constants.n_window
        
        super(DataSupplierMLP, self).__init__(name=name)
        
        self.data = data
        self._data_num = self.data.shape[0]
        
        # Dataset
        self.data_idx = tf.data.Dataset.range(self._data_num).shuffle(self._data_num).batch(self._batch_size)

        # Input data
        cols = []
        for w in range(self._n_window):
            cols += ['days_delta_{}'.format(w + 1),
                     'age_{}'.format(w + 1),
                     'shop_id_{}'.format(w + 1), # shop_idを識別子のままでいいのかあとで要検討
                     'item_num_{}'.format(w + 1),
                    'list_price_sum_{}'.format(w + 1)]
        cols += ['gender', 'pref_cd']

        # ここで何を入力（ノード）にするかを決めている
        self._x = [tf.constant(self.data[c].values) for c in cols]

        # teacher data
        # ここで何を出力（ノード）にするか決めている
        self._t = tf.constant(self.data['bought'].values)

    def _build(self, indices):
        # たくさんあるデータのうちindicesだけとってきて集めてリストを返す
        batch_x = [tf.gather(xi, indices, name='make_x_batch') for xi in self._x] # data11, data32, data24, ...みたいな
        batch_t = tf.gather(self._t, indices, name='make_t_bath') # 010111010101100みたいな

        return batch_x, batch_t

    @staticmethod
    def loss(logits, target, output_size):
        with tf.name_scope('softmax_w/_CrossEntropy'):
            one_hot_target = tf.one_hot(tf.to_int32(target, name='to_int32_one_hot'),
                                        depth=output_size, dtype=tf.float32)
            v = tf.losses.softmax_cross_entropy(one_hot_target, logits)
        return v

    @staticmethod
    def evaluation(logits, target):
        evals = [] # accuracy, precision, recall の値そのもの
        ops = [] # accuracy, precision, recall の update_ops

        # tf.argmaxで最も確からしいラベルを返す
        prediction = tf.argmax(logits, 1, name='argmax_y')

        e, o = tf.metrics.accuracy(target, prediction)
        evals.append(e)
        ops.append(o)

        e, o = tf.metrics.precision(target, prediction)
        evals.append(e)
        ops.append(o)

        e, o = tf.metrics.recall(target, prediction)
        evals.append(e)
        ops.append(o)

        return evals, ops
