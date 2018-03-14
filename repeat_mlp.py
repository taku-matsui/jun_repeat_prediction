# coding: utf-8
import tensorflow as tf
from tensorflow.contrib.lookup import HashTable, KeyValueTensorInitializer
import sonnet as snt
from constants import constants
#layers = [128, 128, 128, 2]
#n_window = 3

class RepeatMLP(snt.AbstractModule):
    def __init__(self, name, shop_id, gender, pref_cd):
        super(RepeatMLP, self).__init__(name=name) # name = 'repeat_mlp'
        self._shop_id_labels = shop_id['labels'] # shop_id
        self._gender_labels = gender['labels'] # gender
        self._pref_cd_labels = pref_cd['labels'] # pref_cd
        self._layers = constants.layers
        self._window = constants.n_window
        
        self._hash_shop_id = None
        self._hash_gender = None
        self._hash_pref_cd = None
        
        ls = [] # Linear
        bns = [] # BatchNorm
        
        with self._enter_variable_scope():
            # shop_id, gender, pref_cd （識別子ID)をEmbedする
            self._emb_shop_id = snt.Embed(vocab_size=len(self._shop_id_labels), embed_dim=shop_id['dim'], name='shop_id_embed')
            self._emb_gender = snt.Embed(vocab_size=len(self._gender_labels), embed_dim=gender['dim'], name='gender_embed')
            self._emb_pref_cd = snt.Embed(vocab_size=len(self._pref_cd_labels), embed_dim=pref_cd['dim'], name='pref_cd_embed')
            
            # 隠れ層のlinear_1, 2, 3とbatch_norm_1, 2, 3をそれぞれリストls, mnsに入れる
            for i, l in enumerate(self._layers[:-1]): # l = 128, 128, 128
                ls.append(snt.Linear(output_size=l, name='linear_{}'.format(i + 1)))
                bns.append(snt.BatchNorm(name='batch_norm_{}'.format(i + 1)))
            
            # 出力層（ノード数2）をリストlsに入れる
            ls.append(snt.Linear(output_size=self._layers[-1], name='linear_{}'.format(len(self._layers))))
            
            self._linears = tuple(ls)
            self._batch_norms = tuple(bns)
            
    def _get_unit_feature(self, x, times):
        # 入力1：x ----> days_delta, age, shop_id, item_num
        # 入力2：times（購入日を1として何回前の来店か）
        # 出力：入力に活性化関数ReLUをかけた後の値
        
        with tf.name_scope('unit_feature_{}_ago'.format(times)):
            day, age, shop_id, item = x
            # 識別子IDでないもの
            day_age_item = tf.concat([tf.expand_dims(day, 1, name='expand_days'),
                                     tf.expand_dims(age, 1, name='expand_age'),
                                     tf.expand_dims(item, 1, name='expand_item')], axis=1)
            # 識別子のもの
            idx_shop_id = self._hash_shop_id.lookup(shop_id)
            
            # 活性化関数（ReLU）をかける
            h_shop_id = tf.nn.relu(self._emb_shop_id(idx_shop_id))
            
            # 全ての入力データを連結
            h = tf.concat([h_shop_id, day_age_item], axis=1)
            
            return h
    def _build(self, x, is_train=False):
        # x = [ d1, age1, shop_id1, item_num1,
        #       d2, age2, shop_id2, item_num2,
        #       d3, age3, shop_id3, item_num3,
        #       gender,
        #       pref_cd
        #     ]
        
        idx_gender = self._hash_gender.lookup(x[-2]) # ここに関しては不明？？？？？
        h_gender = tf.one_hot(idx_gender, depth=len(self._gender_labels), dtype=tf.float32, name="gender_one_hot")
        
        idx_pref_cd = self._hash_pref_cd.lookup(x[-1])
        h_pref_cd = tf.one_hot(idx_pref_cd, depth=len(self._pref_cd_labels), dtype=tf.float32, name="pref_cd_one_hot")
        
        hs = [self._get_unit_feature(x[w * 4: (w + 1) * 4], w + 1) for w in range(self._window)] + [h_gender] + [h_pref_cd]
        
        # gender, pref_cd以外に活性化関数をかけたEmbed + gender (one-hot) + pref_cd (one-hot)
        h = tf.concat(hs, axis=1)
        
        with tf.name_scope('MLP'):
            for l, b in zip(self._linears, self._batch_norms):
                h = tf.nn.relu(l(h)) # concatされたデータに対してReLUをかける
                h = b(h, is_train) # BatchNorm
            h = self._linears[-1](h) # 出力層
        
        if not is_train:
            # 推論phaseではソフトマックスのみ
            h = tf.nn.softmax(h)
        return h
    
    def setup_hash_tables(self):
        # 番号がかぶらないように重複のない番号を割り当てる
        with tf.name_scope('hash_tables'):
            self._hash_shop_id = HashTable(KeyValueTensorInitializer(keys=self._shop_id_labels,
                                                                    values=tuple(range(len(self._shop_id_labels))),
                                                                    key_dtype=tf.string,
                                                                    name='shop_id_hash_init'),
                                          default_value=-1, name='shop_id_hash')
            self._hash_gender = HashTable(KeyValueTensorInitializer(keys=self._gender_labels,
                                                                    values=tuple(range(len(self._gender_labels))),
                                                                    key_dtype=tf.string,
                                                                    name='gender_hash_init'),
                                          default_value=-1, name='gender_hash')
            self._hash_pref_cd = HashTable(KeyValueTensorInitializer(keys=self._pref_cd_labels,
                                                                    values=tuple(range(len(self._pref_cd_labels))),
                                                                    key_dtype=tf.string,
                                                                    name='pref_cd_hash_init'),
                                          default_value=-1, name='pref_cd_hash')
            
            
    def init_hash_tables(self, sess):
        # ハッシュテーブルの初期化を行う
        sess.run(self._hash_shop_id.init)
        sess.run(self._hash_gender.init)
        sess.run(self._hash_pref_cd.init)