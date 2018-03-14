# coding: utf-8

from configparser import ConfigParser
import ast


class Consts(object):
    def __init__(self, path):
        const = ConfigParser()
        const.read(path)

        # data info
        self.data_path = const.get('data', 'data_path') # data_path = haruyama_repeat_predict_data.csv
        self.n_window = ast.literal_eval(const.get('data', 'n_window')) # n_window = 5
        self.split_date = const.get('data', 'split_date') # split_date = 2017-11-19

        # train info
        self.batch_size = ast.literal_eval(const.get('train', 'batch_size')) # batch_size = 10000
        self.layers = ast.literal_eval(const.get('train', 'layers')) # layers = [128, 128, 128, 2]
        self.tr_ratio = ast.literal_eval(const.get('train', 'train_test_ratio')) # train_test_ratio = 0.8
        self.init_lr = ast.literal_eval(const.get('train', 'init_lr')) # 1.e-4
        self.max_iter = ast.literal_eval(const.get('train', 'max_iteration')) # max_iteration = 100
        self.log_dir = const.get('train', 'log_directory') # log_directory = ./summary
        self.model_file_path = const.get('train', 'model_file_path') # model_file_path = predict_mlp.ckpt
        self.predicted_path = const.get('train', 'predict_path') # predict_path = predicted_result.csv


constants = Consts('config.ini')
