# -*- coding: utf-8 -*-

import os
import warnings
import pickle
import signal, sys
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf

########################################################################################################################


def print_b(text, bold=False):
    if bold:
        print(BColors.BOLD + BColors.OKBLUE + str(text) + BColors.ENDC + BColors.ENDC)
    else:
        print(BColors.OKBLUE + str(text) + BColors.ENDC)


def print_g(text, title=True):
    title = "[INFO] " if title else ""
    print(BColors.OKGREEN + title + str(text) + BColors.ENDC)


def to_pickle(path, name, data):
    with open(path+name, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def get_pickle(path, name):
    with open(path+name, 'rb') as handle:
        data = pickle.load(handle)
    return data


def plot_hist(data, column, title="Histogram", title_x="X Axis", title_y="Y Axis", bins=10, save=None):

    plt.ioff()

    items = bins

    plt.hist(data[str(column)], bins=range(1, items + 2), edgecolor='black',
             align="left")  # arguments are passed to np.histogram
    labels = list(map(lambda x: str(x), range(1, items + 1)))
    labels[-1] = "≥" + labels[-1]
    plt.xticks(range(1, items + 1), labels)
    plt.title(str(title))

    plt.xlabel(title_x)
    plt.ylabel(title_y)

    if save is None:
        plt.show()
    else:
        plt.savefig(str(save))

    plt.close()


def print_e(text):
    print(BColors.FAIL + str("[ERROR] ") + str(text) + BColors.ENDC)


def print_w(text, title=True):
    title = "[WARNING] " if title else ""
    print(BColors.WARNING + title + str(text) + BColors.ENDC)


########################################################################################################################


class BColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class ModelClass:

    def __init__(self, city, config, model_name="", seed = 2, load=None):

        signal.signal(signal.SIGINT, self.signal_handler)

        self.CITY = city
        self.PATH = "data/" + self.CITY.lower().replace(" ", "") + "/"
        self.DATA_PATH = self.PATH+model_name.upper()+"/"

        self.IMG_PATH = self.PATH + "images_lowres/"
        self.SEED = seed
        self.CONFIG = config
        self.MODEL_NAME = model_name

        # Hide tf info
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        os.environ['PYTHONHASHSEED'] = '0'

        warnings.filterwarnings('always')
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # Set tf, random and np seeds
        np.random.seed(self.SEED)
        random.seed(self.SEED)
        tf.compat.v1.set_random_seed(self.SEED)

        self.DATA = self.get_data(load=load)

        self.MODEL_PATH = "models/"+self.MODEL_NAME+"_" + self.CITY.lower().replace(" ", "")
        self.SESSION = None

    def signal_handler(self,signal, frame):
        self.stop()
        sys.exit(0)

    def get_model(self):
        raise NotImplementedError

    def get_filtered_data(self, verbose=True):
        raise NotImplementedError

    def get_data(self, load):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def dev(self):
        raise NotImplementedError

    def test(self):
        raise NotImplementedError

    def grid_search_print(self, epoch, train, dev):
        raise NotImplementedError

    def grid_search(self, params, max_epochs = 50, start_n_epochs = 5, last_n_epochs = 5):
        raise NotImplementedError

    def final_train(self, epochs = 1, save=False):
        raise NotImplementedError

    def print_config(self, filter_dt=[]):

        tmp = self.CONFIG
        tmp['seed'] = self.SEED
        tmp['city'] = self.CITY

        print_g("-" * 50, title=False)

        for key, value in tmp.items():

            line = BColors.BOLD + key + ": " + BColors.ENDC + str(value)

            if len(filter_dt)>0:
                if key in filter_dt:
                    print_b(line)

            else:
                print(line)

        print_g("-" * 50, title=False)


