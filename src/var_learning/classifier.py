import time
import numpy as np


class Classifier():
    def __init__(self, x_train=None, x_test=None, y_train=None, y_test=None,
                 name="test", model=None, seed=None,
                 use_gpu=True, gpu_device=0,
                 verbose=0):
        self.x_train = np.array(x_train)
        self.x_test = np.array(x_test)
        self.y_train = np.array(y_train)
        self.y_test = np.array(y_test)
        self.name = name
        self.model = model
        self.seed = seed
        self.use_gpu = use_gpu
        self.gpu_device = gpu_device
        self.acc = None
        self.verbose = verbose

    def new_model(self):
        self.model = 'Model'

    def model_info(self):
        print('binary classification, with seed={}'.format(self.seed))

    def make_model(self, force=0):
        if not force and self.model is not None:
            if self.verbose:
                print('Pre-made model:')
                self.model_info()
        else:
            self.new_model()
            if self.verbose:
                print('New model:')
                self.model_info()

    def learn(self):
        pass

    def run_test(self):
        self.acc = 'accuracy'
        if self.verbose > 0:
            print("test accuracy", self.acc)

    def run_all(self):
        if self.verbose > 0:
            t1 = time.time()
        self.make_model()
        self.learn()
        self.run_test()
        if self.verbose > 0:
            t2 = time.time()
            print("Time: {} sec".format(t2 - t1))
        return self.acc
