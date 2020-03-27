import numpy as np
from sklearn.model_selection import train_test_split
from .formula import Formula
from .plot import hist_two


class VarLearning():
    def __init__(self, name='test', data=None, n_target=1,
                 test_size=0.2, var_labels=None, fix_dim=False,
                 nvalue=3, shot=1,
                 use_int=False, int_check=False,
                 method='Ada', seed=None, verbose=0, **kw):
        self.name = name
        self.n_target = n_target
        self.test_size = test_size
        self.var_labels = var_labels
        self.fix_dim = fix_dim

        self.nvalue = nvalue
        self.shot = shot
        self.use_int = use_int
        self.int_check = int_check
        self.method = method

        self.seed = seed
        self.verbose = verbose

        self.kw = kw

        self.hist = None
        self.score = None

        self.set_data(data)

        self.classifier = None

        self.cmd = {'direct': self.direct,
                    'single': self.single,
                    'random_shot': self.random_shot,
                    'multishot': self.multishot}

    def set_data(self, data):
        if data is None:
            self.data = None
            return
        if type(data) == str:
            with open(data) as f:
                data = [line.split() for line in f.readlines()]
        if self.use_int:
            self.data = np.array(data, int)
        else:
            self.data = np.array(data)
        self.separate_data()

        self.formula = Formula(n_values=self.data[0].size - self.n_target,
                               var_labels=self.var_labels,
                               fix_dim=self.fix_dim)

    def separate_data(self):
        x_data = self.data[:, 0: -1 * self.n_target]
        y_data = self.data[:, -1 * self.n_target:self.data[0].size]

        self.x_train, self.x_test, self.y_train, self.y_test \
            = train_test_split(x_data, y_data,
                               test_size=self.test_size, shuffle=True)
        self.y_train_array = np.array([x[0] for x in self.y_train])
        self.y_test_array = np.array([x[0] for x in self.y_test])

    def run(self, cmd):
        if not self.int_check:
            return self.run_base(cmd)
        else:
            self.use_int = False
            acc_non_int = self.run_base(cmd)
            self.use_int = True
            acc_int = self.run_base(cmd)
            hist_two(acc_non_int, acc_int, bins=100, range=None,
                     name=self.name + '_int_check', xlabel='accuracy',
                     label1='Original', label2='Int')

    def run_base(self, cmd):
        if cmd in self.cmd:
            if cmd == 'multishot':
                self.cmd[cmd]()
            else:
                if self.seed is None:
                    self.seed = 1
                acc = []
                for i in range(self.shot):
                    self.separate_data()
                    acc.append(self.cmd[cmd]()[0])
                    self.seed += 1
                return acc
        else:
            raise RuntimeError('Command: {} is not available'.format(cmd))

    def make_classifier(self, name, x_train, x_test):
        y_train = self.y_train_array
        y_test = self.y_test_array
        if self.method.lower() == 'dnn':
            from var_learning.classifier_dnn import DNN as Classifier
            y_train = self.y_train
            y_test = self.y_test
        elif self.method.lower() in ('decisiontree', 'dt'):
            from var_learning.classifier_dt import DecisionTree as Classifier
        elif self.method.lower() in ('randomforest', 'rf'):
            from var_learning.classifier_rf import RandomForest as Classifier
        elif self.method.lower() in ('ada', 'AdaBoost'):
            from var_learning.classifier_ada import AdaBoost as Classifier
        elif self.method.lower() in ('grad', 'gradientboosting'):
            from var_learning.classifier_grad import GradientBoosting as \
                Classifier
        else:
            raise RuntimeError('Classifier method: {} is not available'.format(
                self.method))
        self.classifier = Classifier(
            x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test,
            name=name, seed=self.seed, verbose=self.verbose,
            **self.kw)

    def direct(self):
        self.make_classifier(self.name + "_direct", self.x_train, self.x_test)
        acc = self.classifier.run_all()
        values = ', '.join(self.formula.var_labels)
        print('{:.3f} {}'.format(acc, values))
        return acc, values


    def single(self):
        acc = []
        values = []
        for i in range(self.x_train[0].size):
            x_train = self.x_train[:, i:i + 1]
            x_test = self.x_test[:, i:i + 1]
            self.make_classifier(self.name + "_" + self.formula.var_labels[i],
                                 x_train, x_test)
            acc.append(self.classifier.run_all())
            values.append(self.formula.var_labels[i])
            print('{:.3f} {}'.format(acc[-1], values[-1]))
        return acc, values

    def random_shot(self):
        x_train = []
        x_test = []
        formula = []

        for i in range(self.nvalue):
            formula.append(
                Formula(n_values=self.x_train[0].size, min_use=1,
                        max_use=self.x_train[0].size * 2,
                        var_labels=self.formula.var_labels,
                        fix_dim=self.fix_dim))
            formula[-1].make_rpn()
            x_train.append(formula[-1].calc(self.x_train))
            x_test.append(formula[-1].calc(self.x_test))

        x_train = np.concatenate(x_train, 1)
        x_test = np.concatenate(x_test, 1)

        self.make_classifier(self.name + "_random", x_train, x_test)
        acc = self.classifier.run_all()

        values = '{} {}'.format([f.rpn for f in formula],
                                [f.get_formula() for f in formula])
        if self.verbose:
            print('{:.3f} {}'.format(acc, values))
        return acc, values, formula

    def multishot(self):
        import datetime
        print('{} Start multishot: shot={}, nvalue={}'.format(
            datetime.datetime.now(), self.shot, self.nvalue))
        top_history = []
        model_list = []
        for i in range(self.shot):
            acc, values, formula = self.random_shot()
            if len(model_list) < 5:
                model_list.append((acc, values, formula))
                model_list.sort(key=lambda x: -x[0])
            else:
                if model_list[-1][0] < acc:
                    model_list.pop()
                    model_list.append((acc, values, formula))
                    model_list.sort(key=lambda x: -x[0])
            if i % 100 == 0:
                top_history.append(model_list[0][0])
                print(datetime.datetime.now())
                print("Top accuracy history: ", end='')
                for h in top_history:
                    print('{:.3f}, '.format(h), end='')
                print('')
                print("Top 5 value combination list at {}".format(i))
                for x in model_list:
                    print('{:.3f}: '.format(x[0]), end='')
                    for f in x[2]:
                        print('{}, '.format(f.rpn), end='')
                    for f in x[2]:
                        print('{}, '.format(f.get_formula()), end='')
                    print('')
