import numpy as np
from sklearn.model_selection import train_test_split
from .two_particles import TwoParticles
from .phys import hist_two_phys, mass, pt
from .formula import Formula


class PhysBinary():
    def __init__(self, signal, bg, name='test', test_size=0.2,
                 nvalue=3, shot=1000,
                 seed=None, model=None, method='DNN',
                 layers=[(8, 'sigmoid'), (8, 'sigmoid'), (8, 'sigmoid')],
                 activation_out='sigmoid',
                 optimizer='adam', loss='binary_crossentropy',
                 metrics=['accuracy'],
                 monitor='val_loss', patience=2,
                 epochs=1000, validation_split=0.1,
                 max_depth=3, n_estimators=50, learning_rate=1.0,
                 algorithm='SAMME.R',
                 verbose=0):
        self.name = name
        self.nvalue = nvalue
        self.shot = shot

        self.seed = seed
        self.model = model
        self.method = method

        self.layers = layers
        self.activation_out = activation_out
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.monitor = monitor
        self.patience = patience
        self.epochs = epochs
        self.validation_split = validation_split

        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.algorithm = algorithm

        self.hist = None
        self.score = None
        self.verbose = verbose

        self.signal = TwoParticles(signal, 1)
        self.bg = TwoParticles(bg, 0)

        self.test_size = test_size
        data = np.concatenate([self.signal.data, self.bg.data])
        self.x_data = data[:, 0:data[0].size - 1]
        self.y_data = data[:, data[0].size - 1:data[0].size]
        self.set_data()

        self.classifier = None

        self.cmd = {'original_hist': self.original_hist,
                    'direct': self.direct,
                    'mass': self.mass,
                    'mass_pt': self.mass_pt,
                    'single': self.single,
                    'random_shot': self.random_shot,
                    'multishot': self.multishot}

    def set_data(self):
        self.x_train, self.x_test, self.y_train, self.y_test \
            = train_test_split(self.x_data, self.y_data,
                               test_size=self.test_size, shuffle=True)
        self.y_train_array = np.array([x[0] for x in self.y_train])
        self.y_test_array = np.array([x[0] for x in self.y_test])

    def run(self, cmd):
        if cmd in self.cmd:
            if cmd == 'multishot':
                self.cmd[cmd]()
            else:
                if self.seed is None:
                    self.seed = 1
                for i in range(self.shot):
                    self.set_data()
                    self.cmd[cmd]()
                    self.seed += 1
        else:
            raise RuntimeError('Command: {} is not available'.format(cmd))

    def original_hist(self):
        hist_two_phys(self.signal.data, self.bg.data, self.name + "_original")

    def make_classifier(self, name, x_train, x_test):
        if self.method.lower() == 'dnn':
            from .classifier_dnn import DNN
            self.classifier = DNN(
                x_train=x_train, x_test=x_test,
                y_train=self.y_train, y_test=self.y_test,
                name=name, seed=self.seed, model=None,
                layers=self.layers,
                activation_out=self.activation_out,
                optimizer=self.optimizer, loss=self.loss,
                metrics=self.metrics,
                monitor=self.monitor, patience=self.patience,
                epochs=self.epochs,
                validation_split=self.validation_split,
                verbose=self.verbose)
        elif self.method.lower() in ('decisiontree', 'dt'):
            from .classifier_dt import DecisionTree
            self.classifier = DecisionTree(
                x_train=x_train, x_test=x_test,
                y_train=self.y_train, y_test=self.y_test,
                name=name, seed=self.seed, model=None,
                max_depth=self.max_depth, verbose=self.verbose)
        elif self.method.lower() in ('randomforest', 'rf'):
            from .classifier_rf import RandomForest
            self.classifier = RandomForest(
                x_train=x_train, x_test=x_test,
                y_train=self.y_train, y_test=self.y_test,
                name=name, seed=self.seed, model=None,
                max_depth=self.max_depth, verbose=self.verbose)
        elif self.method.lower() in ('ada', 'AdaBoost'):
            from .classifier_ada import AdaBoost
            self.classifier = AdaBoost(
                x_train=x_train, x_test=x_test,
                y_train=self.y_train_array, y_test=self.y_test_array,
                name=name, seed=self.seed, model=None,
                max_depth=self.max_depth, n_estimators=self.n_estimators,
                learning_rate=self.learning_rate, algorithm=self.algorithm,
                verbose=self.verbose)
        elif self.method.lower() in ('grad', 'gradientboosting'):
            from .classifier_grad import GradientBoosting
            self.classifier = GradientBoosting(
                x_train=x_train, x_test=x_test,
                y_train=self.y_train_array, y_test=self.y_test_array,
                name=name, seed=self.seed, model=None,
                max_depth=self.max_depth, n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                verbose=self.verbose)
        else:
            raise RuntimeError('Classifier method: {} is not available'.format(
                self.method))

    def direct(self):
        self.make_classifier(self.name + "_direct", self.x_train, self.x_test)
        acc = self.classifier.run_all()
        values = ', '.join(self.signal.var_labels)
        print('{:.3f} {}'.format(acc, values))
        return acc, values

    def mass(self):
        x_train = [[x] for x
                   in mass(self.x_train[:, 0:4], self.x_train[:, 4:8])]
        x_test = [[x] for x in mass(self.x_test[:, 0:4], self.x_test[:, 4:8])]
        self.make_classifier(self.name + "_mass", x_train, x_test)
        acc = self.classifier.run_all()
        values = 'm12'
        print('{:.3f} {}'.format(acc, values))
        return acc, values

    def mass_pt(self):
        x_train = np.array([mass(self.x_train[:, 0:4], self.x_train[:, 4:8]),
                            pt(self.x_train[:, 0], self.x_train[:, 1]),
                            pt(self.x_train[:, 4], self.x_train[:, 5])]).T
        x_test = np.array([mass(self.x_test[:, 0:4], self.x_test[:, 4:8]),
                           pt(self.x_test[:, 0], self.x_test[:, 1]),
                           pt(self.x_test[:, 4], self.x_test[:, 5])]).T
        self.make_classifier(self.name + "_mass_pt", self.x_train, self.x_test)
        acc = self.classifier.run_all()
        values = 'm12, pt1, pt2'
        print('{:.3f} {}'.format(acc, values))
        return acc, values

    def single(self):
        acc = []
        value = []
        for i in range(self.train[0].size):
            x_train = self.x_train[:, i:i + 1]
            x_test = self.x_test[:, i:i + 1]
            self.make_classifier(self.name + "_" + self.signal.var_labels[i],
                            x_train, x_test)
            acc.append(self.classifier.run_all())
            value.append(self.signal.var_labels[i])
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
                        var_labels=self.signal.var_labels))
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

    def prediction_check(self, model, x_test):
        predictions = model.predict(x_test)
        sig_sig = []
        sig_bg = []
        bg_sig = []
        bg_bg = []
        for i in range(predictions.size):
            if y_test[i] == 1:
                if predictions[i] > 0.5:
                    sig_sig.append(self.x_test[i])
                else:
                    sig_bg.append(self.x_test[i])
            elif predictions[i] > 0.5:
                bg_sig.append(self.x_test[i])
            else:
                bg_bg.append(self.x_test[i])
        sig_sig = np.array(sig_sig)
        sig_bg = np.array(sig_bg)
        bg_sig = np.array(bg_sig)
        bg_bg = np.array(bg_bg)
        hist_two_phys(sig_sig, sig_bg, name + "_sig_check")
        hist_two_phys(bg_sig, bg_bg, name + "_bg_check")
