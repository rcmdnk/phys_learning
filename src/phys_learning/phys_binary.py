import numpy as np
from sklearn.model_selection import train_test_split
from .two_particles import TwoParticles
from .phys import hist_two_phys, mass, pt
from .binary_dnn import BinaryDNN
from .formula import Formula


class PhysBinary():
    def __init__(self, signal, bg, name='test', test_size=0.2,
                 nvalue=3, shot=1000,
                 seed=None, model=None,
                 layers=[(8, 'sigmoid'), (8, 'sigmoid'), (8, 'sigmoid')],
                 activation_out='sigmoid',
                 optimizer='adam', loss='binary_crossentropy',
                 metrics=['accuracy'],
                 monitor='val_loss', patience=2,
                 epochs=1000, validation_split=0.1,
                 verbose=0):
        self.name = name
        self.nvalue = nvalue
        self.shot = shot

        self.model = model
        self.layers = layers
        self.activation_out = activation_out
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.monitor = monitor
        self.patience = patience
        self.epochs = epochs
        self.validation_split = validation_split
        self.hist = None
        self.score = None
        self.verbose = verbose

        self.signal = TwoParticles(signal, 1)
        self.bg = TwoParticles(bg, 0)

        data = np.concatenate([self.signal.data, self.bg.data])
        np.random.shuffle(data)
        x_data = data[:, 0:data[0].size - 1]
        y_data = data[:, data[0].size - 1:data[0].size]
        self.x_train, self.x_test, self.y_train, self.y_test \
            = train_test_split(x_data, y_data, test_size=test_size)

        self.bdnn = None

        self.cmd = {'original_hist': self.original_hist,
                    'direct': self.direct,
                    'mass': self.mass,
                    'mass_pt': self.mass_pt,
                    'single': self.single,
                    'random_shot': self.random_shot,
                    'multishot': self.multishot}

    def run(self, cmd):
        if cmd in self.cmd:
            self.cmd[cmd]()
        else:
            raise RuntimeError('Command: {} is not available'.format(cmd))

    def original_hist(self):
        hist_two_phys(self.signal.data, self.bg.data, self.name + "_original")

    def make_bdnn(self, name, x_train, x_test):
        self.bdnn = BinaryDNN(x_train=x_train, x_test=x_test,
                              y_train=self.y_train, y_test=self.y_test,
                              name=name, model=None,
                              layers=self.layers,
                              activation_out=self.activation_out,
                              optimizer=self.optimizer, loss=self.loss,
                              metrics=self.metrics,
                              monitor=self.monitor, patience=self.patience,
                              epochs=self.epochs,
                              validation_split=self.validation_split,
                              verbose=self.verbose)

    def direct(self):
        self.make_bdnn(self.name + "_direct", self.x_train, self.x_test)
        score = self.bdnn.run_all()
        values = ', '.join(self.signal.var_labels)
        if self.verbose:
            print('{:.3f} {}'.format(score[1], values))
        return score, values

    def mass(self):
        x_train = mass(self.x_train[:, 0:4], self.x_train[:, 4:8])
        x_test = mass(self.x_test[:, 0:4], self.x_test[:, 4:8])
        self.make_bdnn(self.name + "_mass", x_train, x_test)
        score = self.bdnn.run_all()
        values = 'm12'
        if self.verbose:
            print('{:.3f} {}'.format(score[1], values))
        return score, values

    def mass_pt(self):
        x_train = np.array([mass(self.x_train[:, 0:4], self.x_train[:, 4:8]),
                            pt(self.x_train[:, 0], self.x_train[:, 1]),
                            pt(self.x_train[:, 4], self.x_train[:, 5])]).T
        x_test = np.array([mass(self.x_test[:, 0:4], self.x_test[:, 4:8]),
                           pt(self.x_test[:, 0], self.x_test[:, 1]),
                           pt(self.x_test[:, 4], self.x_test[:, 5])]).T
        self.make_bdnn(self.name + "_mass_pt", self.x_train, self.x_test)
        score = self.bdnn.run_all()
        values = 'm12, pt1, pt2'
        if self.verbose:
            print('{:.3f} {}'.format(score[1], values))
        return score, values

    def single(self):
        score = []
        value = []
        for i in range(self.train[0].size):
            x_train = self.x_train[:, i:i + 1]
            x_test = self.x_test[:, i:i + 1]
            self.make_bdnn(self.name + "_" + self.signal.var_labels[i],
                           x_train, x_test)
            score.append(self.bdnn.run_all())
            value.append(self.signal.var_labels[i])
            if self.verbose:
                print('{:.3f} {}'.format(score[-1][1], values[-1]))
        return score, values

    def random_shot(self):
        x_train = []
        x_test = []
        formula = []

        for i in range(self.nvalue):
            formula.append(
                Formula(n_values=self.train[0].size, min_use=1,
                        max_use=self.train[0].size * 2,
                        var_labels=self.signal.var_labels))
            formula[-1].make_rpn()
            x_train.append(formula[-1].calc(self.x_train))
            x_test.append(formula[-1].calc(self.x_test))

        x_train = np.concatenate(x_train, 1)
        x_test = np.concatenate(x_test, 1)

        self.make_bdnn(self.name + "_random", x_train, x_test)
        score = self.bdnn.run_all()

        values = '{} {}'.format([f.rpn() for f in formula],
                                [f.formula() for f in formula])
        if self.verbose:
            print('{:.3f} {}'.format(score[1], values))
        return score, values, formula

    def multishot(self):
        import datetime
        print('{} Start multishot: shot={}, nvalue={}, max_val={}'.format(
            datetime.datetime.now(), self.shot, self.nvalue))
        top_history = []
        model_list = []
        for i in range(shot):
            score, values, formula = self.random_shot(self.nvalue)
            if len(model_list) < 5:
                model_list.append((score[1], values, formula))
                model_list.sort(key=lambda x: -x[0])
            else:
                if model_list[-1][0] < score[1]:
                    model_list.pop()
                    model_list.append((score[1], values, formula))
                    model_list.sort(key=lambda x: -x[0])
            if i != 0 and i % 100 == 0:
                top_history.append(model_list[0][0])
                print(datetime.datetime.now())
                print("Top accuracy history: ", end='')
                for h in top_history:
                    print('{:.3f}, '.format(h), end='')
                print('')
                print("{} Top 5 value combination list at {}".format(
                    datetime.datetime.now(), i))
                for x in model_list:
                    print('{:.3f}: '.format(x[0]), end='')
                    for f in x[2]:
                        print('{}, '.format(f.rpn), end='')
                    for f in x[2]:
                        print('{}, '.format(f.formula), end='')
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
