import numpy as np
from var_learning.core import VarLearning
from var_learning.formula import Formula
from var_learning.plot import hist_two
from .two_particles import TwoParticles
from .phys import hist_two_phys, mass, pt


class PhysLearning(VarLearning):
    def __init__(self, signal=None, bg=None,
                 var_labels=["px1", "py1", "pz1", "e1",
                             "px2", "py2", "pz2", "e2"],
                 **kw):
        data = kw['data'] if 'data' in kw else None
        if data is None:
            if signal is not None:
                data = TwoParticles(signal, 1).data
            if bg is not None:
                bg = TwoParticles(bg, 0).data
                if data is None:
                    data = bg
                else:
                    data = np.concatenate([data, bg])

        super().__init__(data=data, var_labels=var_labels, **kw)

        self.cmd.update({'original_hist': self.original_hist,
                         'my_hist': self.my_hist,
                         'x1': self.x1,
                         'x2': self.x2,
                         'y1': self.y2,
                         'y2': self.y2,
                         'z1': self.z1,
                         'z2': self.z2,
                         'e1': self.e1,
                         'e2': self.e2,
                         'mass': self.mass,
                         'mass_pt': self.mass_pt})

    def get_signal(self):
        is_signal = np.array(self.data[:, -1], bool)
        return np.array(self.data[:, 0:-1])[is_signal]

    def get_bg(self):
        is_signal = np.array(self.data[:, -1], bool)
        return np.array(self.data[:, 0:-1])[~is_signal]

    def original_hist(self):
        hist_two_phys(self.get_signal(), self.get_bg(),
                      self.name + "_original")

    def my_hist(self):
        import json

        with open(self.json) as f:
            rpn = json.load(f)
        for i, r in enumerate(rpn):
            formula = Formula(8, var_labels=self.var_labels)
            formula.rpn = r
            var_signal = formula.calc(self.get_signal())
            var_bg = formula.calc(self.get_bg())
            sb = np.concatenate([var_signal, var_bg], 0)

            per10 = np.percentile(sb, 10)
            per90 = np.percentile(sb, 90)
            center = (per90 + per10) / 2
            xmin = 2 * per10 - center
            xmax = 2 * per90 - center

            xlabel = formula.get_formula()
            if len(xlabel) > 50:
                xlabel = xlabel[:47] + '...'

            hist_two(var_signal, var_bg, 100, [xmin, xmax],
                     '{}_{}'.format(self.name, i), xlabel=xlabel,
                     label1='signal', label2='bg')
            print('{}_{}'.format(self.name, i))
        # return pseudo values
        return None, None

    def x1(self):
        return self.one_var(0)

    def y1(self):
        return self.one_var(1)

    def z1(self):
        return self.one_var(2)

    def e1(self):
        return self.one_var(3)

    def x2(self):
        return self.one_var(4)

    def y2(self):
        return self.one_var(5)

    def z2(self):
        return self.one_var(6)

    def e2(self):
        return self.one_var(7)

    def one_var(self, i):
        x_train = self.x_train[:, i:i + 1]
        x_test = self.x_test[:, i:i + 1]
        self.make_classifier(self.name + "_" + self.formula.var_labels[i],
                             x_train, x_test)
        acc = self.classifier.run_all()
        value = self.formula.var_labels[i]
        print('{:.3f} {}'.format(acc, value))
        return acc, value

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
        self.make_classifier(self.name + "_mass_pt", x_train, x_test)
        acc = self.classifier.run_all()
        values = 'm12, pt1, pt2'
        print('{:.3f} {}'.format(acc, values))
        return acc, values

    def prediction_check(self, model, x_test):
        predictions = model.predict(x_test)
        sig_sig = []
        sig_bg = []
        bg_sig = []
        bg_bg = []
        for i in range(predictions.size):
            if self.y_test[i] == 1:
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
        hist_two_phys(sig_sig, sig_bg, self.name + "_sig_check")
        hist_two_phys(bg_sig, bg_bg, self.name + "_bg_check")
