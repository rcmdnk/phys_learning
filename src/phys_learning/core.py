"""
    Core module to provides gcpm functions.
"""

import numpy as np
from sklearn.model_selection import train_test_split
from .__init__ import __version__
from .__init__ import __program__
from .phys import pt, mass, histTwoPhys
from .phys_model import PhysModel
from .model import n_input_check_wrapper


class PhysLearning():
    """Phys Learning"""

    def __init__(self, signal, bg, name='test', epochs=1000, verbose=1,
                 test_size=0.2):
        self.signal = PhysLearning.get_data(signal, 1)
        self.bg = PhysLearning.get_data(bg, 0)
        self.name = name
        self.epochs = epochs
        self.verbose = verbose

        data = np.concatenate([self.signal, self.bg])
        np.random.shuffle(data)
        x_data = data[:, 0:8]
        y_data = data[:, 8]
        self.x_train, self.x_test, self.y_train, self.y_test \
            = train_test_split(x_data, y_data, test_size=test_size)

    def set_verbose(self, verbose):
        self.verbose = verbose

    @staticmethod
    def help():
        print("""
Usage: phys_learning [--config=<config>] [--test=<test>] <command>

    commands:
        run        : Run user process.
        version    : Show version.
        help       : Show this help.
""")

    @staticmethod
    def version():
        print("%s: %s" % (__program__, __version__))

    def run(self, cmd=None):
        if cmd is None or cmd == "basic":
            self.basic_test()
        elif cmd == "device":
            self.show_devices()
        elif cmd == "basic_plot":
            self.basic_plot()
        elif cmd == "single":
            self.single()
        elif cmd == "oneshot":
            self.oneshot()
        elif cmd == "multishot":
            self.multishot()
        else:
            print("usage: phys_learning run --cmd=<subcmd>\n"
                  "\n"
                  "subcmd:\n"
                  "  device:    : Check device\n"
                  "  basic_plot : Make basic plots from input data\n"
                  "  basic      : basic_plot and basic tests\n"
                  "  single     : Tests on all single variable\n"
                  "  oneshot    : Test on random one combination\n"
                  "  multishot  : Tests on random 100 combinations\n"
                  )

    def show_devices(self):
        from tensorflow.python.client import device_lib
        print(device_lib.list_local_devices())

    def basic_plot(self):
        histTwoPhys(self.signal, self.bg, self.name + "_orig")

    def basic_test(self):
        self.basic_plot()
        self.direct()
        self.mass_only()
        self.mass_pt()

    def direct(self):
        n_input_check_wrapper(self.x_train, self.x_test, self.y_train,
                              self.y_test, self.x_test, node=8,
                              layer=1, verbose=self.verbose,
                              epochs=self.epochs,
                              name=self.name + "_direct")

    def mass_only(self):
        x_train = mass(self.x_train[:, 0:4], self.x_train[:, 4:8])
        x_test = mass(self.x_test[:, 0:4], self.x_test[:, 4:8])
        n_input_check_wrapper(x_train, x_test, self.y_train,
                              self.y_test, self.x_test, node=8,
                              layer=1, verbose=self.verbose,
                              epochs=self.epochs,
                              name=self.name + "_mass_only")

    def mass_pt(self):
        x_train = np.array([mass(self.x_train[:, 0:4], self.x_train[:, 4:8]),
                            pt(self.x_train[:, 0], self.x_train[:, 1]),
                            pt(self.x_train[:, 4], self.x_train[:, 5])]).T
        x_test = np.array([mass(self.x_test[:, 0:4], self.x_test[:, 4:8]),
                           pt(self.x_test[:, 0], self.x_test[:, 1]),
                           pt(self.x_test[:, 4], self.x_test[:, 5])]).T
        n_input_check_wrapper(x_train, x_test, self.y_train,
                              self.y_test, self.x_test, node=8,
                              layer=1, verbose=self.verbose,
                              epochs=self.epochs,
                              name=self.name + "_mass_pt")

    def single(self, verbose=None):
        if verbose is None:
            verbose = 0
        for i in range(8):
            x_train = self.x_train[:, i:i + 1]
            x_test = self.x_test[:, i:i + 1]
            acc = n_input_check_wrapper(
                x_train, x_test, self.y_train, self.y_test,
                self.x_test, node=8, layer=1, verbose=verbose,
                epochs=self.epochs, name=self.name + "_" + PhysModel.VALUES[i]
            )[1]
            print('{:.3f} {}'.format(acc, PhysModel.VALUES[i]))

    def oneshot(self, verbose=None):
        if verbose is None:
            verbose = 0
        n = np.random.randint(1, 16)
        pm = PhysModel(n)
        pm.make_rpn()
        x_train = pm.calc(self.x_train)
        x_test = pm.calc(self.x_test)
        acc = n_input_check_wrapper(x_train, x_test, self.y_train,
                                    self.y_test, self.x_test, node=8,
                                    layer=3, verbose=verbose,
                                    epochs=self.epochs,
                                    name=self.name + "_oneshot")[1]
        print('{:.3f} {}'.format(acc, pm.get_formula()))

    def multishot(self, n=100, verbose=None):
        for i in range(n):
            self.oneshot(verbose)

    @staticmethod
    def get_data(data, is_signal):
        if type(data) == str:
            with open(data) as f:
                lines = f.readlines()
            data = []
            for line in lines:
                line = line.strip()
                line = [float(x) for x in line.split()]
                if pt(line[0], line[1]) > pt(line[4], line[5]):
                    data.append(line + [is_signal])
                else:
                    data.append(line[4:8] + line[0:4] + [is_signal])
        return np.array(data)
