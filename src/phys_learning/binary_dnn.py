import time
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping


class BinaryDNN():
    def __init__(self, x_train=None, x_test=None, y_train=None, y_test=None,
                 name="test", model=None,
                 layers=[(8, 'sigmoid'), (8, 'sigmoid'), (8, 'sigmoid')],
                 activation_out='sigmoid',
                 optimizer='adam', loss='binary_crossentropy',
                 metrics=['accuracy'],
                 monitor='val_loss', patience=2,
                 epochs=1000, validation_split=0.1,
                 verbose=0):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.name = name
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

    def make_model(self, force=0):
        if not force and self.model is not None:
            if self.verbose:
                print('Pre-made model:')
                self.model.summary()
            return self.model

        self.model = Sequential()
        if not self.layers:
            raise RuntimeError('BinaryDNN.layer can not be empty '
                               'if model is not defined')

        self.model.add(Dense(self.layers[0][0], input_dim=self.x_train[0].size,
                       activation=self.layers[0][1]))
        for layer in self.layers[1:]:
            self.model.add(Dense(layer[0], activation=layer[1]))

        self.model.add(Dense(1, activation=self.activation_out))
        if self.verbose:
            print('New model:')
            self.model.summary()
        return self.model

    def learn(self):
        self.model.compile(optimizer=self.optimizer, loss=self.loss,
                           metrics=self.metrics)
        early_stopping = EarlyStopping(
            monitor=self.monitor, verbose=self.verbose, patience=self.patience)
        self.hist = self.model.fit(
            self.x_train, self.y_train, epochs=self.epochs,
            validation_split=self.validation_split, callbacks=[early_stopping],
            verbose=self.verbose)
        return self.hist

    def run_test(self):
        self.score = self.model.evaluate(self.x_test, self.y_test,
                                         verbose=self.verbose)
        if self.verbose > 0:
            print("test score", self.score[0])
            print("test accuracy", self.score[1])
        return self.score

    def run_all(self):
        if self.verbose > 0:
            t1 = time.time()
        self.make_model()
        self.learn()
        self.run_test()
        if self.verbose > 0:
            t2 = time.time()
            print("Time: {} sec".format(t2 - t1))
        return self.score
