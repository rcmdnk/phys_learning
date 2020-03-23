from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from .classifier import Classifier


class DNN(Classifier):
    def __init__(self, layers=[(8, 'sigmoid'), (8, 'sigmoid'), (8, 'sigmoid')],
                 activation_out='sigmoid', optimizer='adam',
                 loss='binary_crossentropy', metrics=['accuracy'],
                 monitor='val_loss', patience=2, epochs=1000,
                 validation_split=0.1, **kw):
        super().__init__(**kw)
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

    def new_model(self):
        self.model = Sequential()
        if not self.layers:
            raise RuntimeError('BinaryDNN.layer can not be empty '
                               'if model is not defined')

        self.model.add(Dense(self.layers[0][0], input_dim=self.x_train[0].size,
                       activation=self.layers[0][1]))
        for layer in self.layers[1:]:
            self.model.add(Dense(layer[0], activation=layer[1]))

        self.model.add(Dense(1, activation=self.activation_out))

    def model_inof(self):
        self.model.summary()

    def learn(self):
        self.model.compile(optimizer=self.optimizer, loss=self.loss,
                           metrics=self.metrics)
        early_stopping = EarlyStopping(
            monitor=self.monitor, verbose=self.verbose, patience=self.patience)
        self.hist = self.model.fit(
            self.x_train, self.y_train, epochs=self.epochs,
            validation_split=self.validation_split, callbacks=[early_stopping],
            verbose=self.verbose)

    def run_test(self):
        score = self.model.evaluate(self.x_test, self.y_test,
                                    verbose=self.verbose)
        self.acc = score[1]
        if self.verbose > 0:
            print("test score", score[0])
            print("test accuracy", self.acc)
