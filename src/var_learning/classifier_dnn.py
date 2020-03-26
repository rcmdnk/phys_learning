from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
import tensorflow as tf
from .classifier import Classifier


class DNN(Classifier):
    def __init__(self, use_gpu=True, gpu_device=0,
                 layers=[(8, 'sigmoid'), (8, 'sigmoid'), (8, 'sigmoid')],
                 activation_out='sigmoid', optimizer='adam',
                 loss='binary_crossentropy', metrics=['accuracy'],
                 monitor='val_loss', patience=2, epochs=1000,
                 validation_split=0.1, **kw):
        super().__init__(**kw)
        self.use_gpu = use_gpu
        self.gpu_device = gpu_device
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

        gpu = tf.config.experimental.list_physical_devices('GPU')
        if len(gpu) > 0:
            if self.use_gpu:
                if self.gpu_device == "all":
                    tf.config.experimental.set_visible_devices(gpu, 'GPU')
                    for g in gpu:
                        tf.config.experimental.set_memory_growth(g, True)
                else:
                    if len(gpu) > self.gpu_device:
                        tf.config.experimental.set_visible_devices(
                            gpu[self.gpu_device], 'GPU')
                        tf.config.experimental.set_memory_growth(
                            gpu[self.gpu_device], True)
                    else:
                        tf.config.experimental.set_visible_devices(
                            gpu[0], 'GPU')
                        tf.config.experimental.set_memory_growth(gpu[0], True)
            else:
                tf.config.experimental.set_visible_devices([], 'GPU')

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
