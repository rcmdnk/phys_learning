import time
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping

from .plot import plotTwo
from .phys import histTwoPhys


def input_check(model, x_train, x_test, y_train, y_test,
                verbose=0, name='input_check',
                epochs=1000):
    if verbose > 0:
        model.summary()
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', verbose=verbose,
                                   patience=2)
    hist = model.fit(x_train, y_train,
                     epochs=epochs, verbose=verbose,
                     validation_split=0.1,
                     callbacks=[early_stopping])

    if verbose > 0:
        plotTwo(
            hist.epoch, hist.history["accuracy"],
            hist.epoch, hist.history["loss"],
            name=name + '_history', xlabel="Epoch", ylabel="",
            label1="accuracy", label2="loss")

    score = model.evaluate(x_test, y_test, verbose=verbose)
    if verbose > 0:
        print("test score", score[0])
        print("test accuracy", score[1])

    return score[1]


def n_input_check(x_train, x_test, y_train, y_test,
                  node=8, layer=1, verbose=0, name='input_check',
                  epochs=1000):
    model = Sequential()
    model.add(Dense(node, input_dim=x_train[0].size, activation='sigmoid'))
    if layer > 1:
        for i in range(1, layer):
            model.add(Dense(node, activation='sigmoid'))
    model.add(Dense(1, activation='sigmoid'))
    acc = input_check(model, x_train, x_test, y_train, y_test, verbose, name,
                      epochs)
    return model, acc


def prediction_check(model, x_test, y_test, x_test_full,
                     name='prediction_check'):
    predictions = model.predict(x_test)
    sig_sig = []
    sig_bg = []
    bg_sig = []
    bg_bg = []
    for i in range(predictions.size):
        if y_test[i] == 1:
            if predictions[i] > 0.5:
                sig_sig.append(x_test_full[i])
            else:
                sig_bg.append(x_test_full[i])
        elif predictions[i] > 0.5:
            bg_sig.append(x_test_full[i])
        else:
            bg_bg.append(x_test_full[i])
    sig_sig = np.array(sig_sig)
    sig_bg = np.array(sig_bg)
    bg_sig = np.array(bg_sig)
    bg_bg = np.array(bg_bg)
    histTwoPhys(sig_sig, sig_bg, name + "_sig_check")
    histTwoPhys(bg_sig, bg_bg, name + "_bg_check")


def n_input_check_wrapper(x_train, x_test, y_train, y_test, x_test_full,
                          node=8, layer=1, verbose=0, name='input_check',
                          epochs=1000):

    if verbose > 0:
        t1 = time.time()

    model, acc = n_input_check(x_train, x_test, y_train, y_test, node,
                               layer, verbose, name, epochs)
    if verbose > 0:
        prediction_check(model, x_test, y_test, x_test_full, name)

    if verbose > 0:
        t2 = time.time()
        print("Time: {} sec".format(t2 - t1))

    return model, acc
