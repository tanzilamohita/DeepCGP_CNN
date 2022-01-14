# ==============================
# Tanzila Islam
# PhD Student, Iwate University
# Email: tanzilamohita@gmail.com
# Created Date: 1/10/2022
# ===============================
import tensorflow
from tensorflow import keras
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import numpy as np
import os
from matplotlib import pyplot as plt
# import glob
# import matplotlib.pyplot as plt
import pandas as pd
# from natsort import natsorted
# from keras.utils.vis_utils import plot_model
# from sklearn.metrics import mean_squared_error, make_scorer
import time


# keras items
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Flatten, Conv1D, MaxPooling1D, LSTM #CNNs
from tensorflow.python.keras.layers import LeakyReLU

import keras_tuner
from keras_tuner.tuners import RandomSearch
from keras_tuner.engine.hyperparameters import HyperParameters

# using GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# model will be trained on GPU 0,
# if it is -1, GPU will not use for training
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# load Processed Data
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz/bin/'

start_time = time.time()
X = pd.read_csv('../C7AIR_CNN_Data/C7AIR_X_C1.csv', header=None)
X.columns = np.arange(0, len(X.columns))
Y = pd.read_csv('../C7AIR_CNN_Data/C7AIR_Y.csv').iloc[0:, 1:]
Y.columns = np.arange(0, len(Y.columns))
print(X.shape)
# print(Y)
# data partitioning into train and validation
itrait = 0  # first trait analyzed
# print(Y[itrait])
X_train, X_valid, y_train, y_valid = train_test_split(X, Y[itrait], test_size=0.2)
print(X_train.shape, y_train.shape)
print(X_valid.shape, y_valid.shape)

X2_train = np.expand_dims(X_train, axis=2)
X2_valid = np.expand_dims(X_valid, axis=2)

def build_model(hp):  # random search passes this hyperparameter() object
    nSNP = X_train.shape[1]
    nStride = 3  # stride between convolutions

    model = keras.models.Sequential()

    model.add(Conv1D(hp.Int('input_units',
                            min_value=32,
                            max_value=256,
                            step=32), nStride, input_shape=(nSNP, 1)))

    model.add(Activation('relu'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(rate=0.5))
    model.add(MaxPooling1D(pool_size=2))

    for i in range(hp.Int('n_layers', 1, 5)):  # adding variation of layers.
        model.add(Conv1D(hp.Int(f'conv_{i}_units',
                                min_value=32,
                                max_value=256,
                                step=32), nStride))
        model.add(Activation('relu'))

    # model.add(Flatten())
    # model.add(Dense(10))
    model.add(Flatten())
    model.add(Dense(128, activation='linear'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dense(1))
    # model.add(Activation(hp.Choice('Activation', values=['relu', 'sigmoid', 'linear'])))

    # Tune the learning rate for the optimizer
    # Choose an optimal value from 0.01, 0.001, or 0.0001
    hp_learning_rate = hp.Choice('learning_rate',
            values=[1E-0, 1E-1, 1E-2, 1E-3, 1E-4, 1E-5, 1E-6, 1E-7])
    hp_optimizer = hp.Choice('optimizer', values=['SGD', 'Adam'])
    adm = keras.optimizers.Adam(learning_rate=hp_learning_rate)
    model.compile(optimizer=adm, loss='mse')

    return model


tuner = keras_tuner.Hyperband(build_model,
                     objective='val_loss',
                     max_epochs=10,
                     factor=3)
                     # directory='my_dir',
                     # project_name='intro_to_kt')

stop_early = tensorflow.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
tuner.search(x=X2_train,
             y=y_train,
             verbose=2,     # just slapping this here bc jupyter notebook. The console out was getting messy.
             epochs=50,
             batch_size=450,
             #callbacks=[tensorboard],  # if you have callbacks like tensorboard, they go here.
             validation_data=(X2_valid, y_valid),
             callbacks=[stop_early])

# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
The hyper-parameter search is complete. 
The optimal number of units in the first densely-connected
layer is {best_hps.get('input_units')} and the optimal learning rate for the optimizer
is {best_hps.get('learning_rate')}.
""")

print(tuner.get_best_hyperparameters()[0].values)
tuner.get_best_models()[0].summary()

model = tuner.hypermodel.build(best_hps)
history = model.fit(X2_train, y_train, epochs=400,
                    batch_size=450, validation_data=(X2_valid, y_valid))

val_loss_per_epoch = history.history['val_loss']
best_epoch = val_loss_per_epoch.index(max(val_loss_per_epoch)) + 1
print('Best epoch: %d' % (best_epoch,))

hypermodel = tuner.hypermodel.build(best_hps)

# Retrain the model
hypermodel.fit(X2_train, y_train, epochs=best_epoch, validation_data=(X2_valid, y_valid))

mse_prediction = hypermodel.evaluate(X2_valid, y_valid, batch_size=450)
print('\nMSE in prediction =', mse_prediction)

# get predicted target values
y_hat = hypermodel.predict(X2_valid, batch_size=450)
np.seterr(divide='ignore', invalid='ignore')
corr = np.corrcoef(y_valid, y_hat[:, 0])[0, 1]
print('\nCorr obs vs pred =', corr)

# plot observed vs. predicted targets
plt.title('CNN: Observed vs Predicted Y')
plt.ylabel('Predicted')
plt.xlabel('Observed')
plt.scatter(y_valid, y_hat, marker='o')
# obtain m (slope) and b(intercept) of linear regression line
m, b = np.polyfit(y_valid, y_hat, 1)

# add linear regression line to scatterplot
plt.plot(y_valid, m * y_valid + b)
plt.show()
plt.clf()

print('Total Time: ', time.time() - start_time)

