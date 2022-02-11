# ==============================
# Tanzila Islam
# PhD Student, Iwate University
# Email: tanzilamohita@gmail.com
# Created Date: 1/17/2022
# ===============================
# main modules needed
import pandas as pd
import numpy as np

import seaborn as sns
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale
import tensorflow
from tensorflow import keras
from tensorflow.keras.losses import mean_squared_error, categorical_crossentropy, logcosh
from tensorflow.keras.utils import plot_model
import os
import time
import pathlib


# keras items
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Flatten, Conv1D, MaxPooling1D, LSTM #CNNs
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.layers import LeakyReLU

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz/bin/'
# # fix random seed for reproducibility
# seed = 7
# np.random.seed(seed)

start_time = time.time()
X = pd.read_csv('../HDRA_CNN_Data/HDRA_X_C1.csv', index_col=0)#.iloc[0:, 1:]
# X.columns = np.arange(0, len(X.columns))
print(X.shape)
Y = pd.read_csv('../HDRA_CNN_Data/HDRA_Y.csv', index_col=0)#.iloc[0:, 1:]
print(Y.shape)
#Y.columns = np.arange(0, len(Y.columns))
missing_values_X = X.isnull().sum().sum()
print("Total Missing values in X:", missing_values_X)
missing_values_Y = Y.isnull().sum().sum()
print("Total Missing values in Y:", missing_values_Y)
Y = Y.dropna()
X = X.drop(X.index.difference(Y.index))
# print(X.index)
# print(Y.index)
# print(X)
X.columns = np.arange(0, len(X.columns))
Y.columns = np.arange(0, len(Y.columns))
# print(Y)
print("X shape", X.shape)
print("Y shape", Y.shape)


# data partitioning into train and validation
index = 0  # first trait analyzed
corr_df = []
# print(Y[itrait])
# Y.shape[1]
for i in range(4, 5):
    # print(Y[i])
    print(i)
    X_train, X_valid, y_train, y_valid = train_test_split(X, Y[i], test_size=0.2)
    print("X_train shape and y_train shape", X_train.shape, y_train.shape)
    print("X_valid shape, y_valid shape", X_valid.shape, y_valid.shape)

    # CNN

    batch_size = 128
    epochs = 400
    nSNP = X_train.shape[1]
    print("SNP", nSNP)
    nStride = 2     # stride between convolutions

    X2_train = np.expand_dims(X_train, axis=2)
    X2_valid = np.expand_dims(X_valid, axis=2)

    print("X2_train shape", X2_train.shape)
    print("X2_valid shape", X2_valid.shape)
    # Instantiate
    model_cnn = Sequential()
    # add convolutional layer
    model_cnn.add(Conv1D(32, 2, activation="relu", input_shape=(nSNP, 1)))
    # model_cnn.add(LeakyReLU(alpha=0.1))
    # model_cnn.add(Dropout(0.5))
    # add pooling layer: takes maximum of two consecutive values
    model_cnn.add(MaxPooling1D(pool_size=2))

    # # add convolutional layer
    model_cnn.add(Conv1D(64, 2, activation='relu'))
    # # model_cnn.add(LeakyReLU(alpha=0.1))
    # # model_cnn.add(Dropout(0.5))
    # # add pooling layer: takes maximum of two consecutive values
    model_cnn.add(MaxPooling1D(pool_size=2))
    # # add convolutional layer
    model_cnn.add(Conv1D(128, 2, activation='relu'))
    # # model_cnn.add(LeakyReLU(alpha=0.1))
    # # model_cnn.add(Dropout(0.5))
    # # add pooling layer: takes maximum of two consecutive values
    model_cnn.add(MaxPooling1D(pool_size=2))

    # Solutions above are linearized to accommodate a standard layer
    model_cnn.add(Flatten())
    model_cnn.add(Dense(64, activation='relu'))
    # model_cnn.add(LeakyReLU(alpha=0.1))
    model_cnn.add(Dense(1))


    adm = keras.optimizers.Adam(learning_rate=0.001)
    # Model Compiling (https://keras.io/models/sequential/)
    model_cnn.compile(loss='mse', optimizer=adm)
    #
    # # list some properties
    model_cnn.summary()

    plot_model(model_cnn, to_file='../HDRA_CNN_ModelMetaData/'
        'HDRA_CNN_C1/HDRA_CNN_Flowgraph_C1/HDRA_CNN_Flowgraph_C1.png',
               show_shapes=True, show_layer_names=True)

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,
                       patience=50)
    # # training
    model_cnn_train = model_cnn.fit(X2_train, y_train, epochs=epochs,
                                     batch_size=batch_size,
                                    validation_data=(X2_valid, y_valid),
                                    shuffle=True, callbacks=[es])

    # print(model_cnn_train.history)
    loss = model_cnn_train.history['loss']
    val_loss = model_cnn_train.history['val_loss']
    #epochs = range(epochs)
    plt.plot(loss, label='Training loss')
    plt.plot(val_loss, label='Validation loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    plt.title('Training and validation loss')
    plt.savefig('../HDRA_CNN_ModelMetaData/'
        'HDRA_CNN_C1/HDRA_CNN_Loss_C1/'
                'HDRA_CNN_Loss_C1_Trait_{}'.format(i)
                + '.png')
    # plt.show()
    plt.clf()
    #
    # cross-validation
    mse_prediction = model_cnn.evaluate(X2_valid, y_valid, batch_size=batch_size)
    print('\nMSE in prediction =', mse_prediction)

    # get predicted target values
    y_hat = model_cnn.predict(X2_valid, batch_size=batch_size)
    np.seterr(divide='ignore', invalid='ignore')
    # correlation btw predicted and observed
    corr = np.corrcoef(y_valid, y_hat[:, 0])[0, 1]
    print('\nCorr obs vs pred =', corr)

    data = [y_valid, y_hat[:, 0]]
    df = pd.DataFrame(np.column_stack(data))
    # print(df)

    # plot observed vs. predicted targets
    plt.title('CNN: Observed vs Predicted Y')
    plt.ylabel('Predicted')
    plt.xlabel('Observed')
    plt.scatter(y_valid, y_hat, marker='o')
    #obtain m (slope) and b(intercept) of linear regression line
    m, b = np.polyfit(y_valid, y_hat, 1)

    #add linear regression line to scatterplot
    plt.plot(y_valid, m*y_valid+b)
    plt.savefig('../HDRA_CNN_ModelMetaData/'
        'HDRA_CNN_C1/HDRA_CNN_Prediction_Plot_C1/'
                'HDRA_CNN_Prediction_Plot_C1_Trait_{}'.format(i) + '.png')
    # plt.show()
    plt.clf()

    corr_df.append([i, corr])
    index += 1


# np.savetxt('../HDRA_CNN_ModelMetaData/'
#         'HDRA_CNN_C1/HDRA_CNN_Prediction_Result_C1/'
#                 'HDRA_CNN_Prediction_Result_C1.csv',
#            np.column_stack(corr_df), delimiter=',', fmt='%1.3f', comments='')
#
# pathlib.Path("../HDRA_CNN_ModelMetaData/HDRA_CNN_C1/"
#              "HDRA_CNN_Prediction_Time_C1/"
#         "HDRA_CNN_Prediction_Time_C1.txt").\
#     write_text("HDRA_CNN_Prediction_Time_C1: {}"
#         .format(time.time() - start_time))
print('Total Training Time: ', time.time() - start_time)
