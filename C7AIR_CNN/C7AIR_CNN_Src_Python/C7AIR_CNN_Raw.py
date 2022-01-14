# ==============================
# Tanzila Islam
# PhD Student, Iwate University
# Email: tanzilamohita@gmail.com
# Created Date: 1/10/2022
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
# fix random seed for reproducibility
# seed = 7
# np.random.seed(seed)
start_time = time.time()
X = pd.read_csv('../C7AIR_CNN_Data/C7AIR_X.csv').iloc[0:, 1:]
X.columns = np.arange(0, len(X.columns))
Y = pd.read_csv('../C7AIR_CNN_Data/C7AIR_Y.csv').iloc[0:, 1:]
Y.columns = np.arange(0, len(Y.columns))
print(X.shape)
print(Y.shape)


# data partitioning into train and validation
itrait = 0  # first trait analyzed
# print(Y[itrait])
X_train, X_valid, y_train, y_valid = train_test_split(X, Y[itrait], test_size=0.2)
print(X_train.shape, y_train.shape)
print(X_valid.shape, y_valid.shape)

# CNN
batch_size = 4
epochs = 350
nSNP = X_train.shape[1]
nStride = 3     # stride between convolutions

X2_train = np.expand_dims(X_train, axis=2)
X2_valid = np.expand_dims(X_valid, axis=2)

# Instantiate
model_cnn = Sequential()
# add convolutional layer
model_cnn.add(Conv1D(32, kernel_size=3, strides=nStride, input_shape=(nSNP, 1)))
model_cnn.add(LeakyReLU(alpha=0.1))
# add pooling layer: takes maximum of two consecutive values
model_cnn.add(MaxPooling1D(pool_size=2))
# add convolutional layer
model_cnn.add(Conv1D(64, kernel_size=3, strides=nStride))
model_cnn.add(LeakyReLU(alpha=0.1))
# add pooling layer: takes maximum of two consecutive values
model_cnn.add(MaxPooling1D(pool_size=2))
# add convolutional layer
model_cnn.add(Conv1D(128, kernel_size=3, strides=nStride))
model_cnn.add(LeakyReLU(alpha=0.1))
# add pooling layer: takes maximum of two consecutive values
model_cnn.add(MaxPooling1D(pool_size=2))

# Solutions above are linearized to accommodate a standard layer
model_cnn.add(Flatten())
model_cnn.add(Dense(128, activation='linear'))
model_cnn.add(LeakyReLU(alpha=0.1))
model_cnn.add(Dense(1))

adm = tensorflow.keras.optimizers.Adam(learning_rate=0.01)
# Model Compiling (https://keras.io/models/sequential/)
model_cnn.compile(loss='mean_squared_error', optimizer=adm)

# list some properties
model_cnn.summary()

plot_model(model_cnn, to_file='../C7AIR_CNN_ModelMetaData/'
            'C7AIR_CNN_Flowgraph/C7AIR_CNN_Flowgraph_Raw.png',
           show_shapes=True, show_layer_names=True)

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=30)
# training
model_cnn_train = model_cnn.fit(X2_train, y_train, epochs=epochs,
                                verbose=1, shuffle=True,
                                batch_size=batch_size,
                                validation_data=(X2_valid, y_valid), callbacks=es)

# print(model_cnn_train.history)
loss = model_cnn_train.history['loss']
val_loss = model_cnn_train.history['val_loss']
epochs = range(epochs)
plt.plot(loss, label='Training loss')
plt.plot(val_loss, label='Validation loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.title('Training and validation loss')
plt.savefig('../C7AIR_CNN_ModelMetaData/C7AIR_CNN_Loss/'
            'C7AIR_CNN_Loss_Raw.png')
plt.show()

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
plt.savefig('../C7AIR_CNN_ModelMetaData/C7AIR_CNN_Prediction_Plot/'
            'C7AIR_CNN_Prediction_Raw.png')
plt.show()

pathlib.Path("../C7AIR_CNN_ModelMetaData/C7AIR_CNN_Prediction_Result/"
             "C7AIR_CNN_Prediction_Result_Raw.txt")\
    .write_text("C7AIR_CNN_Prediction_Result_Raw: {}"
                        .format(corr))

pathlib.Path("../C7AIR_CNN_ModelMetaData/C7AIR_CNN_Prediction_Time/"
             "C7AIR_CNN_Prediction_Time_Raw.txt")\
    .write_text("C7AIR_CNN_Prediction_Time_Raw: {}"
                        .format(time.time() - start_time))

print('Total Training Time: ', time.time() - start_time)
