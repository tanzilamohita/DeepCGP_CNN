# ==============================
# Tanzila Islam
# PhD Student, Iwate University
# Email: tanzilamohita@gmail.com
# Created Date: 1/20/2022
# ===============================

import pandas as pd
import statistics
import numpy as np
import matplotlib.pyplot as plt

# processing Ranger Prediction Accuracy
HDRA_raw_ranger = pd.read_csv('../HDRA_CNN_ModelMetaData/HDRA_Prediction_Result_RF/'
                       'HDRA_PredictionAccuracyProcessed_Raw.csv', low_memory=False)
# set height of bar
HDRA_raw_mean_ranger = HDRA_raw_ranger.mean(axis=0)
HDRA_raw_mean_ranger.index = np.arange(1, len(HDRA_raw_mean_ranger)+1)
HDRA_raw_mean_ranger = pd.DataFrame(HDRA_raw_mean_ranger)
# print(HDRA_raw_mean_ranger)

# processing GBLUP Prediction Accuracy
HDRA_raw_gblup = pd.read_csv('../HDRA_CNN_ModelMetaData/HDRA_Prediction_Result_GBLUP/'
                        'HDRA_PredictionAccuracyProcessed_Raw_GBLUP.csv', low_memory=False)
# set height of bar
HDRA_raw_mean_gblup = HDRA_raw_gblup.mean(axis=0)
HDRA_raw_mean_gblup.index = np.arange(1, len(HDRA_raw_mean_gblup)+1)
HDRA_raw_mean_gblup = pd.DataFrame(HDRA_raw_mean_gblup)
# print(HDRA_raw_mean_gblup)
# processing CNN Prediction Accuracy
hdra_raw_CNN = pd.read_csv('../HDRA_CNN_ModelMetaData/HDRA_CNN_Raw/'
                       'HDRA_CNN_Prediction_Result_Raw/'
                       'HDRA_CNN_Prediction_Result_Raw_Thesis.csv',
                       low_memory=False)
hdra_raw_CNN = hdra_raw_CNN.T
hdra_raw_CNN.index = np.arange(1, len(hdra_raw_CNN)+1)
hdra_raw_CNN = pd.DataFrame(hdra_raw_CNN)
# print(hdra_raw_CNN)
hdra_0 = pd.concat([HDRA_raw_mean_gblup, HDRA_raw_mean_ranger, hdra_raw_CNN], axis=1, sort=False)
hdra_0.columns = ['GBLUP', 'RF', 'CNN']
# hdra_0.insert(0, 'BayesB', 0)
print(hdra_0)

ax = hdra_0.plot(kind="bar",
                 figsize=(16, 12), rot=0, capsize=6)
plt.title("Prediction Accuracy of BayesB, GBLUP, RF and CNN (HDRA) (0%)", fontsize=24)
plt.xlabel("Trait ID", fontsize=18)
plt.ylabel("Accuracy", fontsize=18)
plt.legend(loc='lower right', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

plt.savefig('../HDRA_CNN_ModelMetaData/HDRA_CNN_Prediction_Result_Plot/'
                        'HDRA_PredictionAccuracy_BayesB_GBLUP_RF_CNN_Raw.png')
plt.show()
