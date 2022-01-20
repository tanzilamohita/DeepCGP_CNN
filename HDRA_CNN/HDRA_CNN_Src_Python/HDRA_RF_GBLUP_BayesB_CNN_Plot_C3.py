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

# processing BayesB Prediction Accuracy
HDRA_c3_BayesB = pd.read_csv('../HDRA_CNN_ModelMetaData/HDRA_Prediction_Result_BayesB/'
                          'HDRA_PredictionAccuracyProcessed_Compress_3_BayesB.csv', low_memory=False)
# set height of bar
HDRA_c3_mean_BayesB = HDRA_c3_BayesB.mean(axis=0)
HDRA_c3_mean_BayesB.index = np.arange(1, len(HDRA_c3_mean_BayesB)+1)
HDRA_c3_mean_BayesB = pd.DataFrame(HDRA_c3_mean_BayesB)
# print(HDRA_c3_mean_BayesB)
# processing Ranger Prediction Accuracy
HDRA_c3_ranger = pd.read_csv('../HDRA_CNN_ModelMetaData/HDRA_Prediction_Result_RF/'
                       'HDRA_PredictionAccuracyProcessed_Compress_3.csv', low_memory=False)
# set height of bar
HDRA_c3_mean_ranger = HDRA_c3_ranger.mean(axis=0)
HDRA_c3_mean_ranger.index = np.arange(1, len(HDRA_c3_mean_ranger)+1)
HDRA_c3_mean_ranger = pd.DataFrame(HDRA_c3_mean_ranger)
# print(HDRA_raw_mean_ranger)

# processing GBLUP Prediction Accuracy
HDRA_c3_gblup = pd.read_csv('../HDRA_CNN_ModelMetaData/HDRA_Prediction_Result_GBLUP/'
                        'HDRA_PredictionAccuracyProcessed_Compress_3_GBLUP.csv', low_memory=False)
# set height of bar
HDRA_c3_mean_gblup = HDRA_c3_gblup.mean(axis=0)
HDRA_c3_mean_gblup.index = np.arange(1, len(HDRA_c3_mean_gblup)+1)
HDRA_c3_mean_gblup = pd.DataFrame(HDRA_c3_mean_gblup)
# print(HDRA_raw_mean_gblup)
# processing CNN Prediction Accuracy
hdra_c3_CNN = pd.read_csv('../HDRA_CNN_ModelMetaData/HDRA_CNN_C3/'
                       'HDRA_CNN_Prediction_Result_C3/'
                       'HDRA_CNN_Prediction_Result_C3.csv',
                       low_memory=False)
hdra_c3_CNN = hdra_c3_CNN.T
hdra_c3_CNN.index = np.arange(1, len(hdra_c3_CNN)+1)
hdra_c3_CNN = pd.DataFrame(hdra_c3_CNN)
# print(hdra_raw_CNN)
hdra_0 = pd.concat([HDRA_c3_mean_BayesB, HDRA_c3_mean_gblup, HDRA_c3_mean_ranger, hdra_c3_CNN], axis=1, sort=False)
hdra_0.columns = ['BayesB', 'GBLUP', 'RF', 'CNN']
# hdra_0.insert(0, 'BayesB', 0)
print(hdra_0)

ax = hdra_0.plot(kind="bar",
                 figsize=(16, 12), rot=0, capsize=6)
plt.title("Prediction Accuracy of BayesB, GBLUP, RF and CNN (HDRA) (98%)", fontsize=24)
plt.xlabel("Trait ID", fontsize=18)
plt.ylabel("Accuracy", fontsize=18)
plt.legend(loc='lower right', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

plt.savefig('../HDRA_CNN_ModelMetaData/HDRA_CNN_Prediction_Result_Plot/'
                        'HDRA_PredictionAccuracy_BayesB_GBLUP_RF_CNN_C3.png')
plt.show()
