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
HDRA_comp1_ranger = pd.read_csv('../HDRA_CNN_ModelMetaData/HDRA_Prediction_Result_RF/'
                         'HDRA_PredictionAccuracyProcessed_Compress_1.csv', low_memory=False)
HDRA_comp2_ranger = pd.read_csv('../HDRA_CNN_ModelMetaData/HDRA_Prediction_Result_RF/'
                         'HDRA_PredictionAccuracyProcessed_Compress_2.csv', low_memory=False)
HDRA_comp3_ranger = pd.read_csv('../HDRA_CNN_ModelMetaData/HDRA_Prediction_Result_RF/'
                         'HDRA_PredictionAccuracyProcessed_Compress_3.csv', low_memory=False)

# set height of bar
HDRA_raw_mean_ranger = statistics.mean(list(HDRA_raw_ranger.mean(axis=0)))
HDRA_comp1_mean_ranger= statistics.mean(list(HDRA_comp1_ranger.mean(axis=0)))
HDRA_comp2_mean_ranger = statistics.mean(list(HDRA_comp2_ranger.mean(axis=0)))
HDRA_comp3_mean_ranger = statistics.mean(list(HDRA_comp3_ranger.mean(axis=0)))

## Calculate Standard Deviation
HDRA_comp1_SD_RF = np.std(list(HDRA_comp1_ranger.mean(axis=0)*100))
HDRA_comp2_SD_RF = np.std(list(HDRA_comp2_ranger.mean(axis=0)*100))
HDRA_comp3_SD_RF = np.std(list(HDRA_comp3_ranger.mean(axis=0)*100))

HDRA_SD_RF = [HDRA_comp1_SD_RF, HDRA_comp2_SD_RF, HDRA_comp3_SD_RF]
print(HDRA_SD_RF)
#print(HDRA_comp2_SD_RF)
#print(list(HDRA_comp3_ranger.mean(axis=0)*100))

compression_level = ['0%', '57%', '93%', '98%']
accuracy_data_ranger = [HDRA_raw_mean_ranger, HDRA_comp1_mean_ranger, HDRA_comp2_mean_ranger,
                        HDRA_comp3_mean_ranger]
accuracy_df_ranger = pd.DataFrame(np.column_stack(accuracy_data_ranger))
accuracy_df_ranger.columns = list(compression_level)
accuracy_df_ranger = accuracy_df_ranger.transpose()
accuracy_df_ranger = np.array(accuracy_df_ranger)
#print(accuracy_df_ranger[0])
#accuracy_df_ranger = accuracy_df_ranger/accuracy_df_ranger[0]*100
accuracy_df_ranger = pd.DataFrame(accuracy_df_ranger, columns=['RF'],
                                  index=compression_level)
print(accuracy_df_ranger)
#
# processing GBLUP Prediction Accuracy
HDRA_raw_gblup = pd.read_csv('../HDRA_CNN_ModelMetaData/HDRA_Prediction_Result_GBLUP/'
                        'HDRA_PredictionAccuracyProcessed_Raw_GBLUP.csv', low_memory=False)
HDRA_comp1_gblup = pd.read_csv('../HDRA_CNN_ModelMetaData/HDRA_Prediction_Result_GBLUP/'
                          'HDRA_PredictionAccuracyProcessed_Compress_1_GBLUP.csv', low_memory=False)
HDRA_comp2_gblup = pd.read_csv('../HDRA_CNN_ModelMetaData/HDRA_Prediction_Result_GBLUP/'
                         'HDRA_PredictionAccuracyProcessed_Compress_2_GBLUP.csv', low_memory=False)
HDRA_comp3_gblup = pd.read_csv('../HDRA_CNN_ModelMetaData/HDRA_Prediction_Result_GBLUP/'
                         'HDRA_PredictionAccuracyProcessed_Compress_3_GBLUP.csv', low_memory=False)

# set height of bar
HDRA_raw_mean_gblup = statistics.mean(list(HDRA_raw_gblup.mean(axis=0)))
HDRA_comp1_mean_gblup= statistics.mean(list(HDRA_comp1_gblup.mean(axis=0)))
HDRA_comp2_mean_gblup = statistics.mean(list(HDRA_comp2_gblup.mean(axis=0)))
HDRA_comp3_mean_gblup = statistics.mean(list(HDRA_comp3_gblup.mean(axis=0)))

## Calculate Standard Deviation
HDRA_comp1_SD_gblup = np.std(list(HDRA_comp1_gblup.mean(axis=0)*100))
HDRA_comp2_SD_gblup = np.std(list(HDRA_comp2_gblup.mean(axis=0)*100))
HDRA_comp3_SD_gblup = np.std(list(HDRA_comp3_gblup.mean(axis=0)*100))

HDRA_SD_gblup = [HDRA_comp1_SD_gblup, HDRA_comp2_SD_gblup, HDRA_comp3_SD_gblup]
print(HDRA_SD_gblup)

accuracy_data_gblup = [HDRA_raw_mean_gblup, HDRA_comp1_mean_gblup, HDRA_comp2_mean_gblup,
                       HDRA_comp3_mean_gblup]
accuracy_df_gblup = pd.DataFrame(np.column_stack(accuracy_data_gblup))
accuracy_df_gblup = accuracy_df_gblup.transpose()
accuracy_df_gblup = np.array(accuracy_df_gblup)
# print(accuracy_df_gblup)
#accuracy_df_gblup = accuracy_df_gblup/accuracy_df_gblup[0]*100
accuracy_df_gblup = pd.DataFrame(accuracy_df_gblup, columns=['GBLUP'], index=compression_level)
print(accuracy_df_gblup)
#
# processing BayesB Prediction Accuracy
# HDRA_raw_BayesB = pd.read_csv('../HDRA_Data/HDRA_Prediction_Accuracy/'
#                        'HDRA_PredictionAccuracy_Raw_BayesB.csv', low_memory=False)
HDRA_comp1_BayesB = pd.read_csv('../HDRA_CNN_ModelMetaData/HDRA_Prediction_Result_BayesB/'
                          'HDRA_PredictionAccuracyProcessed_Compress_1_BayesB.csv', low_memory=False)
HDRA_comp2_BayesB = pd.read_csv('../HDRA_CNN_ModelMetaData/HDRA_Prediction_Result_BayesB/'
                         'HDRA_PredictionAccuracyProcessed_Compress_2_BayesB.csv', low_memory=False)
HDRA_comp3_BayesB = pd.read_csv('../HDRA_CNN_ModelMetaData/HDRA_Prediction_Result_BayesB/'
                         'HDRA_PredictionAccuracyProcessed_Compress_3_BayesB.csv', low_memory=False)

# print(HDRA_comp2_BayesB.iloc[:, : 3])
# print((HDRA_comp2_BayesB.mean(axis=0)*100))
# set height of bar
# HDRA_raw_mean_BayesB = list(HDRA_raw_BayesB.mean(axis=0)*100)
HDRA_comp1_mean_BayesB = statistics.mean(list(HDRA_comp1_BayesB.mean(axis=0)))
HDRA_comp2_mean_BayesB = statistics.mean(list(HDRA_comp2_BayesB.mean(axis=0)))
HDRA_comp3_mean_BayesB = statistics.mean(list(HDRA_comp3_BayesB.mean(axis=0)))
# print(list(HDRA_comp3_BayesB.mean(axis=0)*100))

## Calculate Standard Deviation
HDRA_comp1_SD_BayesB = np.std(list(HDRA_comp1_BayesB.mean(axis=0)*100))
HDRA_comp2_SD_BayesB = np.std(list(HDRA_comp2_BayesB.mean(axis=0)*100))
HDRA_comp3_SD_BayesB = np.std(list(HDRA_comp3_BayesB.mean(axis=0)*100))

HDRA_SD_BayesB = [HDRA_comp1_SD_BayesB, HDRA_comp2_SD_BayesB, HDRA_comp3_SD_BayesB]
print(HDRA_SD_BayesB)
accuracy_data_BayesB = [0, HDRA_comp1_mean_BayesB, HDRA_comp2_mean_BayesB,
                        HDRA_comp3_mean_BayesB]
accuracy_df_BayesB = pd.DataFrame(np.column_stack(accuracy_data_BayesB))
accuracy_df_BayesB = accuracy_df_BayesB.transpose()
accuracy_df_BayesB = np.array(accuracy_df_BayesB)
# print(accuracy_df_BayesB)
#accuracy_df_BayesB = accuracy_df_BayesB/accuracy_df_BayesB[0]*100
accuracy_df_BayesB = pd.DataFrame(accuracy_df_BayesB, columns=['BayesB'],
                                  index=compression_level)
print(accuracy_df_BayesB)

accuracy_df_CNN = pd.read_csv('../HDRA_CNN_ModelMetaData/HDRA_CNN_Prediction_Result_Plot/'
                       'HDRA_CNN_Prediction_Accuracy.csv', low_memory=False, index_col=0)
accuracy_df_CNN = accuracy_df_CNN.mean(axis=0)
accuracy_df_CNN = pd.DataFrame(accuracy_df_CNN, columns=['CNN'],
                                  index=compression_level)
# print("CNN")
print(accuracy_df_CNN)
accuracy_df = pd.concat([accuracy_df_BayesB, accuracy_df_gblup, accuracy_df_ranger, accuracy_df_CNN],
                        axis=1, sort=False)
print(accuracy_df)
# # accuracy_df.to_csv('../HDRA_Data/HDRA_Prediction_Accuracy/'
# #                         'HDRA_PredictionAccuracy_BayesB_GLUP_RF.csv')
#
# accuracy_df = pd.read_csv('../HDRA_Data/HDRA_Prediction_Accuracy/'
#                         'HDRA_PredictionAccuracy_BayesB_GLUP_RF.csv', low_memory=False)
# print(accuracy_df)
# ax = accuracy_df.plot(kind="bar", yerr=[HDRA_SD_BayesB, HDRA_SD_gblup, HDRA_SD_RF],
#                  figsize=(16, 12), rot=0, capsize=6)
ax = accuracy_df.plot(kind="bar",
                 figsize=(16, 12), rot=0, capsize=6)
plt.title("Prediction Accuracy of BayesB, GBLUP, RF and CNN (HDRA)", fontsize=24)
plt.xlabel("Compression Level", fontsize=18)
plt.ylabel("Accuracy", fontsize=18)
plt.legend(loc='lower right', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

for p in ax.patches:
    if p.get_height() > 0:
        ax.annotate(str("{:.2f}".format(p.get_height())), (p.get_x() + p.get_width() / 2,
                    p.get_height()), fontsize=12, ha='center', va='center',
                    xytext=(0, 8), textcoords='offset points')
    else:
        ax.annotate(str("N/A"), (p.get_x() + p.get_width() / 2,
                    p.get_height()), fontsize=12, ha='center', va='center',
                    xytext=(0, 8), textcoords='offset points')

# for p in ax.patches:
#     ax.annotate(str("{:.2f}".format(p.get_height())), (p.get_x() + p.get_width() / 2,
#                     p.get_height()), fontsize=15, ha='center', va='center',
#                 xytext=(0, 8), textcoords='offset points')
                # xytext=(0, 15), textcoords='offset points')
# # # plt.savefig('../HDRA_Data/HDRA_Prediction_Accuracy/'
# # #                        'HDRA_PredictionAccuracy.png')
# plt.savefig('../HDRA_Data/HDRA_Prediction_Accuracy/'
#                         'HDRA_PredictionRelativeAccuracy_Ranger_GBLUP_BayesB.png')
plt.savefig('../HDRA_CNN_ModelMetaData/HDRA_CNN_Prediction_Result_Plot/'
                        'HDRA_PredictionAccuracy_BayesB_GBLUP_RF_CNN.png')
plt.show()

