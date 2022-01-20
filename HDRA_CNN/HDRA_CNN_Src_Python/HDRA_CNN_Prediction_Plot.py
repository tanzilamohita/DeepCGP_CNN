# ==============================
# Tanzila Islam
# PhD Student, Iwate University
# Email: tanzilamohita@gmail.com
# Created Date: 1/20/2022
# ===============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

hdra_raw = pd.read_csv('../HDRA_CNN_ModelMetaData/HDRA_CNN_Raw/'
                       'HDRA_CNN_Prediction_Result_Raw/'
                       'HDRA_CNN_Prediction_Result_Raw_Thesis.csv',
                       low_memory=False)
hdra_comp1 = pd.read_csv('../HDRA_CNN_ModelMetaData/HDRA_CNN_C1/'
                         'HDRA_CNN_Prediction_Result_C1/'
                         'HDRA_CNN_Prediction_Result_C1_Thesis.csv',
                         low_memory=False)
hdra_comp2 = pd.read_csv('../HDRA_CNN_ModelMetaData/HDRA_CNN_C2/'
                         'HDRA_CNN_Prediction_Result_C2/'
                         'HDRA_CNN_Prediction_Result_C2.csv',
                         low_memory=False)
hdra_comp3 = pd.read_csv('../HDRA_CNN_ModelMetaData/HDRA_CNN_C3/'
                         'HDRA_CNN_Prediction_Result_C3/'
                         'HDRA_CNN_Prediction_Result_C3.csv',
                         low_memory=False)

# # set width of bar
barWidth = 0.25

compression_level = ['0%', '57%', '93%', '98%']
hdra_cnn_df = pd.concat([hdra_raw, hdra_comp1, hdra_comp2, hdra_comp3])
# print(hdra_cnn_df)
hdra_cnn_df = hdra_cnn_df.transpose()
hdra_cnn_df.columns = list(compression_level)
hdra_cnn_df.index = np.arange(1, len(hdra_cnn_df)+1)
print(hdra_cnn_df)
hdra_cnn_df.to_csv('../HDRA_CNN_ModelMetaData/HDRA_CNN_Prediction_Result_Plot/'
                       'HDRA_CNN_Prediction_Accuracy.csv')

hdra_cnn_df.plot(kind="bar", figsize=(15, 10), rot=0, width=0.8)
plt.title("CNN Prediction Accuracy for HDRA", fontsize=24)

plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel("Trait Id", fontsize=18)
plt.ylabel("Accuracy", fontsize=18)
plt.legend(fontsize=12, loc='upper right')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.savefig('../HDRA_CNN_ModelMetaData/HDRA_CNN_Prediction_Result_Plot/'
                       'HDRA_CNN_Prediction_Result_Plot.png')
plt.show()

