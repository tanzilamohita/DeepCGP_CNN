# ==============================
# Tanzila Islam
# PhD Student, Iwate University
# Email: tanzilamohita@gmail.com
# Created Date: 1/20/2022
# ===============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

C7AIR_CNN_ACC = pd.read_csv('../C7AIR_CNN_ModelMetaData/'
                    'C7AIR_CNN_Prediction_Result/'
                    'C7AIR_CNN_PredictionAccuracy.csv', low_memory=False, index_col=0)
print(C7AIR_CNN_ACC)

C7AIR_CNN_ACC.plot(kind="bar", figsize=(12, 8), rot=0)
plt.title("CNN Prediction Accuracy for C7AIR", fontsize=24)
plt.xlabel("Compression Level", fontsize=18)
plt.ylabel("Accuracy (%)", fontsize=18)

plt.legend(loc='upper right', fontsize=12)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.savefig('../C7AIR_CNN_ModelMetaData/C7AIR_CNN_Prediction_Result_Plot/'
                       'C7AIR_CNN_PredictionAccuracy.png')
plt.show()

