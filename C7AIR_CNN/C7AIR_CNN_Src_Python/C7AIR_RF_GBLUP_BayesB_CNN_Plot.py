# ==============================
# Tanzila Islam
# PhD Student, Iwate University
# Email: tanzilamohita@gmail.com
# Created Date: 1/14/2022
# ===============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statistics

C7AIR_Acc = pd.read_csv('../C7AIR_CNN_ModelMetaData/'
                         'C7AIR_PredictionAccuracy_RF_GBLUP_BayesB_CNN/'
                       'C7AIR_PredictionAccuracy_RF_GBLUP_BayesB_CNN.csv',
                               low_memory=False, index_col=0)
print(C7AIR_Acc)

ax = C7AIR_Acc.plot(kind="bar", figsize=(16, 12), rot=0, capsize=6)
plt.title("Prediction Accuracy of BayesB, GBLUP, RF and CNN (C7AIR)", fontsize=24)
plt.xlabel("Compression Level", fontsize=18)
plt.ylabel("Accuracy", fontsize=18)
plt.legend(loc='lower right', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
#print(accuracy_df.index)
for p in ax.patches:
    ax.annotate(str("{:.2f}".format(p.get_height())), (p.get_x() + p.get_width() / 2,
                    p.get_height()), fontsize=15, ha='center', va='center',
                xytext=(0, 8), textcoords='offset points')

plt.savefig('../C7AIR_CNN_ModelMetaData/'
            'C7AIR_PredictionAccuracy_RF_GBLUP_BayesB_CNN/'
                       'C7AIR_PredictionAccuracy_RF_GBLUP_BayesB_CNN.png')
plt.show()
