# ==============================
# Tanzila Islam
# PhD Student, Iwate University
# Email: tanzilamohita@gmail.com
# Created Date: 1/10/2022
# ===============================
import os
import pandas as pd
import numpy as np

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz/bin/'

df_column_name = pd.read_csv('../HDRA_CNN_Data/HDRA_X_C3.csv', index_col=0,
                    low_memory=False)
df_column_name = df_column_name.T
# print(df_column_name.columns)

df = pd.read_csv('../HDRA_CNN_Data/HDRA_Raw_359sel.csv', index_col=0,
                    low_memory=False, chunksize=340634)
# HDRA_Raw_359_sel shape = (359, 681268)
# print(df.shape)
category_number = 2
categories = [[] for i in range(category_number)]
index = 0
for x in df:
    # print(x.shape)
    x = x.T
    x = np.array(x)
    for row in x:
        # print(row[0])
        categories[index].append(row)
        index = (index+1) % category_number

chunk_num = 0
for chunk in categories:
    chunk = pd.DataFrame(chunk)
    print(chunk.shape)
    chunk.to_csv('../HDRA_CNN_Data/HDRA_Raw_Chunk/'
                 'HDRA_X_Raw_{}'.format(chunk_num)
                 + '.csv', header=df_column_name.columns)
    chunk_num += 1

