import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import seaborn as sns
import monai
from monai.transforms import \
    Compose, LoadImaged, AddChanneld, Orientationd, \
    Spacingd, \
    ToTensord,  \
    DataStatsd, \
    ToDeviced
from monai.data import list_data_collate
import torch
import pytorch_lightning as pl
from torchsummary import summary
import sys
sys.path.append(r'../util/python')
import vicra_toolbox

#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def delta_T_magnitude(T_ref, T):
    mag = np.sum(np.square(T-T_ref))
    return mag


# In[ ]:


def Relative_motion_A_to_B_12(img_vc1, img_vc2):
    A = np.pad(img_vc1, pad_width=1)
    B = np.pad(img_vc2, pad_width=1)
    R = vicra_toolbox.Relative_motion_A_to_B(A, B)
    return R[1:13]


# In[ ]:

def clean_df(df):
    cols_to_keep=['PatientID','InjectionID','PatientType','ScanStart',
                   'ThreeD_Cloud', 'COD_nomask','M', 'T']

    # Convert 12 matrix params to a single list in 'M'
    matrix_cols = ['VC_11', 'VC_12', 'VC_13', 'VC_14', 'VC_21', 'VC_22','VC_23', 'VC_24', 'VC_31', 'VC_32', 'VC_33', 'VC_34']
    M = df[matrix_cols].values.tolist()
    df['M'] = M
    df=df[cols_to_keep]
    return df

## using for build a legal dataset from csv file
def build_legal_dataset(df,df_sample):
    mov_cols = ['PatientID','InjectionID','PatientType', 'ScanStart_x',
                'ThreeD_Cloud_x', 'ThreeD_Cloud_y', 'COD_nomask_x', 'COD_nomask_y', 'delta_t','DataType']
    df_sample = df_sample[mov_cols]
    scan_begin_time = 3600
    # Cols to keep
    active_cols = ['PatientID','InjectionID','PatientType','ScanStart',
                'ThreeD_Cloud', 'COD_nomask','M']
    final_cols = ['PatientID','InjectionID','PatientType', 'ScanStart_x',
                'ThreeD_Cloud_x', 'ThreeD_Cloud_y', 'COD_nomask_x', 'COD_nomask_y', 'T', 'delta_t','DataType']
    # Convert 12 matrix params to a single list in 'M'
    matrix_cols = ['VC_11', 'VC_12', 'VC_13', 'VC_14', 'VC_21', 'VC_22','VC_23', 'VC_24', 'VC_31', 'VC_32', 'VC_33', 'VC_34']
    M = df[matrix_cols].values.tolist()
    df['M'] = M
    df = df[active_cols]
    n = df_sample.shape[0]
    #build reference and moving time dataset
    for i in range(n):
        ref_index = df_sample.at[i,'ScanStart_x'] - scan_begin_time
#         print(ref_index)
        df_ref = df.loc[[ref_index]]
        mov_index = ref_index + df_sample.at[i,'delta_t']
        df_mov = df.loc[[mov_index]]
        df_temp = df_ref.merge(df_mov, on=['PatientID','InjectionID','PatientType'], how='left')
        df_temp['T'] = df_temp.apply(lambda row: 
                                vicra_toolbox.RotTransMatrix_6Params(
                                    vicra_toolbox.Relative_motion_A_to_B_12(row['M_x'], row['M_y']), 1), axis=1
                                )
        if (i == 0):
            df_final = df_temp
        else:
            df_final = pd.concat([df_final,df_temp])
    df_sample_1 = df_sample.merge(df_final, on=['PatientID','InjectionID','PatientType','ScanStart_x','ThreeD_Cloud_x','ThreeD_Cloud_y','COD_nomask_x','COD_nomask_y'], how='left')
    return df_sample_1


# In[ ]:


def deal_dataframe(df):
#     df = pd.read_csv(dir)
    # Convert 12 matrix params to a single list in 'MATRIX'
    matrix_cols = ['VC_11', 'VC_12', 'VC_13', 'VC_14', 'VC_21', 'VC_22','VC_23', 'VC_24', 'VC_31', 'VC_32', 'VC_33', 'VC_34']
    M = df[matrix_cols].values.tolist()
    # df_m = pd.DataFrame()
    df['MATRIX'] = M
    df['T'] = df['MATRIX'].apply(lambda x: vicra_toolbox.RotTransMatrix_6Params(x, 1))
    df['delta_T'] = df['T'].apply(lambda t: delta_T_magnitude(df.loc[0,'T'], t))
    df['relative_T'] = df['MATRIX'].apply(lambda t: vicra_toolbox.RotTransMatrix_6Params(Relative_motion_A_to_B_12(df.loc[0,'MATRIX'], t), 1))
    df['delta_relative_T'] = df['relative_T'].apply(lambda t: delta_T_magnitude(df.loc[0,'relative_T'], t))
#     ax = sns.lineplot(data=df, x="ScanStart", y="delta_T")
#     ax = sns.lineplot(data=df, x="ScanStart", y="delta_relative_T") 
#     plt.show()
    return(df) 

