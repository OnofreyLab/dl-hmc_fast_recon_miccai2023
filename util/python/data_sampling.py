import os
import numpy as np
import pandas as pd
import sys
sys.path.append(r'../../util/python')
import vicra_toolbox

def Relative_motion_A_to_B_12(img_vc1, img_vc2):
    A = np.pad(img_vc1, pad_width=1)
    B = np.pad(img_vc2, pad_width=1)
    R = vicra_toolbox.Relative_motion_A_to_B(A, B)
    return R[1:13]

# Input: 1. Summary.csy； 2. delt(t) distribution y=f(x); 3. batch_size (10000); 4. partition: eg, 8:1:1; 5.n, how many delta t for every refernce t
# Output:A csv file includes 10000 lines of ['PatientID','InjectionID','PatientType', 'ThreeD_Cloud_x', 'ThreeD_Cloud_y', 'COD_nomask_x', 'COD_nomask_y', 'T', 'delta_t']
def data_sample(df,par=0.8,n=10,Start_t=3600,TimeSpan=1800,Random_Arg=85):
    # Step1: according to partition, split out the training/valdation/testing (8:1:1) (60-90min) (1800s in total) (600s train 80 80)-> model -> 1800s??
    # 8:1:1 1440training,180validation,180test
    # 每个training随机
    # Step2: randomly sample delt(t) according to the distribution t = 500s; N (delt(t)) = 10000/ 600 
    
    #split data to train, evaluation, testing
    print('hello')
    tr_num=par*TimeSpan
    df_tr = df.sample(frac=par)
    df_valtest = pd.concat([df,df_tr,df_tr]).drop_duplicates(subset='ScanStart',keep=False)
    df_val = df_valtest.sample(frac=0.5)
    df_test = pd.concat([df_valtest,df_val,df_val]).drop_duplicates(subset='ScanStart',keep=False)  
    df_tr['DataType'] = 'Train'
    df_val['DataType'] = 'Validation'
    df_test['DataType'] = 'Testing'
    df_sample = pd.concat([df_tr,df_val])
    df_sample = pd.concat([df_sample,df_test])
    df_sample = df_sample.sort_values(by='ScanStart')

    # Cols to keep
    active_cols = ['PatientID','InjectionID','PatientType','ScanStart',
                   'ThreeD_Cloud', 'COD_nomask','M','DataType']
    final_cols = ['PatientID','InjectionID','PatientType', 'ScanStart_x',
                   'ThreeD_Cloud_x', 'ThreeD_Cloud_y', 'COD_nomask_x', 'COD_nomask_y', 'T', 'delta_t','DataType']
    mov_cols = ['PatientID','InjectionID','PatientType','ScanStart',
                   'ThreeD_Cloud', 'COD_nomask','M']

    # Convert 12 matrix params to a single list in 'M'
    matrix_cols = ['VC_11', 'VC_12', 'VC_13', 'VC_14', 'VC_21', 'VC_22','VC_23', 'VC_24', 'VC_31', 'VC_32', 'VC_33', 'VC_34']
    M = df_sample[matrix_cols].values.tolist()
    df_sample['M'] = M

    # Only keep the necessary info
    df_sample = df_sample[active_cols]

    #build distribution of delta t
    x=np.random.exponential(scale=Random_Arg, size=1800*n)
    x=np.ceil(x)
    x=x.astype(int)
    
    End_t = TimeSpan + Start_t
    #build reference and moving time dataset
    for i in range(0,TimeSpan-1):
        ref_time = df_sample.at[i,'ScanStart']
        df_ref = df_sample.loc[[i]]
        for j in range(0,n):
            mov_time = ref_time + x[i*n+j]
            eff = 1
            while (mov_time >= End_t):
                eff = eff / 2
                mov_time = ref_time + x[i*n+j] * eff
                mov_time = np.round(mov_time)

            df_mov = df_sample.loc[[mov_time-Start_t]]
            df_mov = df_mov[mov_cols]
            temp = df_ref.merge(df_mov, on=['PatientID','InjectionID','PatientType'], how='left')
            print(type(temp))
            if (i*n+j == 0):
                df_final = temp
            else:
                df_final = pd.concat([df_final,temp])
    #calculate T and delta_t   
    df_final['T'] = df_final.apply(lambda row: 
                                 vicra_toolbox.RotTransMatrix_6Params(
                                     Relative_motion_A_to_B_12(row['M_x'], row['M_y']), 1), axis=1
                                 )
    df_final['delta_t'] = df_final.apply(lambda row: row['ScanStart_y']-row['ScanStart_x'], axis=1) 
    df_final = df_final.drop_duplicates(subset=['ScanStart_x','delta_t'],keep='first')                
    print(df_final.shape[0])
    return df_final[final_cols]
    #return df_final
# pd.csvsave()