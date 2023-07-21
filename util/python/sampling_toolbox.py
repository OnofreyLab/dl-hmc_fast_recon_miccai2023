# This toolbox gathers sampling functions allowing to sample the moving images with respect to the reference image in different ways. 

# At the moment (02/24/22), you can find the sampling functions:
# -data_sample: normal distribution sampling
# -data_sample_random: random sampling 
# -data_sample_ref3600: reference image fixed at 3600s. 
# -data_sample_6ref: using 6 reference times fixed at 3600, 3900, 4200, 4500, 4800 and 5100s.

# In each function, n is the number of samples for each reference image. n=6 seems to be the best choice so far.

import pandas as pd
import numpy as np

import sys
sys.path.append(r'../util/python')
import vicra_toolbox

from vicra_toolbox import RotTransMatrix_6Params,Relative_motion_A_to_B
from data_prep_toolbox import delta_T_magnitude, Relative_motion_A_to_B_12, build_legal_dataset, deal_dataframe, clean_df

def data_sample_continuous(df,train_frac=1.0,val_frac=0.0, test_frac=0.0, ref_suffix="_ref", mov_suffix="_mov", force_first_frame=False):
    
    #split data to train, evaluation, testing
    t_frac=1-val_frac-test_frac
    df_tr = df.sample(frac=t_frac)
    df_valtest = pd.concat([df,df_tr,df_tr]).drop_duplicates(subset='ScanStart',keep=False)
    df_val = df_valtest.sample(frac=0.5)
    df_test = pd.concat([df_valtest,df_val,df_val]).drop_duplicates(subset='ScanStart',keep=False)  
    df_tr = df_tr.sort_values(by='ScanStart')    
    df_tr=df_tr.reset_index(drop=True)   
    df_val = df_val.sort_values(by='ScanStart')    
    df_val=df_val.reset_index(drop=True)   
    df_test = df_test.sort_values(by='ScanStart')    
    df_test=df_test.reset_index(drop=True)   
    
    End_t = 5399
    #build reference and moving time dataset for train
    n=int(t_frac*1800)
    sample = int(train_frac)
    
    for i in range(0,n):
        ref_time = df_tr.at[i,'ScanStart']
        df_ref = df_tr.loc[[i]]
        df_ref=df_ref.reset_index(drop=True)   
        for j in range(0,sample):
            mov_time = ref_time + j+1;
            if mov_time<=End_t:
                df_mov = df.loc[[mov_time-3600]]
                df_mov=df_mov.reset_index(drop=True)  
                temp = df_ref.join(df_mov,lsuffix=ref_suffix,rsuffix=mov_suffix)
        
                #print(type(temp))
                if (i*sample+j == 0):
                    df_final = temp
                else:
                    df_final = pd.concat([df_final,temp])
    df_final = df_final.reset_index(drop=True)    
    df_train = df_final

    #build reference and moving time dataset for val
    n=int(val_frac*1800)
    
    for i in range(0,n):
        ref_time = df_tr.at[i,'ScanStart']
        df_ref = df_val.loc[[i]]
        df_ref=df_ref.reset_index(drop=True)   
        for j in range(0,sample):
            mov_time = ref_time + j+1;
            if mov_time<=End_t:
                df_mov = df.loc[[mov_time-3600]]
                df_mov=df_mov.reset_index(drop=True)  
                temp = df_ref.join(df_mov,lsuffix=ref_suffix,rsuffix=mov_suffix)
        
                #print(type(temp))
                if (i*sample+j == 0):
                    df_final1 = temp
                else:
                    df_final1 = pd.concat([df_final1,temp])
    df_final1 = df_final1.reset_index(drop=True)    
    df_validation = df_final1

    #build reference and moving time dataset for test
    n=int(test_frac*1800)
    
    for i in range(0,n):
        ref_time = df_test.at[i,'ScanStart']
        df_ref = df_test.loc[[i]]
        df_ref=df_ref.reset_index(drop=True)   
        for j in range(0,sample):
            mov_time = ref_time + j+1;
            if mov_time<=End_t:
                df_mov = df.loc[[mov_time-3600]]
                df_mov=df_mov.reset_index(drop=True)  
                temp = df_ref.join(df_mov,lsuffix=ref_suffix,rsuffix=mov_suffix)
        
                #print(type(temp))
                if (i*sample+j == 0):
                    df_final2 = temp
                else:
                    df_final2 = pd.concat([df_final2,temp])
    df_final2 = df_final2.reset_index(drop=True)    
    df_testing = df_final2
    
    return df_train,df_validation,df_testing


###From vicra toolbox (normal sampling)###

def data_sample(df,par=0.8,n=10,Start_t=3600,TimeSpan=1800,Random_Arg=85):
    
    def Relative_motion_A_to_B_12(img_vc1, img_vc2):
        A = np.pad(img_vc1, pad_width=1)
        B = np.pad(img_vc2, pad_width=1)
        R = Relative_motion_A_to_B(A, B)
        return R[1:13]

    # Step1: according to partition, split out the training/valdation/testing (8:1:1) (60-90min) (1800s in total) (600s train 80 80)-> model -> 1800s??
    # 8:1:1 1440training,180validation,180test
    # 每个training随机
    # Step2: randomly sample delt(t) according to the distribution t = 500s; N (delt(t)) = 10000/ 600 
    
    #split data to train, evaluation, testing
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
            #print(type(temp))
            if (i*n+j == 0):
                df_final = temp
            else:
                df_final = pd.concat([df_final,temp])
    #calculate T and delta_t   
    df_final['T'] = df_final.apply(lambda row: 
                                 RotTransMatrix_6Params(
                                     Relative_motion_A_to_B_12(row['M_x'], row['M_y']), 1), axis=1
                                 )
    df_final['delta_t'] = df_final.apply(lambda row: row['ScanStart_y']-row['ScanStart_x'], axis=1) 
    df_final = df_final.drop_duplicates(subset=['ScanStart_x','delta_t'],keep='first')                
#     print(df_final.shape[0])
    return df_final

### Totally random ###

def data_sample_random(df,par=0.8, n=6, Start_t=3600,TimeSpan=1800):
    
    def Relative_motion_A_to_B_12(img_vc1, img_vc2):
        A = np.pad(img_vc1, pad_width=1)
        B = np.pad(img_vc2, pad_width=1)
        R = Relative_motion_A_to_B(A, B)
        return R[1:13]

    # Step1: according to partition, split out the training/valdation/testing (8:1:1) (60-90min) (1800s in total) (600s train 80 80)-> model -> 1800s??
    # 8:1:1 1440training,180validation,180test
    # 每个training随机
    # Step2: randomly sample delt(t) according to the distribution t = 500s; N (delt(t)) = 10000/ 600 
    
    #split data to train, evaluation, testing
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
    x=np.random.randint(1,1800,size=n)
    for i in range(1,TimeSpan-1):
        x1=np.random.randint(1,1800-i,size=n)
        x=np.append(x,x1)

    df_final = pd.DataFrame()
    #build reference and moving time dataset
    for i in range(0,TimeSpan-1):
        ref_time = df_sample.at[i,'ScanStart']
        df_ref = df_sample.loc[[i]]
        for j in range(0,n):
            mov_time = ref_time + x[i*n+j]
#             if (mov_time-Start_t>1800):
#                 print(mov_time-Start_t)
#                 print(ref_time)
#                 print(x[i*n+j])
#                 print(i)
#                 print(j)
            df_mov = df_sample.loc[[mov_time-Start_t]]
            df_mov = df_mov[mov_cols]
            temp = df_ref.merge(df_mov, on=['PatientID','InjectionID','PatientType'], how='left')
            #print(type(temp))
            df_final = df_final.append(temp)
    #calculate T and delta_t   
    df_final['T'] = df_final.apply(lambda row: 
                                 vicra_toolbox.RotTransMatrix_6Params(
                                     Relative_motion_A_to_B_12(row['M_x'], row['M_y']), 1), axis=1
                                 )
    df_final['delta_t'] = df_final.apply(lambda row: row['ScanStart_y']-row['ScanStart_x'], axis=1) 
    df_final = df_final.drop_duplicates(subset=['ScanStart_x','delta_t'],keep='first')                
#     print(df_final.shape[0])
    return df_final

### Reference fixed at 3600 ###

def data_sample_ref3600(df,par=0.8,Start_t=3600,TimeSpan=1800):
    
    def Relative_motion_A_to_B_12(img_vc1, img_vc2):
        A = np.pad(img_vc1, pad_width=1)
        B = np.pad(img_vc2, pad_width=1)
        R = Relative_motion_A_to_B(A, B)
        return R[1:13]

    # Step1: according to partition, split out the training/valdation/testing (8:1:1) (60-90min) (1800s in total) (600s train 80 80)-> model -> 1800s??
    # 8:1:1 1440training,180validation,180test
    # 每个training随机
    # Step2: randomly sample delt(t) according to the distribution t = 500s; N (delt(t)) = 10000/ 600 
    
    #split data to train, evaluation, testing
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
    
    End_t = TimeSpan + Start_t
    ref_time = df_sample.at[0,'ScanStart']
    df_ref = df_sample.loc[[0]]
    df_ref = df_ref[mov_cols]
    df_final = pd.DataFrame()
    #build reference and moving time dataset
    for i in range(0,TimeSpan-1):
        mov_time = ref_time + i+1
        df_mov = df_sample.loc[[mov_time-Start_t]]
        df_mov = df_mov[active_cols]
        temp = df_ref.merge(df_mov, on=['PatientID','InjectionID','PatientType'], how='left')
        df_final = df_final.append(temp)
    #calculate T and delta_t   
    df_final['T'] = df_final.apply(lambda row: 
                                 vicra_toolbox.RotTransMatrix_6Params(
                                     Relative_motion_A_to_B_12(row['M_x'], row['M_y']), 1), axis=1
                                 )
    df_final['delta_t'] = df_final.apply(lambda row: row['ScanStart_y']-row['ScanStart_x'], axis=1)                
#     print(df_final.shape[0])
    return df_final

### 6 ref points (3600,3900,4200,4500,4800,5100) ###

def data_sample_6ref(df,par=0.8,Start_t=3600,TimeSpan=1800,ref_span=300):
    
    def Relative_motion_A_to_B_12(img_vc1, img_vc2):
        A = np.pad(img_vc1, pad_width=1)
        B = np.pad(img_vc2, pad_width=1)
        R = Relative_motion_A_to_B(A, B)
        return R[1:13]

    # Step1: according to partition, split out the training/valdation/testing (8:1:1) (60-90min) (1800s in total) (600s train 80 80)-> model -> 1800s??
    # 8:1:1 1440training,180validation,180test
    # 每个training随机
    # Step2: randomly sample delt(t) according to the distribution t = 500s; N (delt(t)) = 10000/ 600 

    # Cols to keep
    active_cols = ['PatientID','InjectionID','PatientType','ScanStart',
                   'ThreeD_Cloud', 'COD_nomask','M','DataType']
    mov_cols = ['PatientID','InjectionID','PatientType','ScanStart',
                   'ThreeD_Cloud', 'COD_nomask','M']

    times= TimeSpan / ref_span
    times=int(times)
    ref_time = 0
    df_final = pd.DataFrame()

    # Convert 12 matrix params to a single list in 'M'
    matrix_cols = ['VC_11', 'VC_12', 'VC_13', 'VC_14', 'VC_21', 'VC_22','VC_23', 'VC_24', 'VC_31', 'VC_32', 'VC_33', 'VC_34']
    M = df[matrix_cols].values.tolist()
    df['M'] = M

    for k in range(times):
        df_temp = df[ref_time+1:1800]

        #split data to train, evaluation, testing
        df_tr = df_temp.sample(frac=par)
        df_valtest = pd.concat([df_temp,df_tr,df_tr]).drop_duplicates(subset='ScanStart',keep=False)
        df_val = df_valtest.sample(frac=0.5)
        df_test = pd.concat([df_valtest,df_val,df_val]).drop_duplicates(subset='ScanStart',keep=False)  
        df_tr['DataType'] = 'Train'
        df_val['DataType'] = 'Validation'
        df_test['DataType'] = 'Testing'
        df_sample = pd.concat([df_tr,df_val])
        df_sample = pd.concat([df_sample,df_test])
        df_sample = df_sample.sort_values(by='ScanStart')
        df_sample = df_sample.reset_index()


        # Only keep the necessary info
        df_sample = df_sample[active_cols]

        # Only keep the necessary info
        df_sample = df_sample[active_cols]

        df_ref = df.loc[[ref_time]]
        df_ref = df_ref[mov_cols]
        
        #build reference and moving time dataset
        for i in range(0,TimeSpan-ref_time-1):
            df_mov = df_sample.loc[[i]]
            df_mov = df_mov[active_cols]
            temp = df_ref.merge(df_mov, on=['PatientID','InjectionID','PatientType'], how='left')
            df_final = df_final.append(temp)
        ref_time= ref_time + ref_span

    #calculate T and delta_t   
    df_final['T'] = df_final.apply(lambda row: 
                                 vicra_toolbox.RotTransMatrix_6Params(
                                     Relative_motion_A_to_B_12(row['M_x'], row['M_y']), 1), axis=1
                                 )
    df_final['delta_t'] = df_final.apply(lambda row: row['ScanStart_y']-row['ScanStart_x'], axis=1)                
#     print(df_final.shape[0])
    return df_final

# New sampling strategies mixing random and normal sampling to balance the moving images distribution 

def mixed_mulsub_sampling(df):
    part1=[3600,3779]
    part2=[3780,3959]
    part3=[3960,4139]
    part4=[4140,4319]
    part5=[4320,4499]
    part6=[4500,4679]
    part7=[4680,5039]
    part8=[5040,5219]
    part9=[5220,5400]
    part10=[3600,4319]
    
    time_list1=np.linspace(part1[0], part1[1], part1[1]-part1[0]+1)
    time_list1=[int(x) for x in time_list1]

    time_list2=np.linspace(part2[0], part2[1], part2[1]-part2[0]+1)
    time_list2=[int(x) for x in time_list2]

    time_list3=np.linspace(part3[0], part3[1], part3[1]-part3[0]+1)
    time_list3=[int(x) for x in time_list3]

    time_list4=np.linspace(part4[0], part4[1], part4[1]-part4[0]+1)
    time_list4=[int(x) for x in time_list4]

    time_list5=np.linspace(part5[0], part5[1], part5[1]-part5[0]+1)
    time_list5=[int(x) for x in time_list5]

    time_list6=np.linspace(part6[0], part6[1], part6[1]-part6[0]+1)
    time_list6=[int(x) for x in time_list6]

    time_list7=np.linspace(part7[0], part7[1], part7[1]-part7[0]+1)
    time_list7=[int(x) for x in time_list7]

    time_list8=np.linspace(part8[0], part8[1], part8[1]-part8[0]+1)
    time_list8=[int(x) for x in time_list8]

    time_list9=np.linspace(part9[0], part9[1], part9[1]-part9[0]+1)
    time_list9=[int(x) for x in time_list9]

    time_list10=np.linspace(part10[0], part10[1], part10[1]-part10[0]+1)
    time_list10=[int(x) for x in time_list10]
    
    df_final1=data_sample(df,par=0.8,n=8,Start_t=3600,TimeSpan=1800,Random_Arg=85)
    df_final2=data_sample(df,par=0.8,n=6,Start_t=3600,TimeSpan=1800,Random_Arg=85)
    df_final3=data_sample(df,par=0.8,n=6,Start_t=3600,TimeSpan=1800,Random_Arg=85)
    df_final4=data_sample(df,par=0.8,n=5,Start_t=3600,TimeSpan=1800)
    df_final5=data_sample_random(df,par=0.8,n=6,Start_t=3600,TimeSpan=1800)
    df_final6=data_sample_random(df,par=0.8,n=6,Start_t=3600,TimeSpan=1800)
    df_final7=data_sample_random(df,par=0.8,n=6,Start_t=3600,TimeSpan=1800)
    df_final8=data_sample_random(df,par=0.8,n=6,Start_t=3600,TimeSpan=1800)
    df_final9=data_sample_random(df,par=0.8,n=6,Start_t=3600,TimeSpan=1800)
    df_final10=data_sample_random(df,par=0.8,n=2,Start_t=3600,TimeSpan=1800)
    
    df_final1_part=df_final1[df_final1['ScanStart_x'].isin(time_list1)].reset_index()
    df_final2_part=df_final2[df_final2['ScanStart_x'].isin(time_list2)].reset_index()
    df_final3_part=df_final3[df_final3['ScanStart_x'].isin(time_list3)].reset_index()
    df_final4_part=df_final4[df_final4['ScanStart_x'].isin(time_list4)].reset_index()
    df_final5_part=df_final5[df_final5['ScanStart_x'].isin(time_list5)].reset_index()
    df_final6_part=df_final6[df_final6['ScanStart_x'].isin(time_list5)].reset_index()
    df_final7_part=df_final7[df_final7['ScanStart_x'].isin(time_list5)].reset_index()
    df_final8_part=df_final8[df_final8['ScanStart_x'].isin(time_list5)].reset_index()
    df_final9_part=df_final9[df_final9['ScanStart_x'].isin(time_list5)].reset_index()
    df_final10_part=df_final10[df_final10['ScanStart_x'].isin(time_list5)].reset_index()
    
    df_new_sampling=pd.concat([df_final1_part,df_final2_part, df_final3_part, df_final4_part, df_final5_part,df_final6_part, df_final7_part, df_final8_part, df_final9_part, df_final10_part])
    
    return df_new_sampling


def mixed_singlesub_sampling(df):
    part1=[3600,3779]
    part2=[3780,3959]
    part3=[3960,4139]
    part4=[4140,4319]
    part5=[4320,4499]
    part6=[4500,4679]
    part7=[4680,5039]
    part8=[5040,5219]
    part9=[5220,5400]
    part10=[3600,4319]
    
    time_list1=np.linspace(part1[0], part1[1], part1[1]-part1[0]+1)
    time_list1=[int(x) for x in time_list1]

    time_list2=np.linspace(part2[0], part2[1], part2[1]-part2[0]+1)
    time_list2=[int(x) for x in time_list2]

    time_list3=np.linspace(part3[0], part3[1], part3[1]-part3[0]+1)
    time_list3=[int(x) for x in time_list3]

    time_list4=np.linspace(part4[0], part4[1], part4[1]-part4[0]+1)
    time_list4=[int(x) for x in time_list4]

    time_list5=np.linspace(part5[0], part5[1], part5[1]-part5[0]+1)
    time_list5=[int(x) for x in time_list5]

    time_list6=np.linspace(part6[0], part6[1], part6[1]-part6[0]+1)
    time_list6=[int(x) for x in time_list6]

    time_list7=np.linspace(part7[0], part7[1], part7[1]-part7[0]+1)
    time_list7=[int(x) for x in time_list7]

    time_list8=np.linspace(part8[0], part8[1], part8[1]-part8[0]+1)
    time_list8=[int(x) for x in time_list8]

    time_list9=np.linspace(part9[0], part9[1], part9[1]-part9[0]+1)
    time_list9=[int(x) for x in time_list9]

    time_list10=np.linspace(part10[0], part10[1], part10[1]-part10[0]+1)
    time_list10=[int(x) for x in time_list10]
    
    df_final1=data_sample(df,par=0.8,n=12,Start_t=3600,TimeSpan=1800,Random_Arg=85)
    df_final2=data_sample(df,par=0.8,n=9,Start_t=3600,TimeSpan=1800,Random_Arg=85)
    df_final3=data_sample(df,par=0.8,n=9,Start_t=3600,TimeSpan=1800,Random_Arg=85)
    df_final4=data_sample(df,par=0.8,n=7,Start_t=3600,TimeSpan=1800)
    df_final5=data_sample_random(df,par=0.8,n=9,Start_t=3600,TimeSpan=1800)
    df_final6=data_sample_random(df,par=0.8,n=8,Start_t=3600,TimeSpan=1800)
    df_final7=data_sample_random(df,par=0.8,n=8,Start_t=3600,TimeSpan=1800)
    df_final8=data_sample_random(df,par=0.8,n=8,Start_t=3600,TimeSpan=1800)
    df_final9=data_sample_random(df,par=0.8,n=8,Start_t=3600,TimeSpan=1800)
    df_final10=data_sample_random(df,par=0.8,n=2,Start_t=3600,TimeSpan=1800)
    
    df_final1_part=df_final1[df_final1['ScanStart_x'].isin(time_list1)].reset_index()
    df_final2_part=df_final2[df_final2['ScanStart_x'].isin(time_list2)].reset_index()
    df_final3_part=df_final3[df_final3['ScanStart_x'].isin(time_list3)].reset_index()
    df_final4_part=df_final4[df_final4['ScanStart_x'].isin(time_list4)].reset_index()
    df_final5_part=df_final5[df_final5['ScanStart_x'].isin(time_list5)].reset_index()
    df_final6_part=df_final6[df_final6['ScanStart_x'].isin(time_list5)].reset_index()
    df_final7_part=df_final7[df_final7['ScanStart_x'].isin(time_list5)].reset_index()
    df_final8_part=df_final8[df_final8['ScanStart_x'].isin(time_list5)].reset_index()
    df_final9_part=df_final9[df_final9['ScanStart_x'].isin(time_list5)].reset_index()
    df_final10_part=df_final10[df_final10['ScanStart_x'].isin(time_list5)].reset_index()
    
    df_new_sampling=pd.concat([df_final1_part,df_final2_part, df_final3_part, df_final4_part, df_final5_part,df_final6_part, df_final7_part, df_final8_part, df_final9_part, df_final10_part])
    
    return df_new_sampling



def data_split_sample(df, train_frac=1.0, val_frac=0.0, test_frac=0.0, ref_suffix="_ref", mov_suffix="_mov", force_first_frame=False):
    """Split and sample a DL-HMC dataset into independent sets.

    Returns three Dataframes containing reference and moving image pairs.
    The validation and testing Dataframes assume that the reference frame is always the earliest time point. This mimics the real inference procedure.

    Args:
    df: the dataframe containing all N datapoints.
    train_frac: the fraction of N data pairs to sample from the NxN reference and moving. This value can be larger than 1.0, e.g. 8.0 will sample 800% of the N values.
    val_frac: the fraction of N data pairs to use as validation data. Default 0.0.
    test_frac: the fraction of N data pairs to use as testing data. Default 0.0.
    ref_suffix: the suffix for reference image column names. Default "_ref".
    mov_suffix: the suffix for moving image column names. Default "_mov".
    force_first_frame: force the first frame to be in the training set. Default "False".
    
    Returns:
    Three Dataframes for training/validation/testing:
    df_train
    df_val (can be None)
    df_test (can be None)
    """

    # Set the train/val/test split

    n = len(df)

    # Create a vector of index values
    I = np.arange(n)
    # print('Original index shape:', I.shape)

    # Uniform random sampling of test and val set index
    # Random sample without replacement (no repeats)
    I_test = np.arange(0)
    df_test = None
    if test_frac>0.0:
        I_test = np.sort(np.random.choice(I, size=int(n*test_frac)-1, replace=False))
        # Always add 0-th index back in
        I_test = np.insert(I_test, 0, [0])
        # print('Test index shape:', I_test.shape)
        # print('Test: ', I_test)

        # Test data
        df_test_mov = df.iloc[I_test[1:]].reset_index(drop=True)
        df_test_ref = pd.concat([df.iloc[I_test[0]].to_frame().T]*len(df_test_mov), ignore_index=True).reset_index(drop=True)
        df_test = df_test_ref.join(df_test_mov, lsuffix=ref_suffix, rsuffix=mov_suffix)



    I_val = np.arange(0)
    df_val = None
    if val_frac>0.0:
        # Remove the test sample indices
        I_valid = np.delete(I, I_test)
        # print('Valid index shape:', I_valid.shape)
        I_val = np.sort(np.random.choice(I_valid, size=int(n*val_frac), replace=False))
        # print('Validation index shape:', I_val.shape)
        # print('Validation: ', I_val)

        # Validation data
        df_val_mov = df.iloc[I_val[1:]].reset_index(drop=True)
        df_val_ref = pd.concat([df.iloc[I_val[0]].to_frame().T]*len(df_val_mov), ignore_index=True).reset_index(drop=True)
        df_val = df_val_ref.join(df_val_mov, lsuffix=ref_suffix, rsuffix=mov_suffix)



    # Select the training data pairs from a matrix of possible values
    M = np.arange(n*n).reshape(n,n)
    # print(M.shape)
    # print(np.sum(M>0))
    # Remove all validation and test data indexes
    M[:,I_val] = 0
    # print(np.sum(M>0))
    M[:,I_test] = 0
    # print(np.sum(M>0))
    # Only keep the upper triangle
    M = np.triu(M)
    # print(np.sum(M>0))
    I_valid = M[M>0]
    # print(I_valid.shape)
    I_train = np.sort(np.random.choice(I_valid, size=int(n*train_frac), replace=False))
    if force_first_frame:
        I_train=np.insert(I_train, 0, [0])
    # print(I_train.shape)

    idx = np.unravel_index(I_train, shape=M.shape)

    # Return the split samples as Dataframes
    # Training data
    df_train_ref = df.iloc[idx[0]].reset_index(drop=True)
    df_train_mov = df.iloc[idx[1]].reset_index(drop=True)
    df_train = df_train_ref.join(df_train_mov, lsuffix=ref_suffix, rsuffix=mov_suffix)


    # Return all train/val/test Dataframes
    return df_train, df_val, df_test


def add_T_deltaT(df_train_tmp, df_val_tmp, df_test_tmp):
    
    delta_T_list=[]
    for i in range(len(df_train_tmp)):
        delta_T_list.append(df_train_tmp['ScanStart_mov'][i]-df_train_tmp['ScanStart_ref'][i])

    df_train_tmp['delta_t']=delta_T_list

    delta_T_list=[]
    for i in range(len(df_val_tmp)):
        delta_T_list.append(df_val_tmp['ScanStart_mov'][i]-df_val_tmp['ScanStart_ref'][i])

    df_val_tmp['delta_t']=delta_T_list

    delta_T_list=[]
    for i in range(len(df_test_tmp)):
        delta_T_list.append(df_test_tmp['ScanStart_mov'][i]-df_test_tmp['ScanStart_ref'][i])

    df_test_tmp['delta_t']=delta_T_list

    df_train_tmp['T'] = df_train_tmp.apply(lambda row: 
                                     vicra_toolbox.RotTransMatrix_6Params(
                                         Relative_motion_A_to_B_12(row['M_ref'], row['M_mov']), 1), axis=1
                                     )
    df_val_tmp['T'] = df_val_tmp.apply(lambda row: 
                                     vicra_toolbox.RotTransMatrix_6Params(
                                         Relative_motion_A_to_B_12(row['M_ref'], row['M_mov']), 1), axis=1
                                     )
    df_test_tmp['T'] = df_test_tmp.apply(lambda row: 
                                     vicra_toolbox.RotTransMatrix_6Params(
                                         Relative_motion_A_to_B_12(row['M_ref'], row['M_mov']), 1), axis=1
                                     )
    
    return df_train_tmp, df_val_tmp, df_test_tmp

def add_T_deltaT_12(df_train_tmp, df_val_tmp, df_test_tmp):
        
    delta_T_list=[]
    for i in range(len(df_train_tmp)):
        delta_T_list.append(df_train_tmp['ScanStart_mov'][i]-df_train_tmp['ScanStart_ref'][i])

    df_train_tmp['delta_t']=delta_T_list

    delta_T_list=[]
    for i in range(len(df_val_tmp)):
        delta_T_list.append(df_val_tmp['ScanStart_mov'][i]-df_val_tmp['ScanStart_ref'][i])

    df_val_tmp['delta_t']=delta_T_list

    delta_T_list=[]
    for i in range(len(df_test_tmp)):
        delta_T_list.append(df_test_tmp['ScanStart_mov'][i]-df_test_tmp['ScanStart_ref'][i])

    df_test_tmp['delta_t']=delta_T_list

    df_train_tmp['T'] = df_train_tmp.apply(lambda row: 
                                     vicra_toolbox.Relative_motion_A_to_B_12(row['M_ref'], row['M_mov']),axis=1)
    df_val_tmp['T'] = df_val_tmp.apply(lambda row: 
                                     vicra_toolbox.Relative_motion_A_to_B_12(row['M_ref'], row['M_mov']),axis=1)
    df_test_tmp['T'] = df_test_tmp.apply(lambda row: 
                                     vicra_toolbox.Relative_motion_A_to_B_12(row['M_ref'], row['M_mov']),axis=1)
    
    return df_train_tmp, df_val_tmp, df_test_tmp