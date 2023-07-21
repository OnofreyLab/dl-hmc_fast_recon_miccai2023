import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from numpy.linalg import inv

# ----------------------------------------------------------------------- 
# **NOTE #1: METHOD 1**
#    One liner: Rotation/Translation to Decomposed 6 elements

#    INPUT    : 12 Vicra elements (Vicra position 2-13)
#    Order    : R11,R12,R13,T14,R21,R22,R23,T24,R31,R32,R33,T34

#    OUTPUT   : Decomposed 6 parameters 
#    Order    : Tx,Ty,Tz,Rx,Ry,Rz

# **NOTE #2: METHOD 2**
#    One liner: Decomposed 6 elements to Rotation/Translation

#    INPUT    : Decomposed 6 parameters 
#    Order    : Tx,Ty,Tz,Rx,Ry,Rz

#    OUTPUT   : 12 Vicra elements (Vicra position 2-13) 
#    Order    : R11,R12,R13,T14,R21,R22,R23,T24,R31,R32,R33,T34
# ----------------------------------------------------------------------- 

def RotTransMatrix_6Params(Input_elements, Method):
    if Method == 1:
        decomposed_transMatrix = np.zeros((6))
        decomposed_transMatrix[0] = Input_elements[3]
        decomposed_transMatrix[1] = Input_elements[7]
        decomposed_transMatrix[2] = Input_elements[11]
        
        Rot_Mat = R.from_matrix([
            [Input_elements[0], Input_elements[1], Input_elements[2]],
            [Input_elements[4], Input_elements[5], Input_elements[6]],
            [Input_elements[8], Input_elements[9], Input_elements[10]]
        ])
        eul = Rot_Mat.as_euler('XYZ', degrees=True)

        decomposed_transMatrix[3] = eul[0]
        decomposed_transMatrix[4] = eul[1]
        decomposed_transMatrix[5] = eul[2]
        
        Output = decomposed_transMatrix
        return Output
    
    elif Method == 2:
        degrees_xyz = Input_elements[3:6]
        eul = R.from_euler('XYZ',degrees_xyz, degrees=True)
        rotation_matrix_from_decomposed = eul.as_matrix()
        
        one = rotation_matrix_from_decomposed[0][0]
        two = rotation_matrix_from_decomposed[0][1]
        three = rotation_matrix_from_decomposed[0][2]
        four = Input_elements[0]
        five =rotation_matrix_from_decomposed[1][0]
        six = rotation_matrix_from_decomposed[1][1]
        seven = rotation_matrix_from_decomposed[1][2]
        eight = Input_elements[1]
        nine = rotation_matrix_from_decomposed[2][0]
        ten = rotation_matrix_from_decomposed[2][1]
        eleven = rotation_matrix_from_decomposed[2][2]
        twelve = Input_elements[2]

        Output = np.array([one,two,three,four,five,six,seven,eight,nine,ten,eleven,twelve])
        return Output
        
    else:
        Output = [];
        print('Input method is not recognized in this function. Please choose 1 or 2.')
        print('1 = Rotation/Translation to 6 elements')
        print('2 = 6 elements to Rotation/Translation')
        return Output
    


# A function to compute the relative motion between frame A and frame B.
# Assume you have vc for A to reference and vc for B to reference, we need
# a function to compute A to B, i.e., A is the new reference and B is the
# new moving.

# Input : 
#     Frame002 is your new REFERENCE  --> input img_vc1 (14 elements)
#     Frame006 is your new MOVE FRAME --> input img_vc2 (14 elements)
#     Order : Time,R11,R12,R13,T14,R21,R22,R23,T24,R31,R32,R33,T34,uncertainty

# Output:
#     One line VC format (14 elements) - relative motion from frame B to frame A
#     Order : Time,R11,R12,R13,T14,R21,R22,R23,T24,R31,R32,R33,T34,uncertainty

# Example:
# R : reference gate;
# A : moving gate1;
# B : moving gate2;

# R = img_vc1 * A;
# R = img_vc2 * B;
# => A = inv(img_vc1) * img_vc2 * B

# Output: output_matr = inv(img_vc1) * img_vc2

def Relative_motion_A_to_B(img_vc1, img_vc2):
    # calcualte relative motion
    img_matr1 = np.array([
            [img_vc1[1],   img_vc1[2],   img_vc1[3],   img_vc1[4]],
            [img_vc1[5],   img_vc1[6],   img_vc1[7],   img_vc1[8]],
            [img_vc1[9],  img_vc1[10],  img_vc1[11],  img_vc1[12]],
            [0, 0, 0, 1]
        ],dtype=float)
    img_matr1_inv = inv(img_matr1)
    
    img_matr2 = np.array([
            [img_vc2[1],   img_vc2[2],   img_vc2[3],   img_vc2[4]],
            [img_vc2[5],   img_vc2[6],   img_vc2[7],   img_vc2[8]],
            [img_vc2[9],  img_vc2[10],  img_vc2[11],  img_vc2[12]],
            [0, 0, 0, 1]
        ],dtype=float)
    
    matr_move = np.dot(img_matr1_inv, img_matr2)
    matr_move_reshape = np.reshape(matr_move, (1,16))
    output_matr = np.insert(matr_move_reshape[0][0:12], 0, 0)
    output_matr = np.append(output_matr, 1)
    
    return output_matr



# ARGUMENTS: 
#   ** Input **
#   img_vc1  : 1 line of VC, 14 elements
#   img_vc2  : 1 line of VC, 14 elements
#   Order    : Time,R11,R12,R13,T14,R21,R22,R23,T24,R31,R32,R33,T34,uncertainty

#   ** Output **
#   MOLAR_VC_matrix_full : resulting transformation matrix (1 line of VC, 14 elements)
#   Order                : Time,R11,R12,R13,T14,R21,R22,R23,T24,R31,R32,R33,T34,uncertainty

#   VC_6_params          : resulting 6 parameters (translation and rotation xyz)
#   Order                : Tx,Ty,Tz,Rx,Ry,Rz

# Example:
# R : reference gate;
# A : moving gate1;
# B : moving gate2;

# R = img_vc1 * A;
# A = img_vc2 * B;
# => R = img_vc1 * img_vc2 * B

# Output: MOLAR_VC_matrix_full = img_vc1 * img_vc2

def DL_HMC_concat_VC(img_vc1, img_vc2):
    img_matr1 = np.array([
            [img_vc1[1],   img_vc1[2],   img_vc1[3],   img_vc1[4]],
            [img_vc1[5],   img_vc1[6],   img_vc1[7],   img_vc1[8]],
            [img_vc1[9],  img_vc1[10],  img_vc1[11],  img_vc1[12]],
            [0, 0, 0, 1]
        ],dtype=float)
    img_matr1_inv = inv(img_matr1)
    
    img_matr2 = np.array([
            [img_vc2[1],   img_vc2[2],   img_vc2[3],   img_vc2[4]],
            [img_vc2[5],   img_vc2[6],   img_vc2[7],   img_vc2[8]],
            [img_vc2[9],  img_vc2[10],  img_vc2[11],  img_vc2[12]],
            [0, 0, 0, 1]
        ],dtype=float)
    
    matr_move = np.dot(img_matr1_inv, img_matr2)
    matr_move_reshape = np.reshape(matr_move, (1,16))
    MOLAR_VC_matrix_full = np.insert(matr_move_reshape[0][0:12], 0, 0)
    MOLAR_VC_matrix_full = np.append(MOLAR_VC_matrix_full, 1)
    
    # vc 6 parameters
    decomposed_transMatrix = np.zeros((6))
    decomposed_transMatrix[0] = matr_move_reshape[0][3]
    decomposed_transMatrix[1] = matr_move_reshape[0][7]
    decomposed_transMatrix[2] = matr_move_reshape[0][11]

    Rot_Mat = R.from_matrix([
        [matr_move_reshape[0][0], matr_move_reshape[0][1], matr_move_reshape[0][2]],
        [matr_move_reshape[0][4], matr_move_reshape[0][5], matr_move_reshape[0][6]],
        [matr_move_reshape[0][8], matr_move_reshape[0][9], matr_move_reshape[0][10]]
    ])
    eul = Rot_Mat.as_euler('XYZ', degrees=True)

    decomposed_transMatrix[3] = eul[0]
    decomposed_transMatrix[4] = eul[1]
    decomposed_transMatrix[5] = eul[2]

    Output = decomposed_transMatrix
    return MOLAR_VC_matrix_full, decomposed_transMatrix

#input 12 relative motion parameter and 3600 12 motion parameter
#output 6 motion parameter
def DL_HMC_concat_VC_zty(img_vc1, img_vc2):
    img_matr1 = np.array([
            [img_vc1[0],   img_vc1[1],   img_vc1[2],   img_vc1[3]],
            [img_vc1[4],   img_vc1[5],   img_vc1[6],   img_vc1[7]],
            [img_vc1[8],  img_vc1[9],  img_vc1[10],  img_vc1[11]],
            [0, 0, 0, 1]
        ],dtype=float)
    
    img_matr2 = np.array([
            [img_vc2[0],   img_vc2[1],   img_vc2[2],   img_vc2[3]],
            [img_vc2[4],   img_vc2[5],   img_vc2[6],   img_vc2[7]],
            [img_vc2[8],  img_vc2[9],  img_vc2[10],  img_vc2[11]],
            [0, 0, 0, 1]
        ],dtype=float)
    
    matr_move = np.dot(img_matr1, img_matr2)
    matr_move_reshape = np.reshape(matr_move, (1,16))
    MOLAR_VC_matrix_full = matr_move_reshape[0][0:12]
    
    # vc 6 parameters
    decomposed_transMatrix = np.zeros((6))
    decomposed_transMatrix[0] = matr_move_reshape[0][3]
    decomposed_transMatrix[1] = matr_move_reshape[0][7]
    decomposed_transMatrix[2] = matr_move_reshape[0][11]

    Rot_Mat = R.from_matrix([
        [matr_move_reshape[0][0], matr_move_reshape[0][1], matr_move_reshape[0][2]],
        [matr_move_reshape[0][4], matr_move_reshape[0][5], matr_move_reshape[0][6]],
        [matr_move_reshape[0][8], matr_move_reshape[0][9], matr_move_reshape[0][10]]
    ])
    eul = Rot_Mat.as_euler('XYZ', degrees=True)

    decomposed_transMatrix[3] = eul[0]
    decomposed_transMatrix[4] = eul[1]
    decomposed_transMatrix[5] = eul[2]

    Output = decomposed_transMatrix
    return decomposed_transMatrix

def Relative_motion_A_to_B_12(img_vc1, img_vc2):
    A = np.pad(img_vc1, pad_width=1)
    B = np.pad(img_vc2, pad_width=1)
    R = Relative_motion_A_to_B(A, B)
    return R[1:13]

# Input: 1. Summary.csy； 2. delt(t) distribution y=f(x); 3. batch_size (10000); 4. partition: eg, 8:1:1; 5.n, how many delta t for every refernce t
# Output:A csv file includes 10000 lines of ['PatientID','InjectionID','PatientType', 'ThreeD_Cloud_x', 'ThreeD_Cloud_y', 'COD_nomask_x', 'COD_nomask_y', 'T', 'delta_t']
def data_sample(df,par=0.8,n=10,Start_t=3600,TimeSpan=1800,Random_Arg=85):
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
    print(df_final.shape[0])
    return df_final
    #return df_final
# pd.csvsave()

def build_netinput_diff1(df, Start_t=3600, TimeSpan=1800):
    """
    Randomly sample head motion data.
    
    """
    #new
    df_sample = df
    
    # Cols to keep
    active_cols = ['PatientID','InjectionID','PatientType','ScanStart',
                   'ThreeD_Cloud', 'COD_nomask','M']
    final_cols = ['PatientID','InjectionID','PatientType', 'ScanStart_x', 'ScanStart_y',
                   'ThreeD_Cloud_x', 'ThreeD_Cloud_y', 'COD_nomask_x', 'COD_nomask_y', 'T', 'delta_t']
    mov_cols = ['PatientID','InjectionID','PatientType','ScanStart',
                   'ThreeD_Cloud', 'COD_nomask','M']

    # Convert 12 matrix params to a single list in 'M'
    matrix_cols = ['VC_11', 'VC_12', 'VC_13', 'VC_14', 'VC_21', 'VC_22','VC_23', 'VC_24', 'VC_31', 'VC_32', 'VC_33', 'VC_34']
    M = df_sample[matrix_cols].values.tolist()
    df_sample['M'] = M
    
    # Only keep the necessary info
    df_sample = df_sample[active_cols]
    
    # Sort by ScanStart time
    df_sample = df_sample.sort_values(by='ScanStart')
    
    End_t = TimeSpan + Start_t
    reference_t = Start_t - 3600
    #build reference and moving time dataset
    df_ref = df_sample.loc[[reference_t]]
    mov_time = reference_t
    df_mov = df_sample.loc[[mov_time]]
    df_mov = df_mov[mov_cols]
    temp = df_ref.merge(df_mov, on=['PatientID','InjectionID','PatientType'], how='left')    
    df_final = temp
    for i in range(0,TimeSpan-1):
        ref_time = df_sample.at[i,'ScanStart']
        df_ref = df_sample.loc[[i]]
        mov_time = ref_time + 1
        df_mov = df_sample.loc[[mov_time-Start_t]]
        df_mov = df_mov[mov_cols]
        temp = df_ref.merge(df_mov, on=['PatientID','InjectionID','PatientType'], how='left')    
        df_final = pd.concat([df_final,temp])
    #calculate T and delta_t   
    df_final['T'] = df_final.apply(lambda row: 
                                     Relative_motion_A_to_B_12(row['M_x'], row['M_y']), axis=1)
    df_final['delta_t'] = df_final.apply(lambda row: row['ScanStart_y']-row['ScanStart_x'], axis=1)              
    #print(df_final.shape[0])
    df_final = df_final[final_cols]
    df_final=df_final.rename(columns={'ScanStart_x':'ScanStart_ref', 'ScanStart_y':'ScanStart_mov', 'ThreeD_Cloud_x':'ThreeD_Cloud_ref', 'ThreeD_Cloud_y':'ThreeD_Cloud_mov', 'COD_nomask_x':'COD_nomask_ref', 'COD_nomask_y':'COD_nomask_mov'})
    
    return df_final

def build_netinput_fixed_reference_12(df, Start_t=3600, TimeSpan=1800):
    """
    Randomly sample head motion data.
    
    """
    df_sample = df
    
    # Cols to keep
    active_cols = ['PatientID','InjectionID','PatientType','ScanStart',
                   'ThreeD_Cloud', 'COD_nomask','M']
    final_cols = ['PatientID','InjectionID','PatientType', 'ScanStart_x', 'ScanStart_y',
                   'ThreeD_Cloud_x', 'ThreeD_Cloud_y', 'COD_nomask_x', 'COD_nomask_y', 'T', 'delta_t']
    mov_cols = ['PatientID','InjectionID','PatientType','ScanStart',
                   'ThreeD_Cloud', 'COD_nomask','M']

    # Convert 12 matrix params to a single list in 'M'
    matrix_cols = ['VC_11', 'VC_12', 'VC_13', 'VC_14', 'VC_21', 'VC_22','VC_23', 'VC_24', 'VC_31', 'VC_32', 'VC_33', 'VC_34']
    M = df_sample[matrix_cols].values.tolist()
    df_sample['M'] = M
    
    # Only keep the necessary info
    df_sample = df_sample[active_cols]
    
    # Sort by ScanStart time
    df_sample = df_sample.sort_values(by='ScanStart')
    End_t = TimeSpan + Start_t
    reference_t = Start_t - 3600
    
    #build reference and moving time dataset
    df_ref = df_sample.loc[[reference_t]]
    mov_time = reference_t
    df_mov = df_sample.loc[[mov_time]]
    df_mov = df_mov[mov_cols]
    temp = df_ref.merge(df_mov, on=['PatientID','InjectionID','PatientType'], how='left')    
    df_final = temp
    for i in range(0,TimeSpan-1):
        df_ref = df_sample.loc[[reference_t]]
        mov_time = i+1+reference_t
        df_mov = df_sample.loc[[mov_time]]
        df_mov = df_mov[mov_cols]
        temp = df_ref.merge(df_mov, on=['PatientID','InjectionID','PatientType'], how='left')    
        df_final = pd.concat([df_final,temp])
    #calculate T and delta_t   
    df_final['T'] = df_final.apply(lambda row: 
                                     Relative_motion_A_to_B_12(row['M_x'], row['M_y']), axis=1)
    df_final['delta_t'] = df_final.apply(lambda row: row['ScanStart_y']-row['ScanStart_x'], axis=1)              
    df_final = df_final[final_cols]
    df_final=df_final.rename(columns={'ScanStart_x':'ScanStart_ref', 'ScanStart_y':'ScanStart_mov', 'ThreeD_Cloud_x':'ThreeD_Cloud_ref', 'ThreeD_Cloud_y':'ThreeD_Cloud_mov', 'COD_nomask_x':'COD_nomask_ref', 'COD_nomask_y':'COD_nomask_mov'})
    
    return df_final

def build_netinput_fixed_reference(df, Start_t=3600, TimeSpan=1800):
    """
    Randomly sample head motion data.
    
    """
    df_sample = df
    
    # Cols to keep
    active_cols = ['PatientID','InjectionID','PatientType','ScanStart',
                   'ThreeD_Cloud', 'COD_nomask','M']
    final_cols = ['PatientID','InjectionID','PatientType', 'ScanStart_x', 'ScanStart_y',
                   'ThreeD_Cloud_x', 'ThreeD_Cloud_y', 'COD_nomask_x', 'COD_nomask_y', 'T', 'delta_t']
    mov_cols = ['PatientID','InjectionID','PatientType','ScanStart',
                   'ThreeD_Cloud', 'COD_nomask','M']

    # Convert 12 matrix params to a single list in 'M'
    matrix_cols = ['VC_11', 'VC_12', 'VC_13', 'VC_14', 'VC_21', 'VC_22','VC_23', 'VC_24', 'VC_31', 'VC_32', 'VC_33', 'VC_34']
    M = df_sample[matrix_cols].values.tolist()
    df_sample['M'] = M
    
    # Only keep the necessary info
    df_sample = df_sample[active_cols]
    
    # Sort by ScanStart time
    df_sample = df_sample.sort_values(by='ScanStart')
    End_t = TimeSpan + Start_t
    reference_t = Start_t - 3600
    
    #build reference and moving time dataset
    df_ref = df_sample.loc[[reference_t]]
    mov_time = reference_t
    df_mov = df_sample.loc[[mov_time]]
    df_mov = df_mov[mov_cols]
    temp = df_ref.merge(df_mov, on=['PatientID','InjectionID','PatientType'], how='left')    
    df_final = temp
    for i in range(0,TimeSpan-1):
        df_ref = df_sample.loc[[reference_t]]
        mov_time = i+1+reference_t
        df_mov = df_sample.loc[[mov_time]]
        df_mov = df_mov[mov_cols]
        temp = df_ref.merge(df_mov, on=['PatientID','InjectionID','PatientType'], how='left')    
        df_final = pd.concat([df_final,temp])
    #calculate T and delta_t   
    df_final['T'] = df_final.apply(lambda row: 
                                 RotTransMatrix_6Params(
                                     Relative_motion_A_to_B_12(row['M_x'], row['M_y']), 1), axis=1
                                 )
    df_final['delta_t'] = df_final.apply(lambda row: row['ScanStart_y']-row['ScanStart_x'], axis=1)              
    df_final = df_final[final_cols]
    df_final=df_final.rename(columns={'ScanStart_x':'ScanStart_ref', 'ScanStart_y':'ScanStart_mov', 'ThreeD_Cloud_x':'ThreeD_Cloud_ref', 'ThreeD_Cloud_y':'ThreeD_Cloud_mov', 'COD_nomask_x':'COD_nomask_ref', 'COD_nomask_y':'COD_nomask_mov'})
    
    return df_final
