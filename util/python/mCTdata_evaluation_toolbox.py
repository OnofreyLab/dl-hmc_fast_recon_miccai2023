import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import seaborn as sns
import sys
sys.path.append(r'../util/python')
import vicra_toolbox

# Build a df containing the patient ID, tracer, time, vicra and model parameters,for 12 output 
def build_df_results_12(test_set, df_input_diff_all, predictions):
    
    """Build a df containing the vicra and predicted parameters at each time step for all testing patients. 

    Args:
    test_set: the list of all testing patients.
    df_input_diff_all: a list of dfs (one for each patient) containing the path to the cloud images, the delta_t, etc for t=3601s
    to t=5399s.
    predictions: a list containing a prediction list (the 6 parameters at each moment) for each testing patient. 
    
    Returns:
    df_pred_all: the df containing vicra and model predictions at each moment for every testing patient. 
    """
    df_pred_all=[]
    
    for i in range(len(test_set)):
        
        df_res_temp=df_input_diff_all[i]
        if 'index' in df_res_temp:
            del df_res_temp['index']
    
        # Recover tracer name and summary path from the 3D cloud file paths

        tracers=[]

        for k in range(len(df_res_temp)):
            path_split=df_res_temp['ThreeD_Cloud_ref'][k].split('/')
            tracers.append(path_split[6])

        df_res_temp['tracer']=tracers
#         df_res_temp['summary_path']=summary_path
        summary_path=df_res_temp['ThreeD_Cloud_ref'][k].split('3D_Clouds_nii')[0] + 'Summary_'+str(df_res_temp['PatientID'][0])+'_3600_5400.csv'
        summary=pd.read_csv(summary_path)
        matrix_cols = ['VC_11', 'VC_12', 'VC_13', 'VC_14', 'VC_21', 'VC_22','VC_23', 'VC_24', 'VC_31', 'VC_32', 'VC_33', 'VC_34']
        M = summary[matrix_cols].values.tolist()
        summary['MATRIX'] = M
        #12 parameters (M) => 6 parameters (T)
        summary['T'] = summary['MATRIX'].apply(lambda x: vicra_toolbox.RotTransMatrix_6Params(x, 1))
        df_res_temp['vicra parameters']=summary['T']
    
        time=[]
        for l in range(1800):
            time.append(3600+df_res_temp['delta_t'][l])
        df_res_temp['time']=time
    
        # Delete the columns that are not useful

        del df_res_temp['InjectionID']
        del df_res_temp['PatientType']
        del df_res_temp['ThreeD_Cloud_ref']
        del df_res_temp['ThreeD_Cloud_mov']
        del df_res_temp['COD_nomask_ref']
        del df_res_temp['COD_nomask_mov']
        del df_res_temp['delta_t']
        del df_res_temp['T']

        #Add a column containing the network predictions 
        
        df_res_temp['model parameters'] = predictions[1800*i:1800*(i+1)]
        parameter13_3600 = vicra_toolbox.RotTransMatrix_6Params(df_res_temp.at[0,'vicra parameters'], 2)
        df_res_temp['model parameters'] = df_res_temp['model parameters'].apply(lambda x: vicra_toolbox.DL_HMC_concat_VC_zty(parameter13_3600,x)) 
        if 'level_0' in df_res_temp:
            del df_res_temp['level_0']
            
        df_pred_all.append(df_res_temp)
            
    df_pred_fin=df_pred_all[0]
    for j in range(len(df_pred_all)-1):
        df_pred_fin=pd.concat([df_pred_fin,df_pred_all[j+1]])

    return df_pred_fin

# Build a df containing the patient ID, tracer, time, vicra and model parameters,for 12 and continuous output 
def build_df_results_12_continuous(test_set, df_input_diff_all, predictions):
    
    """Build a df containing the vicra and predicted parameters at each time step for all testing patients. 

    Args:
    test_set: the list of all testing patients.
    df_input_diff_all: a list of dfs (one for each patient) containing the path to the cloud images, the delta_t, etc for t=3601s
    to t=5399s.
    predictions: a list containing a prediction list (the 6 parameters at each moment) for each testing patient. 
    
    Returns:
    df_pred_all: the df containing vicra and model predictions at each moment for every testing patient. 
    """
    df_pred_all=[]
    
    for i in range(len(test_set)):
        
        df_res_temp=df_input_diff_all[i]
        if 'index' in df_res_temp:
            del df_res_temp['index']
    
        # Recover tracer name and summary path from the 3D cloud file paths

        tracers=[]

        for k in range(len(df_res_temp)):
            path_split=df_res_temp['ThreeD_Cloud_ref'][k].split('/')
            tracers.append(path_split[6])

        df_res_temp['tracer']=tracers
#         df_res_temp['summary_path']=summary_path
        summary_path=df_res_temp['ThreeD_Cloud_ref'][k].split('3D_Clouds_nii')[0] + 'Summary_'+str(df_res_temp['PatientID'][0])+'_3600_5400.csv'
        summary=pd.read_csv(summary_path)
        matrix_cols = ['VC_11', 'VC_12', 'VC_13', 'VC_14', 'VC_21', 'VC_22','VC_23', 'VC_24', 'VC_31', 'VC_32', 'VC_33', 'VC_34']
        M = summary[matrix_cols].values.tolist()
        summary['MATRIX'] = M
        #12 parameters (M) => 6 parameters (T)
        summary['T'] = summary['MATRIX'].apply(lambda x: vicra_toolbox.RotTransMatrix_6Params(x, 1))
        df_res_temp['vicra parameters']=summary['T']
    
        time=[]
        for l in range(1800):
            time.append(3600+df_res_temp['delta_t'][l])
        df_res_temp['time']=time
    
        # Delete the columns that are not useful

        del df_res_temp['InjectionID']
        del df_res_temp['PatientType']
        del df_res_temp['ThreeD_Cloud_ref']
        del df_res_temp['ThreeD_Cloud_mov']
        del df_res_temp['COD_nomask_ref']
        del df_res_temp['COD_nomask_mov']
        del df_res_temp['delta_t']
        del df_res_temp['T']

        #Add a column containing the network predictions 
        
        df_res_temp['model parameters 12'] = predictions[1800*i:1800*(i+1)]
        df_res_temp['model parameters'] = df_res_temp['vicra parameters']
        parameter13_3600 = vicra_toolbox.RotTransMatrix_6Params(df_res_temp.at[0,'vicra parameters'], 2)
        df_res_temp.at[0,'model parameters'] = vicra_toolbox.DL_HMC_concat_VC_zty(parameter13_3600,df_res_temp.at[0,'model parameters 12']) 
        for j in range(1799):
            parameter13_3600=vicra_toolbox.RotTransMatrix_6Params(df_res_temp.at[i,'model parameters'], 2)
            df_res_temp.at[i+1,'model parameters'] = vicra_toolbox.DL_HMC_concat_VC_zty(parameter13_3600,df_res_temp.at[i+1,'model parameters 12']) 
        if 'level_0' in df_res_temp:
            del df_res_temp['level_0']
            
        df_pred_all.append(df_res_temp)
            
    df_pred_fin=df_pred_all[0]
    for j in range(len(df_pred_all)-1):
        df_pred_fin=pd.concat([df_pred_fin,df_pred_all[j+1]])

    return df_pred_fin

# Build a df containing the patient ID, tracer, time, vicra and model parameters

def build_df_results(test_set, df_input_diff_all, predictions):
    
    """Build a df containing the vicra and predicted parameters at each time step for all testing patients. 

    Args:
    test_set: the list of all testing patients.
    df_input_diff_all: a list of dfs (one for each patient) containing the path to the cloud images, the delta_t, etc for t=3601s
    to t=5399s.
    predictions: a list containing a prediction list (the 6 parameters at each moment) for each testing patient. 
    
    Returns:
    df_pred_all: the df containing vicra and model predictions at each moment for every testing patient. 
    """
    df_pred_all=[]
    
    for i in range(len(test_set)):
        
        df_res_temp=df_input_diff_all[i].copy()
        if 'index' in df_res_temp:
            del df_res_temp['index']
    
        # Recover tracer name and summary path from the 3D cloud file paths

        tracers=[]

        for k in range(len(df_res_temp)):
            path_split=df_res_temp['ThreeD_Cloud_ref'][k].split('/')
            tracers.append(path_split[6])

        df_res_temp['tracer']=tracers
#         df_res_temp['summary_path']=summary_path
        summary_path=df_res_temp['ThreeD_Cloud_ref'][k].split('3D_Clouds_nii')[0] + 'Summary_'+str(df_res_temp['PatientID'][0])+'_3600_5400.csv'
        summary=pd.read_csv(summary_path)
        matrix_cols = ['VC_11', 'VC_12', 'VC_13', 'VC_14', 'VC_21', 'VC_22','VC_23', 'VC_24', 'VC_31', 'VC_32', 'VC_33', 'VC_34']
        M = summary[matrix_cols].values.tolist()
        summary['MATRIX'] = M
        #12 parameters (M) => 6 parameters (T)
        summary['T'] = summary['MATRIX'].apply(lambda x: vicra_toolbox.RotTransMatrix_6Params(x, 1))
        df_res_temp['vicra parameters']=summary['T']
    
        time=[]
        for l in range(1800):
            time.append(3600+df_res_temp['delta_t'][l])
        df_res_temp['time']=time
    
        # Delete the columns that are not useful

        del df_res_temp['InjectionID']
        del df_res_temp['PatientType']
        del df_res_temp['ThreeD_Cloud_ref']
        del df_res_temp['ThreeD_Cloud_mov']
        del df_res_temp['COD_nomask_ref']
        del df_res_temp['COD_nomask_mov']
        del df_res_temp['delta_t']
        del df_res_temp['T']

        #Add a column containing the network predictions 
        
        df_res_temp['model parameters'] = predictions[1800*i:1800*(i+1)]
        df_res_temp['model parameters'] = df_res_temp['model parameters'].apply(lambda x: vicra_toolbox.RotTransMatrix_6Params(x, 2))
        parameter13_3600 = vicra_toolbox.RotTransMatrix_6Params(df_res_temp.at[0,'vicra parameters'], 2)
        df_res_temp['model parameters'] = df_res_temp['model parameters'].apply(lambda x: vicra_toolbox.DL_HMC_concat_VC_zty(parameter13_3600,x)) 
        if 'level_0' in df_res_temp:
            del df_res_temp['level_0']
            
        df_pred_all.append(df_res_temp)
            
    df_pred_fin=df_pred_all[0]
    for j in range(len(df_pred_all)-1):
        df_pred_fin=pd.concat([df_pred_fin,df_pred_all[j+1]])

    return df_pred_fin

def build_df_results_6ref(test_set, df_input_diff_all, predictions):
    
    """Build a df containing the vicra and predicted parameters at each time step for all testing patients. 

    Args:
    test_set: the list of all testing patients.
    df_input_diff_all: a list of dfs (one for each patient) containing the path to the cloud images, the delta_t, etc for t=3601s
    to t=5399s.
    predictions: a list containing a prediction list (the 6 parameters at each moment) for each testing patient. 
    
    Returns:
    df_pred_all: the df containing vicra and model predictions at each moment for every testing patient. 
    """
    df_pred_all=[]
    
    for i in range(len(test_set)):
        
        df_res_temp=df_input_diff_all[i].copy()
        if 'index' in df_res_temp:
            del df_res_temp['index']
    
        # Recover tracer name and summary path from the 3D cloud file paths

        tracers=[]

        for k in range(len(df_res_temp)):
            path_split=df_res_temp['ThreeD_Cloud_ref'][k].split('/')
            tracers.append(path_split[6])

        df_res_temp['tracer']=tracers
#         df_res_temp['summary_path']=summary_path
        summary_path=df_res_temp['ThreeD_Cloud_ref'][k].split('3D_Clouds_nii')[0] + 'Summary_'+str(df_res_temp['PatientID'][0])+'_3600_5400.csv'
        summary=pd.read_csv(summary_path)
        matrix_cols = ['VC_11', 'VC_12', 'VC_13', 'VC_14', 'VC_21', 'VC_22','VC_23', 'VC_24', 'VC_31', 'VC_32', 'VC_33', 'VC_34']
        M = summary[matrix_cols].values.tolist()
        summary['MATRIX'] = M
        #12 parameters (M) => 6 parameters (T)
        summary['T'] = summary['MATRIX'].apply(lambda x: vicra_toolbox.RotTransMatrix_6Params(x, 1))
        df_res_temp['vicra parameters']=summary['T']

        time=[]
        for l in range(1800):
            time.append(3600+l)
        df_res_temp['time']=time

        # Delete the columns that are not useful
        del df_res_temp['InjectionID']
        del df_res_temp['PatientType']
        del df_res_temp['ThreeD_Cloud_ref']
        del df_res_temp['ThreeD_Cloud_mov']
        del df_res_temp['COD_nomask_ref']
        del df_res_temp['COD_nomask_mov']
        del df_res_temp['delta_t']
        del df_res_temp['T']

        df_res_temp['model parameters'] = predictions[1800*i:1800*(i+1)]
        #Add a column containing the network predictions 
        for jj in range(1800): 
            df_res_temp.at[jj,'model parameters'] = vicra_toolbox.RotTransMatrix_6Params(df_res_temp.at[jj,'model parameters'], 2)
            temp_index = jj // 300
            parameter13_3600 = vicra_toolbox.RotTransMatrix_6Params(df_res_temp.at[temp_index*300,'vicra parameters'], 2)
            df_res_temp.at[jj,'model parameters'] = vicra_toolbox.DL_HMC_concat_VC_zty(parameter13_3600,df_res_temp.at[jj,'model parameters']) 
        if 'level_0' in df_res_temp:
            del df_res_temp['level_0']
            
        df_pred_all.append(df_res_temp)
            
    df_pred_fin=df_pred_all[0]
    for j in range(len(df_pred_all)-1):
        df_pred_fin=pd.concat([df_pred_fin,df_pred_all[j+1]])

    return df_pred_fin

def show_df_loss(df_loss):
    
    """Display statistical indicators about the loss (mean, std, ...) and plot it. 

    Args:
    df_loss: the df containing the losses between vicra and the network at different moments given by the "Time" column.
    """
        
    print(df_loss.describe())
    
    plt.plot(figsize=(8,5))
    plt.title("Testing loss")
#     plt.xlabel('Time(s)')
    plt.ylabel('Loss (RMS)')
    df_loss['Loss'].plot()
    plt.show()
    
    return 

def plot_vicra_network(tracer, patient, df_results, time_window, network_name, trans_range=[-10,10], rot_range=[-4,4]):
    
    """Plot the translational and rotational motions in the x, y and z-directions during a time window of your choice. One curve for
    the vicra parameters and one for the network prediction.
    Args:
    tracer: a string to specify the tracer since some patients were injected with different tracers.
    patient: the patient id (string).
    df_results: the df built with the build_df_results function.
    time_window: the time interval during which the motion will be ploted. 
    network_name: the name of the network you used to predict the motion (just for the graph legends).
    trans_range: the range of the y axis plot values for translation (for uniform visualization among subjects). Default is [-10,10] 
    rot_range: the range of the y axis plot values for rotation (for uniform visualization among subjects). Default is [-4,4]
    """
    
    # Select the part of df_results corresponding to the inputs
    df_results_tracer=df_results.loc[df_results['tracer'].str.contains(tracer)]
    df_results_patient=df_results_tracer.loc[df_results_tracer['UniqueID'].str.contains(patient)]
    time_list=np.linspace(time_window[0], time_window[1], time_window[1]-time_window[0]+1)
    time_list=[int(x) for x in time_list]
    df_results_time = df_results_patient[df_results_patient['time'].isin(time_list)].reset_index()
    win_len=time_window[1]-time_window[0]
    
    x1=range(win_len+1)
    
    #build fake vicra dataset -- fixed reference 
    trans_x_n = np.zeros(win_len+1)
    trans_y_n = np.zeros(win_len+1)
    trans_z_n = np.zeros(win_len+1)
    rot_x_n = np.zeros(win_len+1)
    rot_y_n = np.zeros(win_len+1)
    rot_z_n = np.zeros(win_len+1)
    temp = df_results_time['vicra parameters'][0]
    trans_x_n[0] = temp[0]
    trans_y_n[0] = temp[1]
    trans_z_n[0] = temp[2]
    rot_x_n[0] = temp[3]
    rot_y_n[0] = temp[4]
    rot_z_n[0] = temp[5]
    for i in range(win_len):
        temp = df_results_time['model parameters'][i]
        trans_x_n[i+1] = temp[0]
        trans_y_n[i+1] = temp[1]
        trans_z_n[i+1] = temp[2]
        rot_x_n[i+1] = temp[3]
        rot_y_n[i+1] = temp[4]
        rot_z_n[i+1] = temp[5]

    # about vicra curve, use T of dp, add T in each row
    trans_x = np.zeros(win_len+1)
    trans_y = np.zeros(win_len+1)
    trans_z = np.zeros(win_len+1)
    rot_x = np.zeros(win_len+1)
    rot_y = np.zeros(win_len+1)
    rot_z = np.zeros(win_len+1)
    for i in range(win_len+1):
        temp = df_results_time['vicra parameters'][i]
        trans_x[i] = temp[0]
        trans_y[i] = temp[1]
        trans_z[i] = temp[2]
        rot_x[i] = temp[3]
        rot_y[i] = temp[4]
        rot_z[i] = temp[5]

    # Translation 

    fig1, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(15,4))
    #plot trans_x
    ax1.plot(x1,trans_x_n,label=network_name,linewidth=1,color='r') 
    ax1.plot(x1,trans_x,label='Vicra data',linewidth=1,color='b') 
    ax1.set_title('Translation in x')
    ax1.set_ylim(trans_range[0], trans_range[1])
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Translation (mm)')
    ax1.legend()

    #plot trans_y
    ax2.plot(x1,trans_y_n,label=network_name,linewidth=1,color='r') 
    ax2.plot(x1,trans_y,label='Vicra data',linewidth=1,color='b') 
    ax2.set_title('Translation in y')
    ax2.set_ylim(trans_range[0], trans_range[1])
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Translation (mm)')
    ax2.legend()

    #plot trans_z
    ax3.plot(x1,trans_z_n,label=network_name,linewidth=1,color='r') 
    ax3.plot(x1,trans_z,label='Vicra data',linewidth=1,color='b')
    ax3.set_title('Translation in z')
    ax3.set_ylim(trans_range[0], trans_range[1])
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Translation (mm)')
    ax3.legend()
    plt.show()

    # Rotation 

    fig1, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(15,4))
    #plot trans_x
    ax1.plot(x1,rot_x_n,label=network_name,linewidth=1,color='r') 
    ax1.plot(x1,rot_x,label='Vicra data',linewidth=1,color='b') 
    ax1.set_title('Rotation in x')
    ax1.set_ylim(rot_range[0], rot_range[1])
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Rotation (°)')
    ax1.legend()

    #plot trans_y
    ax2.plot(x1,rot_y_n,label=network_name,linewidth=1,color='r') 
    ax2.plot(x1,rot_y,label='Vicra data',linewidth=1,color='b') 
    ax2.set_title('Rotation in y')
    ax2.set_ylim(rot_range[0], rot_range[1])
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Rotation (°)')
    ax2.legend()

    #plot trans_z
    ax3.plot(x1,rot_z_n,label=network_name,linewidth=1,color='r') 
    ax3.plot(x1,rot_z,label='Vicra data',linewidth=1,color='b')
    ax3.set_title('Rotation in z')
    ax3.set_ylim(rot_range[0], rot_range[1])
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Rotation (°)')
    ax3.legend()
    plt.show()
    
    return

def plot_diff_vicra_network(tracer, patient, df_results, time_window, network_name):
    
    """Plot the difference between the motion given by vicra and the network prediction. One graph for the translation and 
    one for the rotation. 

    Args:
    tracer: a string to specify the tracer since some patients were injected with different tracers.
    patient: the patient id (string).
    df_results: the df built with the build_df_results function.
    time_window: the time interval during which the motion will be ploted. 
    network_name: the name of the network you used to predict the motion (just for the graph legends).
    """
    
    # Select the part of df_results corresponding to the inputs
    df_results_tracer=df_results.loc[df_results['tracer'].str.contains(tracer)]
    df_results_patient=df_results_tracer.loc[df_results_tracer['UniqueID'].str.contains(patient)]
    time_list=np.linspace(time_window[0], time_window[1], time_window[1]-time_window[0]+1)
    time_list=[int(x) for x in time_list]
    df_results_time = df_results_patient[df_results_patient['time'].isin(time_list)].reset_index()
    
    win_len=time_window[1]-time_window[0]
    
    x1=range(win_len+1)
    
   #build fake vicra dataset -- fixed reference 
    trans_x_n = np.zeros(win_len+1)
    trans_y_n = np.zeros(win_len+1)
    trans_z_n = np.zeros(win_len+1)
    rot_x_n = np.zeros(win_len+1)
    rot_y_n = np.zeros(win_len+1)
    rot_z_n = np.zeros(win_len+1)
    
    for i in range(win_len+1):
        temp = df_results_time['model parameters'][i]
        trans_x_n[i] = temp[0]
        trans_y_n[i] = temp[1]
        trans_z_n[i] = temp[2]
        rot_x_n[i] = temp[3]
        rot_y_n[i] = temp[4]
        rot_z_n[i] = temp[5]

    # about vicra curve, use T of dp, add T in each row
    trans_x = np.zeros(win_len+1)
    trans_y = np.zeros(win_len+1)
    trans_z = np.zeros(win_len+1)
    rot_x = np.zeros(win_len+1)
    rot_y = np.zeros(win_len+1)
    rot_z = np.zeros(win_len+1)
    for i in range(win_len+1):
        temp = df_results_time['vicra parameters'][i]
        trans_x[i] = temp[0]
        trans_y[i] = temp[1]
        trans_z[i] = temp[2]
        rot_x[i] = temp[3]
        rot_y[i] = temp[4]
        rot_z[i] = temp[5]

    #plot difference between network and vicra
    trans_x_d = trans_x - trans_x_n
    trans_y_d = trans_y - trans_y_n
    trans_z_d = trans_z - trans_z_n
    plt.plot(x1,trans_x_d,label='x',linewidth=1,color='r') 
    plt.plot(x1,trans_y_d,label='y',linewidth=1,color='b')
    plt.plot(x1,trans_z_d,label='z',linewidth=1,color='y')
    plt.title('Difference between Vicra and '+str(network_name)+' - Translation')
    plt.xlabel('Time (s)') 
    plt.ylabel('Difference in translational motion') 
    plt.legend()
    plt.show()

    rot_x_d = rot_x - rot_x_n
    rot_y_d = rot_y - rot_y_n
    rot_z_d = rot_z - rot_z_n
    plt.plot(x1,rot_x_d,label='x',linewidth=1,color='r') 
    plt.plot(x1,rot_y_d,label='y',linewidth=1,color='b')
    plt.plot(x1,rot_z_d,label='z',linewidth=1,color='y')
    plt.title('Difference between Vicra and '+str(network_name)+' - Rotation')
    plt.xlabel('Time (s)') 
    plt.ylabel('Difference in rotational motion')
    plt.legend()
    plt.show()
    
    return 


def save_synthetic_vicra(tracer, patient, df_results, time_window, network_name):
    
    """Save the symthetic vicra info given by the network predictions as a csv file. This is especially useful to use the 
    plot_networks_comparison function, since it needs these csv files as input. 

    Args:
    tracer: a string to specify the tracer since some patients were injected with different tracers.
    patient: the patient id (string).
    df_results: the df built with the build_df_results function.
    time_window: the time interval during which the predictions will be saved. 
    network_name: the name of the network you used to predict the motion (just for the path).
    """
    
    # Select the part of df_results corresponding to the inputs
    df_results_tracer=df_results.loc[df_results['tracer'].str.contains(tracer)]
    df_results_patient=df_results_tracer.loc[df_results_tracer['UniqueID'].str.contains(patient)]
    time_list=np.linspace(time_window[0], time_window[1], time_window[1]-time_window[0]+1)
    time_list=[int(x) for x in time_list]
    df_results_time = df_results_patient[df_results_patient['time'].isin(time_list)].reset_index()
    
    win_len=time_window[1]-time_window[0]
    
    x1=range(win_len+1)
        
    #build fake vicra dataset -- fixed reference 
    trans_x_n = np.zeros(win_len+1)
    trans_y_n = np.zeros(win_len+1)
    trans_z_n = np.zeros(win_len+1)
    rot_x_n = np.zeros(win_len+1)
    rot_y_n = np.zeros(win_len+1)
    rot_z_n = np.zeros(win_len+1)
    
    for i in range(win_len+1):
        temp = df_results_time['model parameters'][i]
        trans_x_n[i] = temp[0]
        trans_y_n[i] = temp[1]
        trans_z_n[i] = temp[2]
        rot_x_n[i] = temp[3]
        rot_y_n[i] = temp[4]
        rot_z_n[i] = temp[5]
        
    output_n = np.zeros((1800,6),dtype=np.float)
    for i in range(1800):
        output_n[i][0] = trans_x_n[i]
        output_n[i][1] = trans_y_n[i]
        output_n[i][2] = trans_z_n[i]
        output_n[i][3] = rot_x_n[i]
        output_n[i][4] = rot_y_n[i]
        output_n[i][5] = rot_z_n[i]
        
    pd.DataFrame(output_n).to_csv("synthetic_vicra_"+network_name+'_'+patient+".csv")
    np.savetxt('synthetic_vicra_'+network_name+'_'+patient+'.txt',output_n,fmt='%0.6f')
    return 

def save_synthetic_vicra_zty(tracer, patient, df_results, time_window, network_name):
    
    """Save the symthetic vicra info given by the network predictions as a csv file. This is especially useful to use the 
    plot_networks_comparison function, since it needs these csv files as input. 

    Args:
    tracer: a string to specify the tracer since some patients were injected with different tracers.
    patient: the patient id (string).
    df_results: the df built with the build_df_results function.
    time_window: the time interval during which the predictions will be saved. 
    network_name: the name of the network you used to predict the motion (just for the path).
    """
    
    # Select the part of df_results corresponding to the inputs
    df_results_tracer=df_results.loc[df_results['tracer'].str.contains(tracer)]
    df_results_patient=df_results_tracer.loc[df_results_tracer['UniqueID'].str.contains(patient)]
    time_list=np.linspace(time_window[0], time_window[1], time_window[1]-time_window[0]+1)
    time_list=[int(x) for x in time_list]
    df_results_time = df_results_patient[df_results_patient['time'].isin(time_list)].reset_index()
    
    win_len=time_window[1]-time_window[0]
    
    x1=range(win_len+1)
        
    #build fake vicra dataset -- fixed reference 
    trans_x_n = np.zeros(win_len+1)
    trans_y_n = np.zeros(win_len+1)
    trans_z_n = np.zeros(win_len+1)
    rot_x_n = np.zeros(win_len+1)
    rot_y_n = np.zeros(win_len+1)
    rot_z_n = np.zeros(win_len+1)
    
    for i in range(win_len+1):
        temp = df_results_time['vicra parameters'][i]
        trans_x_n[i] = temp[0]
        trans_y_n[i] = temp[1]
        trans_z_n[i] = temp[2]
        rot_x_n[i] = temp[3]
        rot_y_n[i] = temp[4]
        rot_z_n[i] = temp[5]
        
    output_n = np.zeros((1800,6),dtype=np.float)
    for i in range(1800):
        output_n[i][0] = trans_x_n[i]
        output_n[i][1] = trans_y_n[i]
        output_n[i][2] = trans_z_n[i]
        output_n[i][3] = rot_x_n[i]
        output_n[i][4] = rot_y_n[i]
        output_n[i][5] = rot_z_n[i]
        
    pd.DataFrame(output_n).to_csv("vicra_"+network_name+'_'+patient+".csv")
    np.savetxt('vicra_'+network_name+'_'+patient+'.txt',output_n,fmt='%0.6f')
    return 


def print_loss(tracer, patient, df_results, time_window):
    
    """Print the Mean Square Error between the network and Vicra in the x y and z directions for translational and rotational motions. 

    Args:
    tracer: a string to specify the tracer since some patients were injected with different tracers.
    patient: the patient id (string).
    df_results: the df built with the build_df_results function.
    time_window: the time interval during which the mse will be computed. 
    """
    
    # Select the part of df_results corresponding to the inputs
    df_results_tracer=df_results.loc[df_results['tracer'].str.contains(tracer)]
    df_results_patient=df_results_tracer.loc[df_results_tracer['UniqueID'].str.contains(patient)]
    time_list=np.linspace(time_window[0], time_window[1], time_window[1]-time_window[0]+1)
    time_list=[int(x) for x in time_list]
    df_results_time = df_results_patient[df_results_patient['time'].isin(time_list)].reset_index()
    
    win_len=time_window[1]-time_window[0]
    
    x1=range(win_len+1)
    
    #build fake vicra dataset -- fixed reference 
    trans_x_n = np.zeros(win_len+1)
    trans_y_n = np.zeros(win_len+1)
    trans_z_n = np.zeros(win_len+1)
    rot_x_n = np.zeros(win_len+1)
    rot_y_n = np.zeros(win_len+1)
    rot_z_n = np.zeros(win_len+1)
    
    for i in range(win_len+1):
        temp = df_results_time['model parameters'][i]
        trans_x_n[i] = temp[0]
        trans_y_n[i] = temp[1]
        trans_z_n[i] = temp[2]
        rot_x_n[i] = temp[3]
        rot_y_n[i] = temp[4]
        rot_z_n[i] = temp[5]

    # about vicra curve, use T of dp, add T in each row
    trans_x = np.zeros(win_len+1)
    trans_y = np.zeros(win_len+1)
    trans_z = np.zeros(win_len+1)
    rot_x = np.zeros(win_len+1)
    rot_y = np.zeros(win_len+1)
    rot_z = np.zeros(win_len+1)
    for i in range(win_len+1):
        temp = df_results_time['vicra parameters'][i]
        trans_x[i] = temp[0]
        trans_y[i] = temp[1]
        trans_z[i] = temp[2]
        rot_x[i] = temp[3]
        rot_y[i] = temp[4]
        rot_z[i] = temp[5]
        
    mse_trans_x=np.square(trans_x-trans_x_n)
    mse_trans_y=np.square(trans_y-trans_y_n)
    mse_trans_z=np.square(trans_z-trans_z_n)
    
    mse_rot_x=np.square(rot_x-rot_x_n)
    mse_rot_y=np.square(rot_y-rot_y_n)
    mse_rot_z=np.square(rot_z-rot_z_n)
    
    print('Mean square errors between vicra and this network: \n')
    print('Translation:')
    print('x: ', np.round(np.mean(mse_trans_x), decimals=2))
    print('y: ', np.round(np.mean(mse_trans_y), decimals=2))
    print('z: ', np.round(np.mean(mse_trans_z), decimals=2))
    
#     print('Mean square errors between vicra and this network: \n')
    print('Rotation:')
    print('x: ', np.round(np.mean(mse_rot_x), decimals=2))
    print('y: ', np.round(np.mean(mse_rot_y), decimals=2))
    print('z: ', np.round(np.mean(mse_rot_z), decimals=2))
        
    return 
    
    
# Difference with other network

def plot_networks_comparison(tracer, patient, df_results, time_window, path_other_network, this_network, the_other_network):
    
    """Plot the translational and rotational motions in the x, y and z directions for Vicra and different networks.  

    Args:
    tracer: a string to specify the tracer since some patients were injected with different tracers.
    patient: the patient id (string).
    df_results: the df built with the build_df_results function.
    time_window: the time interval during which the motion will be ploted. 
    this_network: the name of the network associated to df_results.
    the_other_network: the name of the network that we want to compare to this network. 
    
    N.B: To be improved to compare more than 2 networks and to be based on csv fils only.
    """

    with open(path_other_network, newline = '') as file:
        reader = csv.reader(file,
                            quoting = csv.QUOTE_ALL,
                            delimiter = ',')

        # storing all the rows in an output list
        output = []
        for row in reader:
            output.append(row[:])
            
    df_results_tracer=df_results.loc[df_results['tracer'].str.contains(tracer)]
    df_results_patient=df_results_tracer.loc[df_results_tracer['UniqueID'].str.contains(patient)]
    time_list=np.linspace(time_window[0], time_window[1], time_window[1]-time_window[0]+1)
    time_list=[int(x) for x in time_list]
    df_results_time = df_results_patient[df_results_patient['time'].isin(time_list)].reset_index()
    
    win_len=time_window[1]-time_window[0]
    
    x1=range(win_len+1)
    
    #build fake vicra dataset -- fixed reference 
    trans_x_n = np.zeros(win_len+1)
    trans_y_n = np.zeros(win_len+1)
    trans_z_n = np.zeros(win_len+1)
    rot_x_n = np.zeros(win_len+1)
    rot_y_n = np.zeros(win_len+1)
    rot_z_n = np.zeros(win_len+1)
    
    for i in range(win_len+1):
        temp = df_results_time['model parameters'][i]
        trans_x_n[i] = temp[0]
        trans_y_n[i] = temp[1]
        trans_z_n[i] = temp[2]
        rot_x_n[i] = temp[3]
        rot_y_n[i] = temp[4]
        rot_z_n[i] = temp[5]

    # about vicra curve, use T of dp, add T in each row
    trans_x = np.zeros(win_len+1)
    trans_y = np.zeros(win_len+1)
    trans_z = np.zeros(win_len+1)
    rot_x = np.zeros(win_len+1)
    rot_y = np.zeros(win_len+1)
    rot_z = np.zeros(win_len+1)
    for i in range(win_len+1):
        temp = df_results_time['vicra parameters'][i]
        trans_x[i] = temp[0]
        trans_y[i] = temp[1]
        trans_z[i] = temp[2]
        rot_x[i] = temp[3]
        rot_y[i] = temp[4]
        rot_z[i] = temp[5]
            
    output = [item for sublist in output for item in sublist]
    
    content=output[7:]
    
    tr_x_dn=[]
    tr_y_dn=[]
    tr_z_dn=[]

    rot_x_dn=[]
    rot_y_dn=[]
    rot_z_dn=[]

    for i in range(1800):
        tr_x_dn.append(content[i*7+1])
        tr_y_dn.append(content[i*7+2])
        tr_z_dn.append(content[i*7+3])

        rot_x_dn.append(content[i*7+4])
        rot_y_dn.append(content[i*7+5])
        rot_z_dn.append(content[i*7+6])

    tr_x_dn=np.array(tr_x_dn).astype('float64')
    tr_y_dn=np.array(tr_y_dn).astype('float64')
    tr_z_dn=np.array(tr_z_dn).astype('float64')

    rot_x_dn=np.array(rot_x_dn).astype('float64')
    rot_y_dn=np.array(rot_y_dn).astype('float64')
    rot_z_dn=np.array(rot_z_dn).astype('float64')
    
    tr_x_dn_cut=tr_x_dn[time_window[0]-3600:time_window[1]-3599]
    tr_y_dn_cut=tr_y_dn[time_window[0]-3600:time_window[1]-3599]
    tr_z_dn_cut=tr_z_dn[time_window[0]-3600:time_window[1]-3599]
    
    
    rot_x_dn_cut=rot_x_dn[time_window[0]-3600:time_window[1]-3599]
    rot_y_dn_cut=rot_y_dn[time_window[0]-3600:time_window[1]-3599]
    rot_z_dn_cut=rot_z_dn[time_window[0]-3600:time_window[1]-3599]
    
    
    # Translation 

    fig2, (ax4, ax5, ax6) = plt.subplots(1,3, figsize=(15,4))
    #plot trans_x
    ax4.plot(x1,trans_x_n,label=this_network,linewidth=1,color='r') 
    ax4.plot(x1,trans_x,label='Vicra',linewidth=1,color='b') 
    ax4.plot(x1,tr_x_dn_cut,label=the_other_network,linewidth=1,color='g') 
    ax4.set_title('Translation in x')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Translation (mm)')
    ax4.legend()

    #plot trans_y
    ax5.plot(x1,trans_y_n,label=this_network,linewidth=1,color='r') 
    ax5.plot(x1,trans_y,label='Vicra',linewidth=1,color='b') 
    ax5.plot(x1,tr_y_dn_cut,label=the_other_network,linewidth=1,color='g') 
    ax5.set_title('Translation in y')
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Translation (mm)')
    ax5.legend()

    #plot trans_z
    ax6.plot(x1,trans_z_n,label=this_network,linewidth=1,color='r') 
    ax6.plot(x1,trans_z,label='Vicra',linewidth=1,color='b')
    ax6.plot(x1,tr_z_dn_cut,label=the_other_network,linewidth=1,color='g') 
    ax6.set_title('Translation in z')
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Translation (mm)')
    ax6.legend()
    plt.show()

    # Rotation 

    fig2, (ax4, ax5, ax6) = plt.subplots(1,3, figsize=(15,4))
    #plot trans_x
    ax4.plot(x1,rot_x_n,label=this_network,linewidth=1,color='r') 
    ax4.plot(x1,rot_x,label='Vicra',linewidth=1,color='b') 
    ax4.plot(x1,rot_x_dn_cut,label=the_other_network,linewidth=1,color='g') 
    ax4.set_title('Rotation in x')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Rotation (°)')
    ax4.legend()

    #plot trans_y
    ax5.plot(x1,rot_y_n,label=this_network,linewidth=1,color='r') 
    ax5.plot(x1,rot_y,label='Vicra',linewidth=1,color='b') 
    ax5.plot(x1,rot_y_dn_cut,label=the_other_network,linewidth=1,color='g') 
    ax5.set_title('Rotation in y')
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Rotation (°)')
    ax5.legend()
    
    #plot trans_z
    ax6.plot(x1,rot_z_n,label=this_network,linewidth=1,color='r') 
    ax6.plot(x1,rot_z,label='Vicra',linewidth=1,color='b')
    ax6.plot(x1,rot_z_dn_cut,label=the_other_network,linewidth=1,color='g') 
    ax6.set_title('Rotation in z')
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Rotation (°)')
    ax6.legend()
    plt.show()
    
    return
    

def make_boxplot(*args, **kwargs):
    
    """Make boxplots of the RMSE between the vicra and model predictions for different models to compare them. This RMSE (instead of just 
    the MSE, for visualisation purpose) combines the errors in the translational and rotational directions for the three x, y and z 
    axes. Can be used for testing dataset results, inference,... 

    Args:
    *args: the df containing the results. Ideally, they should be outputs of the build_df_results function, but as long as 
    they have 'vicra parameters' and 'model parameters' columns containing the parameters at each moment, the function should work.
    **kwargs: a list of the different model names.
  
    """
    
    df_results_all=[]
    i=0
    
    for df in args: 
        
        df['model_name']=kwargs['models'][i] #add a column with the model name 
        
        # Get the vicra parameters in individual columns
        df['trans_x_vicra']=[df['vicra parameters'][i][0] for i in range(len(df))]
        df['trans_y_vicra']=[df['vicra parameters'][i][1] for i in range(len(df))]
        df['trans_z_vicra']=[df['vicra parameters'][i][2] for i in range(len(df))]
        df['rot_x_vicra']=[df['vicra parameters'][i][3] for i in range(len(df))]
        df['rot_y_vicra']=[df['vicra parameters'][i][4] for i in range(len(df))]
        df['rot_z_vicra']=[df['vicra parameters'][i][5] for i in range(len(df))]
        
        # Get the model parameters in individual columns 
        df['trans_x_model']=[df['model parameters'][i][0] for i in range(len(df))]
        df['trans_y_model']=[df['model parameters'][i][1] for i in range(len(df))]
        df['trans_z_model']=[df['model parameters'][i][2] for i in range(len(df))]
        df['rot_x_model']=[df['model parameters'][i][3] for i in range(len(df))]
        df['rot_y_model']=[df['model parameters'][i][4] for i in range(len(df))]
        df['rot_z_model']=[df['model parameters'][i][5] for i in range(len(df))]
        
        # Compute the global translational and rotational errors 
        df['Error_x_trans'] = np.square(df['trans_x_model'] - df['trans_x_vicra'])
        df['Error_y_trans'] = np.square(df['trans_y_model'] - df['trans_y_vicra'])
        df['Error_z_trans'] = np.square(df['trans_z_model'] - df['trans_z_vicra'])
        df['Error_trans'] = (df['Error_x_trans'] + df['Error_y_trans'] + df['Error_z_trans'])/3.0

        df['Error_x_rot'] = np.square(df['rot_x_model'] - df['rot_x_vicra'])
        df['Error_y_rot'] = np.square(df['rot_y_model'] - df['rot_y_vicra'])
        df['Error_z_rot'] = np.square(df['rot_z_model'] - df['rot_z_vicra'])
        df['Error_rot'] = (df['Error_x_rot'] + df['Error_y_rot'] + df['Error_z_rot'])/3.0

        # Compute the global RMSE
        df['Error_glob'] = np.sqrt(df['Error_trans']+df['Error_rot']/2) 

#         perc_90=np.percentile(df['Error_glob'], 90) #exclude the top 10% based on the assumption that major motion can be detected 
        #and removed by tools such as MCCOD. 
#         df=df[df['Error_glob'] < perc_90]

        df_results_all.append(df)
        i+=1
        
    df_all=pd.concat(df_results_all)
    
    #Plot the boxplots
    g = sns.catplot(
    data=df_all,
    x='model_name',
    y='Error_glob',
#     hue='Tracer',
    kind='box'
)

    return 
