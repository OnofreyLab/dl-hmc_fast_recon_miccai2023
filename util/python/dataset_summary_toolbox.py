# A toolbox to perform an analysis and get an overview of the DL-HMC data.

# Available functions so far: 
# -get_data: links the spreadsheet and the notebook and get the data for the wanted tracers 
# -print_info: print the info gathered by get_data
# -compute_delta_T: compute delta_T for each patient and add it to the summary
# -plot_delta_T_all: plot the delta_T computed by compute_delta_T
# -delta_T_norm: normalizes delta_T and plots it
# -delta_T_smoothed: smoothes delta_T_norm and plots it
# -data_analysis_display: displays information about the motion (magnitude, when do the major movements occur, mean and SD of delta_T...)
# -average_motion: computes and plot the average delta_T at each second among patients
# -make_gif_3Dcloud: make a gif with the 3D cloud data for a list of patients

import os
import oauth2client
import gspread
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import monai
from monai.transforms import \
    Compose, LoadImaged, AddChanneld, Orientationd, \
    Spacingd, \
    ToTensord,  \
    DataStatsd, \
    ToDeviced
from monai.data import list_data_collate    
from PIL import Image, ImageDraw
import glob
import sys
sys.path.append(r'../util/python')
import vicra_toolbox
import nibabel as nib

# A function to link the spreadsheet to the notebook and get the data that is ready for the wanted tracers. 

# Inputs: 
#     key: a string corresponding the .json key (e.g. 'dl-hmc-data-analysis-2c4ea7d9c181.json')
#     tracers: list of tracers of interest (must be the same spelling than in the spreadsheet)

# Outputs:
#     df: concatenated dfs of the different tracers 
#     df_ready: subset of df containing only the patients for which the data generation was done
#     patients_ready: list of lists of preprocessed patients for each tracer

def get_data(key, tracers):

    ### Link the script to the Google sheet and create a df with the data for all tracers ###

    from oauth2client.service_account import ServiceAccountCredentials

    key='dl-hmc-data-analysis-2c4ea7d9c181.json'

    # Define the scope
    scope = ['https://spreadsheets.google.com/feeds','https://www.googleapis.com/auth/drive']

    # Add credentials to the account 
    #/!\ the .json key must be in your repository
    creds = ServiceAccountCredentials.from_json_keyfile_name(key, scope)

    # Authorize the clientsheet 
    client = gspread.authorize(creds)

    # Get the instance of the spreadsheet
    sheet=client.open('DL_HMC-list')

    # Get the different sheets corresponding to the different tracers
    tracers_sheet=['FDG_old', 'FDG_simu', 'APP311_old', 'FDG', 'APP311'] #must be updated when tracers are added 

    sheets=[]
    records_data=[]
    records_data_values=[]
    records_data_copy=[]

    for tracer in tracers:
        records_data_copy_tracer=[]
        tracer_idx=tracers_sheet.index(tracer)
        sheets.append(sheet.get_worksheet(tracer_idx))
    #     records_data.append(sheets[tracer_idx].get_all_records()) doing some corrections here
        records_data_values.append(sheet.get_worksheet(tracer_idx).get_all_values())
    dict_keys=records_data_values[0][0]

    for tracer in tracers:
        tracer_idx=tracers_sheet.index(tracer)
        records_data_values_tracer=sheet.get_worksheet(tracer_idx).get_all_values()
        for i in range(len(records_data_values_tracer)-1):
            patient_dict=dict(zip(dict_keys, records_data_values_tracer[i+1]))
            records_data_copy_tracer.append(patient_dict)
        records_data_copy.append(records_data_copy_tracer)


    records_data=records_data_copy
    df=[]
    patients_ready=[]
    df = pd.DataFrame.from_dict(records_data[0])

    # for i in range(len(tracers)-1):
    #     df = pd.concat([df, pd.DataFrame.from_dict(records_data[i+1])]) #contains all patients for all tracers

    ### Get the data that is ready ###

    df_ready=df.loc[df['Status (Done/Running)'].str.contains('Done')]

    #     print("Overall, ", len(df_ready), " patients out of", len(df), " are ready.")

    for i in range(len(tracers)):
        tracer=tracers[i]
        df_int=df.loc[df['tracer'].str.contains(tracer)]
        df_ready_int=df_int.loc[df_int['Status (Done/Running)'].str.contains('Done')]

        patients_ready_int=(df_ready_int['PatientID'].tolist())

    #         print("For tracer ", tracers[i], " ", len(patients_ready_int), " patients out of ", len(df_int), " are ready.")
    #         print("Ready patients are", patients_ready_int)

        patients_ready.append(patients_ready_int)
        
    return df, df_ready, patients_ready

# A function to tell you how many patients are ready for each tracer, and their IDs.

def print_info(tracers):
    df, df_ready, patients_ready=get_data('dl-hmc-data-analysis-2c4ea7d9c181.json', tracers)
    
    for i in range(len(tracers)):
        tracer=tracers[i]
        df_int=df.loc[df['tracer'].str.contains(tracer)]
        df_ready_int=df_int.loc[df_int['Status (Done/Running)'].str.contains('Done')]

        patients_ready_int=(df_ready_int['PatientID'].tolist())

        print("For tracer ", tracers[i], " ", len(patients_ready_int), " patients out of ", len(df_int), " are ready.")
        print("Ready patients are", patients_ready_int)
        
    return 


# A function to compute delta_T for some patients and add T, delta_T, relative_T and delta_reltaive_T to their summaries.

# Inputs: 
#     tracer: your choice of tracer (string in a list)
#     patients: the list of patients for which you want to compute T, delta_T, relative_T and delta_relative_T

# Outputs:
#     summaries: list of summaries containing the path to MOLAR reconstruction and so on for each patient 
#     delta_T_all: list of lists containing the delta_T values for each patient of patients

def compute_delta_T(tracer, patients, data_type='real', force_scan=None):
    
    # Adapt tracer name for simulation data 
    tracer1 = tracer[0]
    if data_type=='simulation':
        tracer[0] = tracer[0] + '_simu' 
        
    df, df_ready, patients_ready=get_data('dl-hmc-data-analysis-2c4ea7d9c181.json', tracer)
    df_ready= df_ready.loc[df_ready['PatientID'].isin(patients)].reset_index(drop=True)
    
    #Check for and remove duplicates
    if force_scan: #If we want to force data for a patient to be from a scan, otherwise the first occurence is kept
        df_ready_clean=df_ready.drop_duplicates(subset=['PatientID'], keep=False)
        for el in force_scan:
            df_ready_clean=pd.concat((df_ready_clean,(df_ready.loc[df_ready['InjectionID']==el[1]]))).reset_index(drop=True)
    else:
        df_ready_clean=df_ready.drop_duplicates(subset=['PatientID'])  
    
    # Transform the dates (e.g. 2015/7/3 => 20150703)
    dates=df_ready_clean[['date', 'tracer']]
    dates_list=dates['date'].tolist()
    dates_nb=[]
#     img_power=df_ready['image_power'].tolist()

    for date in dates_list:
        date_split=date.split('/')
        year=date_split[0]
        if len(date_split[1])<2:
            month="0"+date_split[1]
        else:
            month=date_split[1]
        if len(date_split[2])<2:
            day="0"+date_split[2]
        else:
            day=date_split[2]
        date_corr=year+month+day
        dates_nb.append(date_corr)

    dates_nb=np.array(dates_nb)    
    dates.insert(2, 'dates_nb', dates_nb)
    # Find path to access the data in /data16/

    paths=[]
    
    patients_ready=(df_ready_clean[['PatientID', 'tracer']])
    patients=list(patients_ready['PatientID'])
    analysis_id=df_ready_clean[['InjectionID', 'tracer']]  
    analysis_id_only=list(analysis_id['InjectionID'])
    
    if data_type=='real':
        for k in range(len(df_ready_clean)):
            paths.append(r'/data16/public/registration-brain_PET/data/mCT_real/'+str(tracer[0])+'/'+str(dates_nb[k])+'_'+str(patients[k])+'/'+str(analysis_id_only[k]))
    if data_type=='simulation':
        for k in range(len(df_ready_clean)):
            paths.append(r'/data16/public/registration-brain_PET/data/mCT_simulation/'+str(tracer1)+'/'+str(dates_nb[k])+'_'+str(patients[k])+'/'+str(analysis_id_only[k]))
    
    # Get summaries paths 

    summaries_paths=[]
    if data_type=='simulation':
        for l in range(len(patients)):
            summaries_paths.append(os.path.join(paths[l],'Summary_'+str(dates_nb[k])+'_'+patients[l]+'_3600_5400.csv'))
    if data_type=='real':
        for l in range(len(patients)):
            summaries_paths.append(os.path.join(paths[l],'Summary_'+patients[l]+'_3600_5400.csv'))

        
    # Functions to compute T, delta_T, ...
    
    def delta_T_magnitude(T_ref, T):
        mag = np.sum(np.square(T-T_ref))
        return mag

    def Relative_motion_A_to_B_12(img_vc1, img_vc2):
        A = np.pad(img_vc1, pad_width=1)
        B = np.pad(img_vc2, pad_width=1)
        R = vicra_toolbox.Relative_motion_A_to_B(A, B)
        return R[1:13]

    summaries=[]
    delta_T_all=[]
    recons_tracer=[]
    
    j=0
    
    # Compute and add T, delta_T, relative_T and relative_delta_T to the summaries for each patient

    for path in summaries_paths:
        
        delta_T_all_temp=[]
        relative_T_all_temp=[]
        delta_relative_T_all_temp=[]
        summary = pd.read_csv(path)
        matrix_cols = ['VC_11', 'VC_12', 'VC_13', 'VC_14', 'VC_21', 'VC_22','VC_23', 'VC_24', 'VC_31', 'VC_32', 'VC_33', 'VC_34']
        M = summary[matrix_cols].values.tolist()
        summary['MATRIX'] = M
        
        #12 parameters (M) => 6 parameters (T)
        summary['T'] = summary['MATRIX'].apply(lambda x: vicra_toolbox.RotTransMatrix_6Params(x, 1))
        summary['delta_T'] = summary['T'].apply(lambda t: delta_T_magnitude(summary.loc[0,'T'], t))
        delta_T_all_temp.append(summary['delta_T'])
        summary['relative_T'] = summary['MATRIX'].apply(lambda t: vicra_toolbox.RotTransMatrix_6Params(Relative_motion_A_to_B_12(summary.loc[0,'MATRIX'], t), 1)) #would be wrt the first s ?
        summary['delta_relative_T'] = summary['relative_T'].apply(lambda t: delta_T_magnitude(summary.loc[0,'relative_T'], t))
#         summary['image_power']=img_power[j]
        
        delta_T_all.append(delta_T_all_temp)
        
        summaries.append(summary)
        j+=1
    
    return summaries, delta_T_all


# A function to plot delta_T

# Inputs: 
#     tracer: your choice of tracer (string in a list)
#     patients: the list of patients for which you want to plot delta_T
#     delta_T_all: delta_T for the patients (computed thansks to compute_delta_T)

def plot_delta_T_all(tracer, patients, delta_T_all):

    x=np.linspace(3600,5400,num=1800)

    fig, ax=plt.subplots(1,1, figsize=(15,8))
    
    for patient in patients:
        patient_delta_T=np.array(delta_T_all[patients.index(patient)])
        patient_delta_T=np.transpose(patient_delta_T)
        ax=plt.plot(x,patient_delta_T)
        plt.legend(patients)
    plt.title("delta_T magnitude - " + str(tracer[0]), fontsize=12)
    plt.xlabel("Time (s)", fontsize=12)
    plt.ylabel("delta_T magnitude", fontsize=12)
    plt.savefig('delta_T_'+str(tracer[0])+'.png')
        
    return   


# A function to normalize delta_T for some patients and plot it. 

# Inputs: 
#     tracer: your choice of tracer (string)
#     patients: the list of patients for which you want to plot normalized delta_T. 

# Output:
#     delta_T_norm: normalized delta_T 

def delta_T_norm(tracer, patients):

    summaries, delta_T_all=compute_delta_T(tracer, patients)
    delta_T_norm=[]
    x=np.linspace(3600,5400,num=1800)

    def normalization(item, lis):
        item_norm=item/np.max(lis) #min=0
        return item_norm

    fig2, ax2=plt.subplots(1,1, figsize=(15,8))

    for patient in patients:
        delta_T_pat=delta_T_all[patients.index(patient)]
        delta_T_pat_norm=[normalization(item, delta_T_pat) for item in delta_T_pat]
        delta_T_pat_norm=np.array(delta_T_pat_norm).T
        delta_T_norm.append(delta_T_pat_norm)
        ax2=plt.plot(x,delta_T_pat_norm)
        plt.legend(patients)
    plt.title("delta_T magnitude - " + str(tracer[0]), fontsize=12)
    plt.xlabel("Time (s)", fontsize=12)
    plt.ylabel("delta_T magnitude", fontsize=12)
    plt.savefig('delta_T_'+str(tracer[0])+'.png')

    return delta_T_norm


# A function to smooth delta_T for some patients and plot it. 

# Inputs: 
#     tracer: your choice of tracer (string in a list)
#     patients: the list of patients for which you want to compute and plot smoothed delta_T 
#     window: the smoothing window (len of smoothing window in s) 

# Output:
#     delta_T_smoothed: smoothed delta_T 

def delta_T_smoothed(tracer, patients, window): 
    
    from scipy.ndimage.filters import uniform_filter1d
    
    def normalization(item, lis):
        item_norm=item/np.max(lis) #min=0
        return item_norm
    
#     patients_tracer=patients_ready_all.loc[patients_ready_all['tracer'].str.contains(tracer)]
#     patients=list(patients_tracer['PatientID'])

    x=np.linspace(3600,5400,num=1800)

    fig3, ax3=plt.subplots(1,1, figsize=(15,8))
#     df_int=df.loc[df['tracer'].str.contains(tracer)]

    summaries, delta_T_all=compute_delta_T(tracer, patients)

    delta_T_smoothed=[]
    
    for patient in patients:

        delta_T_pat=delta_T_all[patients.index(patient)]

#         delta_T_pat_norm=[normalization(item, delta_T_pat) for item in delta_T_pat]
        delta_T_movav=uniform_filter1d(delta_T_pat, size=window)
        delta_T_movav=np.array(delta_T_movav).T
        delta_T_smoothed.append(delta_T_movav)

        ax3=plt.plot(x,delta_T_movav)
        plt.legend(patients)
    plt.title("delta_T magnitude smoothed - " + str(tracer[0]), fontsize=12)
    plt.xlabel("Time (s)", fontsize=12)
    plt.ylabel("delta_T magnitude", fontsize=12)
    plt.savefig('delta_T_'+str(tracer[0])+'.png')

    return delta_T_smoothed


# A function to display information about the motion (magnitude, when do the major movements occur, mean and SD of delta_T...)
# Mean and SD of delta_T could be used as QC for the algorithm outputs.

# Input : 
#     tracers: your choice of tracers (list)
#     patients: the list of patients for each tracer (list of list) 

def data_analysis_display(tracers, patients):

    # Motion magnitude distribution 
    
    bins=20
    
    motmag_list_all=[]
    
    fig0, ax0 = plt.subplots(1,1,figsize=(7,4))
    l=0
    for tracer in tracers:

        df, df_ready, patients_ready=get_data('dl-hmc-data-analysis-2c4ea7d9c181.json', tracers[l])
        df_ready= df_ready.loc[df_ready['PatientID'].isin(patients[l])]
        motmag_list=df_ready['MotMag_entire_EM_mm'].tolist()
        plt.hist(motmag_list, bins, alpha=0.5, label=tracers[l])
        motmag_list_all.append(motmag_list)
        l+=1
    
    plt.legend()
    plt.title('Motion magnitude histogram')
    plt.xlabel('Motion (mm)')
    plt.ylabel('Number of patients')

    x=np.linspace(3600,5400,num=1800)

#     for tracer in tracers:
        
#         patients_tracer=patients_ready_all.loc[patients_ready_all['tracer'].str.contains(tracer)]
#         patients=list(patients_tracer['PatientID'])
        
#         delta_T_tracer=delta_tracers.loc[delta_tracers['tracer'].str.contains(tracer)]
#         delta_T_tracer=list(delta_T_tracer['delta_T'])


    # Histogram of big movements amplitudes

    global_max_all=[]
    
    p=0
    
    for tracer in tracers:
        summaries, delta_T_tracer=compute_delta_T(tracers[p], patients[p])
        global_max_tracer=[]
        for patient_delta_T in delta_T_tracer:
            patient_delta_T=patient_delta_T[0]
            patient_delta_T=list(patient_delta_T)
            global_max=[]
            for i in range(10):
                new_max=np.max(patient_delta_T)
                global_max.append(new_max)
                del patient_delta_T[patient_delta_T.index(new_max)-5:patient_delta_T.index(new_max)+5]
            global_max_tracer.append(global_max)
        global_max_all.append(global_max_tracer)
        p+=1

    p=0

    fig3, ax3 = plt.subplots(1,1,figsize=(7,4))
    
    for tracer in range(len(tracers)):

        flat_max = [item for sublist in global_max_all[p] for item in sublist]
        plt.hist(flat_max, bins, alpha=0.5, label=tracers[p])
        
        p+=1
        
    plt.title('delta_T maxima histogram')
    plt.legend()
    plt.xlabel('delta_T')
    plt.ylabel('Number of maximum')

    # Find the moments corresponding to global maxima for each patient

    time_max_all=[]

    p=0
    
    for tracer in tracers:
        
        summaries, delta_T_tracer=compute_delta_T(tracers[p], patients[p])
        
        #delta_T_tracer=delta_tracers.loc[delta_tracers['tracer'].str.contains(tracer)]
        time_max_tracer=[]
        for patient_delta_T in delta_T_tracer:
            patient_delta_T=patient_delta_T[0]
            x=list(np.linspace(3600,5400,num=1800))
            patient_delta_T=list(patient_delta_T)
            time_global_max=[]
            for i in range(10):
                new_max=np.max(patient_delta_T)
                new_time_max=x[patient_delta_T.index(new_max)]
                time_global_max.append(new_time_max)
                del patient_delta_T[patient_delta_T.index(new_max)-5:patient_delta_T.index(new_max)+5]
                del x[x.index(new_time_max)-5:x.index(new_time_max)+5]
            time_max_tracer.append(time_global_max)
        time_max_all.append(time_max_tracer)
        p+=1

    p=0
    
    fig3, ax3 = plt.subplots(1,1,figsize=(7,4))
    
    for tracer in range(len(tracers)):

        time_flat_max = [item for sublist in time_max_all[p] for item in sublist]
        plt.hist(time_flat_max, bins, alpha=0.5, label=tracers[p])
        
        p+=1
        
    plt.title('Time of motion maximum occurence histogram')
    plt.xlabel('Time (s)')
    plt.ylabel('Number of maximum')
    plt.legend()

    # Mean and SD of delta_T_all 

    mean_all=[]
    sd_all=[]
    
    p=0

    for tracer in tracers:
        summaries, delta_T_tracer=compute_delta_T(tracers[p], patients[p])
        #delta_T_tracer=delta_tracers.loc[delta_tracers['tracer'].str.contains(tracer)]
        mean_tracer=[]
        sd_tracer=[]
        for patient_delta_T in delta_T_tracer:
            patient_delta_T=np.array(patient_delta_T[0])
            mean_tracer.append(np.mean(patient_delta_T))
            sd_tracer.append(np.std(patient_delta_T))
        mean_all.append(mean_tracer)
        sd_all.append(sd_tracer)
        
    p=0

    for tracer in range(len(tracers)):
        
        fig5, ax5 = plt.subplots(1,1,figsize=(7,4))
        plt.hist(mean_all[p], bins=20, label=tracers[p], alpha=0.5)
        plt.title('Mean delta_T histogram')
        plt.xlabel('delta_T')
        plt.ylabel('Number of patients')
        plt.legend()
        
        fig6, ax6 = plt.subplots(1,1,figsize=(7,4))
        plt.hist(sd_all[p], bins=20, label=tracers[p], alpha=0.5)
        plt.title('delta_T SD histogram')
        plt.xlabel('delta_T SD')
        plt.ylabel('Number of patients')
        plt.legend()

        p+=1

    # MotMag during entire scan 

#     n=0
    
#     for tracer in motmag_list_all:
        
#         print("For ", tracers[n]," tracer, mean total motion during the scan is ", np.mean(tracer), "mm.")
#         print("For ", tracers[n]," tracer, total motion during the scan standard deviation is ", np.std(tracer), "mm.")
#         n+=1
        
    return 

# A function to average the motion (delta_T) among the patients at each moment and plot it. 

# Input : 
#     tracers: your choice of tracers (list)
#     patients: the list of patients for each tracer (list of list)

def average_motion(tracers, patients):
    
    p=0
    average_mot=[]
    
    for tracer in tracers:
    
        tracer_list=[]
        tracer_list.append(tracer)
        
        average_mot_tracer=[]
    
        summaries, delta_T_tracer=compute_delta_T(tracer_list, patients[p])
        delta_T_tracer=list(delta_T_tracer)  
        print(len(delta_T_tracer))

        s=0
        
        print(len(patients[p]))

        for i in range(1800):
            for j in range(len(patients[p])):
                s+=delta_T_tracer[j][0][i]
            
            average_mot_tracer.append(s/len(patients[p]))
            s=0
        average_mot.append(average_mot_tracer)
        p+=1
            
    x=np.linspace(3600,5400,num=1800)
    fig, ax=plt.subplots(1,1, figsize=(7,4))
    
    p=0
    
    for tracer in tracers:
        average_mot[p]=np.array(average_mot[p])
        average_mot[p]=average_mot[p].T
        ax=plt.plot(x,average_mot[p]/len(patients[p]))
        p+=1
    plt.title("Average delta_T at each moment", fontsize=12)
    plt.xlabel("Time (s)", fontsize=12)
    plt.ylabel("delta_T magnitude", fontsize=12)
    plt.savefig('average_delta_T_all.png')
    plt.legend(tracers)
    
    return average_mot


# A function to make gifs for your presentations. 

def make_gif_3Dcloud(tracer, patients): #Do one by one patient
    
    from sampling_toolbox import data_sample_random

    summaries, delta_T_tracer=compute_delta_T(tracer, patients)
    
    k=0

    for patient in patients:

        df_tmp=summaries[k] #df
        paths_to_clouds = []
        df_sample=data_sample_random(df_tmp,par=0.8,n=1,Start_t=3600,TimeSpan=1800)
        df_sample=df_sample.reset_index(drop=True)

        images_gif=[]
            
        KEYS = ['ThreeD_Cloud_x']
        
        tr=df_sample

        tr_dict = tr.to_dict('records')
        for i in range(len(tr_dict)):
            x = tr_dict[i]['ThreeD_Cloud_x'].find('nii')
            fn_cloud1 = tr_dict[i]['ThreeD_Cloud_x'][0:x] + 'nii_monai_resize'
            x = x+3
            y = tr_dict[i]['ThreeD_Cloud_x'].find('3dcld')
            fn_cloud2 =  tr_dict[i]['ThreeD_Cloud_x'][x:y] + '3dcld_monai_rz.nii'
            tr_dict[i]['ThreeD_Cloud_x'] = fn_cloud1 + fn_cloud2
            paths_to_clouds.append(tr_dict[i]['ThreeD_Cloud_x'])
            
            
        ref_image=nib.load(paths_to_clouds[0]).get_fdata()
        slice_num = ref_image.shape[2]//2
        my_dpi=96
        num_display = 90
        
        j=0

        for i in range(num_display):
            ref_image = nib.load(paths_to_clouds[i*20]).get_fdata()
            slice_num = ref_image.shape[2]//2
            plt.figure(figsize=(5,5))
            plt.imshow(ref_image[:,:,slice_num], cmap='gray')
            plt.savefig('gif_frame/'+str(patient)+'_frame'+str(j)+'.jpg', dpi=my_dpi*0.5)
            plt.close()

            j+=1

        path_stored=r'/notebooks/gif_frame/'

        images=[]

        for i in range(num_display):
            images.append(Image.open('gif_frame/'+str(patient)+'_frame'+str(i)+'.jpg'))
        img_gif=images[0]
        img_gif.save('3dcloud_gif'+str(patient)+'.gif', format="GIF", append_images=images,
                       save_all=True, duration=10, loop=0)
        k+=1

    return 


