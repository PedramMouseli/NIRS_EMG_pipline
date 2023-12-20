#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 19:13:14 2023

@author: moayedilab
"""

from data_processing import muscle_data
import pandas as pd
import numpy as np
import time

# initialize the data class
data_folder = '/Users/moayedilab/Library/CloudStorage/OneDrive-UniversityofToronto/Clench_data/'

sub_id = 'C049'

nirs_path = f'{data_folder}/NIRS_excel/{sub_id}.xlsx'
emg_path = f'{data_folder}/EMG_mat/{sub_id}.mat'
plot_path = f'{data_folder}/plots'
pickle_path = f'{data_folder}/preprocessed'

# generate the class
sub_data = muscle_data(sub_id, nirs_path, emg_path, plot_path, scaling_coef=1, stress_included=True)

# import the data
sub_data.import_nirs()
sub_data.import_emg()

# check the events
events = sub_data.check_events()

# modify events
# sub_data.remove_event(0)
# sub_data.add_event(434.62)

# events = sub_data.check_events()

# generate task events
sub_data.generate_task_events(offset=15)

# plot EMG
# task_array = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
task_array = [1,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
# task_array = [1,2,3,5,6,7,8,9,10,11,12,13,14,15,16]
mvc_events = [3,4]

sub_data.visualization(plot_type='emg',events=True, side="l", nirschannel=1, nirsmeasure='tsi',
                  task_array=task_array, mvc_events=mvc_events, save_plot=False)

# confirm events
sub_data.confirm_events()

# sync / normalize nirs
sub_data.sync_normalize_nirs()

# plot data
sub_data.visualization(plot_type='nirs',events=True, side="r", nirschannel=1, nirsmeasure='tsi',
                  task_array=task_array, mvc_events=mvc_events, save_plot=False)



##### Preprocess all subjects

nirs_excel = pd.read_excel("/Users/moayedilab/Library/CloudStorage/OneDrive-UniversityofToronto/NIRS_clenching/NIRS processing.xlsx")
data_folder = '/Users/moayedilab/Library/CloudStorage/OneDrive-UniversityofToronto/Clench_data/'
plot_path = f'{data_folder}/plots'
pickle_path = f'{data_folder}/preprocessed'

exclusion_list = ['C003', 'C031', 'C033']

for j, sub_id in enumerate(nirs_excel['sub_id'][91:]):
    
    if (sub_id in exclusion_list) or (nirs_excel.loc[nirs_excel['sub_id']==sub_id]['group'].item()=='TMD'):
        continue
    
    start_time = time.time()
    ### read parameters from the excel file
    sub_params = nirs_excel.loc[nirs_excel['sub_id']==sub_id]
    
    sub_num = sub_params['sub_num'].item()

    print(f'processing {sub_num}')
    print(f'subject id: {sub_id}\n')
    
    offset = sub_params['offset'].item()
    mvc_events = np.fromstring(sub_params['mvc_events'].item(), dtype=int, sep=',')
    clench_array = np.fromstring(sub_params['clench_order'].item(), dtype=int, sep=',')
    scaling_coef = sub_params['scaling_coef'].item()
    add_events = np.fromstring(sub_params['add_event'].astype(str).item(), dtype=float, sep=',')
    rm_events = np.fromstring(sub_params['rm_event'].astype(str).item(), dtype=float, sep=',')
    stress_included = sub_params['stress_included'].item()
    ignore_mvc_rest = sub_params['ignore_mvc_rest'].item()
    
    ### process data
    nirs_path = f'{data_folder}/NIRS_excel/{sub_id}.xlsx'
    emg_path = f'{data_folder}/EMG_mat/{sub_id}.mat'
    
    sub_data = muscle_data(sub_id, nirs_path, emg_path, plot_path, scaling_coef=scaling_coef, stress_included=stress_included)

    # import the data
    sub_data.import_nirs()
    sub_data.import_emg()
    
    # add/remove events
    if not sub_params['rm_event'].isna().item():
        sub_data.remove_event(rm_events)  
    
    if not sub_params['add_event'].isna().item():
        sub_data.add_event(add_events)
        
    # generate task events
    sub_data.generate_task_events(offset=offset)
    sub_data.confirm_events()
        
    # sync / normalize nirs
    sub_data.sync_normalize_nirs()
    
    # set parameters
    sub_data.set_params(sub_num, clench_array, mvc_events)
    
    # extract data to pickle
    sub_data.extract_nirs(pickle_path)
    sub_data.extract_emg(pickle_path, ignore_mvc_rest)
    
    end_time = time.time()
    
    print (sub_num+' data svaed to pickle.\nProcessing time: '+str(round(end_time-start_time, 3))+' seconds\n')
    print('-------------------------------\n')
        
    
    
### plot
sub = 'sub_086'
emg_clench_array = emg_clench.loc[sub][f'r_task_1']
emg_clench_array = pd.concat([emg_clench_array, emg_clench.loc[sub][f'r_rest_1']])
for i in range(2,16):
    emg_clench_array = pd.concat([emg_clench_array, emg_clench.loc[sub][f'r_task_{i}']])
    emg_clench_array = pd.concat([emg_clench_array, emg_clench.loc[sub][f'r_rest_{i}']])

plt.plot(np.arange(len(emg_clench_array))/2048, np.array(emg_clench_array))
