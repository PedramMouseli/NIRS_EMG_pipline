#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 14:06:21 2023

@author: moayedilab
"""

import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import loadmat
import pandas as pd
import numpy as np
from utils import normalize

class muscle_data:
    
    def __init__(self, sub_id, nirs_path, emg_path, plot_path, scaling_coef=1, stress_included=True):
        self.sub_id = sub_id
        self.nirs_path = nirs_path
        self.emg_path = emg_path
        self.plot_path = plot_path
        self.scaling_coef = scaling_coef
        self.stress_included = stress_included
        
        self.nirs_sf = 50.0
        self.emg_sf = 2048.0
        
        self.nirs_data = None
        self.emg_data = None
        self.events = None
        self.task_events = None
        self.nirs_start = None
        self.task_array = None
        self.mvc_events = None
        self.sub_num = None
        
    def import_emg(self, feedback=True):

        emg_data = loadmat(self.emg_path, squeeze_me=True, chars_as_strings=True)
        
        if feedback:
            emg_dict = {'time':emg_data["Time"], 'emg_masseter_r':emg_data["Data"][:,1], 'emg_masseter_l':emg_data["Data"][:,0], emg_data["Description"][2]:emg_data["Data"][:,2],
                    emg_data["Description"][3]:emg_data["Data"][:,3], emg_data["Description"][4]:emg_data["Data"][:,4]}
        else:
            emg_dict = {'time':emg_data["Time"], 'emg_masseter_r':emg_data["Data"][:,1], 'emg_masseter_l':emg_data["Data"][:,0]}
        
        self.emg_data = pd.DataFrame(data=emg_dict)
        
        print('EMG data imported!')
        
        return self
        
    def import_nirs(self):

        nirs_excel = pd.read_excel(self.nirs_path, header=None)
        
        channels_start = nirs_excel.index[nirs_excel[1]=='(Sample number)'].item()
        channels_end = nirs_excel.index[nirs_excel[1]=='(Event)'].item()
        
        channels_name = nirs_excel.loc[channels_start:channels_end,1].to_frame().reset_index(drop=True)
        
        nirs_data = nirs_excel.loc[channels_end+3:, :len(channels_name)-1].reset_index(drop=True)
        
        # changing channel names to labels
        channels_name.loc[channels_name[1].str.contains('[6359] Rx1-Tx1,Tx2,Tx3 TSI%', regex=False),1] = 'ltsi1'
        channels_name.loc[channels_name[1].str.contains('[6359] Rx1-Tx1,Tx2,Tx3 TSI Fit Factor', regex=False),1] = 'ltsifit1'
        channels_name.loc[channels_name[1].str.contains('[6341] Rx1-Tx1,Tx2,Tx3 TSI%', regex=False),1] = 'rtsi1'
        channels_name.loc[channels_name[1].str.contains('[6341] Rx1-Tx1,Tx2,Tx3 TSI Fit Factor', regex=False),1] = 'rtsifit1'
        channels_name.loc[channels_name[1].str.contains('[6359] Rx1-Tx1 tHb', regex=False),1] = 'lthb1'
        channels_name.loc[channels_name[1].str.contains('[6359] Rx1-Tx1 HbDiff', regex=False),1] = 'lhbdiff1'
        channels_name.loc[channels_name[1].str.contains('[6359] Rx1-Tx2 tHb', regex=False),1] = 'lthb2'
        channels_name.loc[channels_name[1].str.contains('[6359] Rx1-Tx2 HbDiff', regex=False),1] = 'lhbdiff2'
        channels_name.loc[channels_name[1].str.contains('[6359] Rx1-Tx3 tHb', regex=False),1] = 'lthb3'
        channels_name.loc[channels_name[1].str.contains('[6359] Rx1-Tx3 HbDiff', regex=False),1] = 'lhbdiff3'
        channels_name.loc[channels_name[1].str.contains('[6341] Rx1-Tx1 tHb', regex=False),1] = 'rthb1'
        channels_name.loc[channels_name[1].str.contains('[6341] Rx1-Tx1 HbDiff', regex=False),1] = 'rhbdiff1'
        channels_name.loc[channels_name[1].str.contains('[6341] Rx1-Tx2 tHb', regex=False),1] = 'rthb2'
        channels_name.loc[channels_name[1].str.contains('[6341] Rx1-Tx2 HbDiff', regex=False),1] = 'rhbdiff2'
        channels_name.loc[channels_name[1].str.contains('[6341] Rx1-Tx3 tHb', regex=False),1] = 'rthb3'
        channels_name.loc[channels_name[1].str.contains('[6341] Rx1-Tx3 HbDiff', regex=False),1] = 'rhbdiff3'
        channels_name.loc[channels_name[1].str.contains('[6359] Rx1-Tx1 O2Hb', regex=False),1] = 'lo2hb1'
        channels_name.loc[channels_name[1].str.contains('[6359] Rx1-Tx1 HHb', regex=False),1] = 'lhhb1'
        channels_name.loc[channels_name[1].str.contains('[6359] Rx1-Tx2 O2Hb', regex=False),1] = 'lo2hb2'
        channels_name.loc[channels_name[1].str.contains('[6359] Rx1-Tx2 HHb', regex=False),1] = 'lhhb2'
        channels_name.loc[channels_name[1].str.contains('[6359] Rx1-Tx3 O2Hb', regex=False),1] = 'lo2hb3'
        channels_name.loc[channels_name[1].str.contains('[6359] Rx1-Tx3 HHb', regex=False),1] = 'lhhb3'
        channels_name.loc[channels_name[1].str.contains('[6341] Rx1-Tx1 O2Hb', regex=False),1] = 'ro2hb1'
        channels_name.loc[channels_name[1].str.contains('[6341] Rx1-Tx1 HHb', regex=False),1] = 'rhhb1'
        channels_name.loc[channels_name[1].str.contains('[6341] Rx1-Tx2 O2Hb', regex=False),1] = 'ro2hb2'
        channels_name.loc[channels_name[1].str.contains('[6341] Rx1-Tx2 HHb', regex=False),1] = 'rhhb2'
        channels_name.loc[channels_name[1].str.contains('[6341] Rx1-Tx3 O2Hb', regex=False),1] = 'ro2hb3'
        channels_name.loc[channels_name[1].str.contains('[6341] Rx1-Tx3 HHb', regex=False),1] = 'rhhb3'
        channels_name.loc[channels_name[1].str.contains('(Sample number)', regex=False),1] = 'time'
        channels_name.loc[channels_name[1].str.contains('(Event)', regex=False),1] = 'events'
        
        
        for ch_num in range(len(channels_name)):
            nirs_data = nirs_data.rename(columns={ch_num:channels_name[1][ch_num]})
        
        
        events = np.array(nirs_data[nirs_data['events'].notnull()].index.tolist()) / self.nirs_sf
        
        nirs_data['time'] = nirs_data['time'] / self.nirs_sf
        
        self.nirs_data = nirs_data
        self.events = events
        
        print('NIRS data imported!')
        
        return self
    
    def sync_normalize_nirs(self):
        # first_event = self.nirs_data[self.nirs_data['events'].notnull()].index[0]
        self.nirs_data = self.nirs_data.loc[self.nirs_start:,:].reset_index(drop=True)
        
        baseline_end = int(round((self.events[1])*self.nirs_sf))
        baseline_start = int(round((self.events[1] -60)*self.nirs_sf))
        
        for channel in self.nirs_data.columns.values:
            if channel != 'time' and channel != 'events' and channel !='ltsi1' and channel != 'rtsi1':
                self.nirs_data[channel] = normalize(self.nirs_data[channel], base_events=[baseline_start, baseline_end], baseline=True, mean=True, std=True)

        self.nirs_data['time'] = (self.nirs_data['time'] - self.nirs_data['time'][0])
        
        return self
    
    def add_event(self, event):
        self.events = np.sort(np.append(self.events, event))
        
        return self
    
    def remove_event(self, rm_index):
        int_index = [int(x) for x in rm_index]
        self.events = np.delete(self.events, int_index)
        
        return self
    
    def generate_task_events(self, offset):
        self.nirs_data['time'] *= self.scaling_coef
        nirs_annotations = (self.events - self.events[0]) * self.scaling_coef
        
        num_events = int(2*((len(self.emg_data['time'])/self.emg_sf)//60)-3)

        task_events = np.zeros(num_events)
        task_events[0] = nirs_annotations[1] + offset
        for i in range(1, num_events):
            task_events[i] = task_events[i-1]+30
            
        self.task_events = task_events
        self.synced_events = nirs_annotations
        
        return self
    
    def confirm_events(self):
        self.nirs_start = int(round(self.events[0]*self.nirs_sf))
        self.events = np.copy(self.synced_events)
        
        return self
    
    def check_events(self):
        if self.stress_included and len(self.events) != 13:
            print('There should be 13 events but ' + str(len(self.events)) + ' events saved')
            
        elif not self.stress_included and len(self.events) != 4:
            print('There should be 4 events but ' + str(len(self.events)) + ' events saved')
            
        else: print('Correct number of events! '+ str(len(self.events)) + ' events saved')
        
        return self.events
    
    def set_params(self, sub_num, task_array, mvc_events=[3,4]):
        self.sub_num = sub_num
        self.task_array = task_array
        self.mvc_events = mvc_events
        
        return self
    
    def visualization(self, plot_type='both',events=True, side="r", nirschannel=1, nirsmeasure='sto2',
                      task_array=None, mvc_events=None, save_plot=False):
        
        """
        Plot the EMG and NIRS signals.


        Parameters
        ----------
        plot_type : str
            Can be 'emg', 'nirs', 'both'. The default is 'both'.
            
        nirschannel : int 
            NIRS channel to plot. The default is 1.
        
        nirsmeasures : str
            NIRS measure to plot. Options are: "o2hb", "hhb", "thb", "hbdiff". The default is 'sto2'.
        

        Returns
        -------
        Plots of EMG and NIRS signals.
        
        """
        
        if side == 'r':
            side_label = 'right'
        elif side == 'l':
            side_label = 'left'
        
        if plot_type=='emg' or plot_type=='both':
            
            if plot_type=='both':
                fig = plt.figure(f'{self.sub_id}; {side_label} side, NIRS {nirsmeasure}, EMG', figsize=[27,8])
                border_width = 0.05
                fig.add_axes([0+0.5*border_width,0+2*border_width,1-border_width,1-4*border_width])
            
            if plot_type=='emg':
                fig = plt.figure(f'{self.sub_id}; {side_label} side, EMG', figsize=[20,5])
                border_width = 0.05
                fig.add_axes([0+0.7*border_width,0+2*border_width,1-border_width,1-4*border_width])
            
            plt.plot(self.emg_data['time'], self.emg_data[f'emg_masseter_{side}'])
            
            if events:
                for xc in self.synced_events[:mvc_events[1]]:
                    plt.axvline(x=xc, color='k', linestyle='--')
                
                mvc_start = self.synced_events[mvc_events[0]-1]
                mvc_end = self.synced_events[mvc_events[1]-1]
                
                plt.axvspan(mvc_start, mvc_end, alpha=0.3, color="green")
                    
                for i in task_array:
                    task_start = int(round(self.task_events[(i*2)-2]))
                    task_end = int(round(self.task_events[(i*2)-1]))
                    plt.axvline(x=task_start, color='r', linestyle='--')
                    plt.axvline(x=task_end, color='r', linestyle='--')
                    plt.axvspan(task_start, task_end, alpha=0.3, color="blue")
            
            # plt.xlim(left=0, right=1400)
            plt.xlabel("time (sec)")
            plt.ylabel("\u03BC V")
            # plt.legend([emgchannel + ' masseter EMG'])
            plt.title(f'{self.sub_id} \n{side_label} side EMG')
            plt.show()
            if save_plot:
                plt.savefig(f'{self.plot_path}/EMG_{self.sub_id}_{side_label} masseter.png',dpi=600)
        
        
        if plot_type=='nirs' or plot_type=='both':
            
            if plot_type=='both':
                fig = plt.figure(f'{self.sub_id}; {side_label} side, NIRS {nirsmeasure}, EMG', figsize=[27,8])
                border_width = 0.05
                fig.add_axes([0+0.5*border_width,0+2*border_width,1-border_width,1-4*border_width])
                mag = 30 # magnification factor
                
            if plot_type=='nirs':
                fig = plt.figure(f'{self.sub_id}; {side_label} side, NIRS {nirsmeasure} Tx{nirschannel}', figsize=[27,8])
                border_width = 0.05
                fig.add_axes([0+0.5*border_width,0+2*border_width,1-border_width,1-4*border_width])
                mag = 1 # magnification factor
            
            if self.stress_included:
                plt.plot(self.nirs_data['time'][:int(round(self.synced_events[mvc_events[1]]))*int(self.nirs_sf)],
                         self.nirs_data[side + nirsmeasure + str(nirschannel)][:int(round(self.synced_events[mvc_events[1]]))*int(self.nirs_sf)]*mag)
            else:
                plt.plot(self.nirs_data['time'] , self.nirs_data[side + nirsmeasure + str(nirschannel)]*mag)
                
            if events:
                for xc in self.synced_events[:mvc_events[1]]:
                    plt.axvline(x=xc, color='k', linestyle='--')
                
                mvc_start = self.synced_events[mvc_events[0]-1]
                mvc_end = self.synced_events[mvc_events[1]-1]
                
                plt.axvspan(mvc_start, mvc_end, alpha=0.3, color="green")
                
                for i in task_array:
                    task_start = int(round(self.task_events[(i*2)-2]))
                    task_end = int(round(self.task_events[(i*2)-1]))
                    plt.axvline(x=task_start, color='r', linestyle='--')
                    plt.axvline(x=task_end, color='r', linestyle='--')
                    plt.axvspan(task_start, task_end, alpha=0.3, color="blue")
            
            # plt.xlim(left=0, right=1400)
            plt.xlabel("time (sec)")
            plt.ylabel(nirsmeasure)
            
            # plt.legend(['Right masseter '+nirsmeasure+' Tx'+str(nirschannel)])
            plt.title(f'{self.sub_id}\n{side_label} masseter {nirsmeasure} Tx {nirschannel}')
            plt.show()
            if save_plot:
                plt.savefig(f'{self.plot_path}/NIRS_{self.sub_id}_{side_label} masseter {nirsmeasure} Tx{nirschannel}.png',dpi=600)
    
    
    def extract_nirs(self, pickle_path):

        trial_dict = {}
        mvc_trial = {}
        
        for j,i in enumerate(self.task_array):
        
            task_start = int(round(self.task_events[(i*2)-2]*self.nirs_sf))
            task_end = int(round(self.task_events[(i*2)-1]*self.nirs_sf))
            rest_end = int(round(self.task_events[i*2]*self.nirs_sf))
            
            for side in ["r","l"]:
                for measure in ["o2hb","thb","tsi"]:
                    for sensor in ["1","2","3"]:
                        for mode in ["task","rest"]:
                            if measure=="tsi" and sensor!="1":
                                continue
                            if mode=="task":
                                trial_dict.update({side+"_"+measure+sensor+"_"+mode+"_"+str(j+1):np.array(self.nirs_data[side+measure+sensor])[task_start:task_end]})
                            elif mode=="rest":
                                trial_dict.update({side+"_"+measure+sensor+"_"+mode+"_"+str(j+1):np.array(self.nirs_data[side+measure+sensor])[task_end:rest_end]})
        
        
        mvc_start = int(round((self.events[self.mvc_events[0]-1])*self.nirs_sf))
        mvc_end = int(round((self.events[self.mvc_events[1]-1])*self.nirs_sf))
        mvc_rest_end_1 = int(round((self.events[self.mvc_events[1]-1] +30)*self.nirs_sf))
        mvc_rest_end_2 = int(round((self.events[self.mvc_events[1]-1] +60)*self.nirs_sf))
        pre_rest2_start = int(round((self.events[1] -30)*self.nirs_sf))
        pre_rest2_end = int(round((self.events[1])*self.nirs_sf))
        pre_rest1_start = int(round((self.events[1] -60)*self.nirs_sf))
        
        for side in ["r","l"]:
            for measure in ["o2hb","thb","tsi"]:
                for sensor in ["1","2","3"]:
                    for mode in ["mvc","rest1","rest2","pre_rest1","pre_rest2"]:
                        if measure=="tsi" and sensor!="1":
                            continue
                        if mode=="mvc":
                            mvc_trial.update({side+"_"+measure+sensor+"_"+mode:np.array(self.nirs_data[side+measure+sensor])[mvc_start:mvc_end]})
                        elif mode=="rest1":
                            trial_dict.update({side+"_"+measure+sensor+"_mvc_"+mode:np.array(self.nirs_data[side+measure+sensor])[mvc_end:mvc_rest_end_1]})
                        elif mode=="rest2":
                            trial_dict.update({side+"_"+measure+sensor+"_mvc_"+mode:np.array(self.nirs_data[side+measure+sensor])[mvc_rest_end_1:mvc_rest_end_2]})
                        elif mode=="pre_rest1":
                            trial_dict.update({side+"_"+measure+sensor+"_"+mode:np.array(self.nirs_data[side+measure+sensor])[pre_rest1_start:pre_rest2_start]})
                        elif mode=="pre_rest2":
                            trial_dict.update({side+"_"+measure+sensor+"_"+mode:np.array(self.nirs_data[side+measure+sensor])[pre_rest2_start:pre_rest2_end]})
         
                            
        trial_df = pd.DataFrame(data=trial_dict)
        mvc_df = pd.DataFrame(data=mvc_trial)
        
        trial_df.to_pickle(f'{pickle_path}/nirs/{self.sub_num}_nirs.pkl')
        mvc_df.to_pickle(f'{pickle_path}/nirs/mvc/{self.sub_num}_mvc_nirs.pkl')



    def extract_emg(self, pickle_path, ignore_mvc_rest=0):

        trial_dict = {}
        mvc_trial = {}
        
        for j,i in enumerate(self.task_array):
        
            task_start = int(round(self.task_events[(i*2)-2]*self.emg_sf))
            task_end = int(round(self.task_events[(i*2)-1]*self.emg_sf))
            rest_end = int(round(self.task_events[i*2]*self.emg_sf))
            
            for side in ["r","l"]:
                for mode in ["task","rest"]:
                    if mode=="task":
                        trial_dict.update({side+"_"+mode+"_"+str(j+1):np.array(self.emg_data["emg_masseter_"+side])[task_start:task_end]})
                    elif mode=="rest":
                        trial_dict.update({side+"_"+mode+"_"+str(j+1):np.array(self.emg_data["emg_masseter_"+side])[task_end:rest_end]})
        
        
        mvc_start = int(round((self.events[self.mvc_events[0]-1])*self.emg_sf))
        mvc_end = int(round((self.events[self.mvc_events[1]-1])*self.emg_sf))
        mvc_rest_end_1 = int(round((self.events[self.mvc_events[1]-1] +30)*self.emg_sf))
        mvc_rest_end_2 = int(round((self.events[self.mvc_events[1]-1] +60)*self.emg_sf))
        pre_rest2_start = int(round((self.events[1] -30)*self.emg_sf))
        pre_rest2_end = int(round((self.events[1])*self.emg_sf))
        pre_rest1_start = int(round((self.events[1] -60)*self.emg_sf))
        
        if ignore_mvc_rest:
            modes = ["mvc","pre_rest1","pre_rest2"]
        else: modes = ["mvc","rest1","rest2","pre_rest1","pre_rest2"]
        for side in ["r","l"]:
            for mode in modes:
                if mode=="mvc":
                    mvc_trial.update({side+"_"+mode:np.array(self.emg_data["emg_masseter_"+side])[mvc_start:mvc_end]})
                elif mode=="rest1":
                    trial_dict.update({side+"_mvc_"+mode:np.array(self.emg_data["emg_masseter_"+side])[mvc_end:mvc_rest_end_1]})
                elif mode=="rest2":
                    trial_dict.update({side+"_mvc_"+mode:np.array(self.emg_data["emg_masseter_"+side])[mvc_rest_end_1:mvc_rest_end_2]})
                elif mode=="pre_rest1":
                    trial_dict.update({side+"_"+mode:np.array(self.emg_data["emg_masseter_"+side])[pre_rest1_start:pre_rest2_start]})
                elif mode=="pre_rest2":
                    trial_dict.update({side+"_"+mode:np.array(self.emg_data["emg_masseter_"+side])[pre_rest2_start:pre_rest2_end]})
         
                            
        trial_df = pd.DataFrame(data=trial_dict)
        mvc_df = pd.DataFrame(data=mvc_trial)
            
        trial_df.to_pickle(f'{pickle_path}/emg/{self.sub_num}_emg.pkl')
        mvc_df.to_pickle(f'{pickle_path}/emg/mvc/{self.sub_num}_mvc_emg.pkl')
        