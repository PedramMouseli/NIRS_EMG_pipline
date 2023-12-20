#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 21:30:04 2022

@author: moayedilab
"""
import numpy as np
from sklearn.linear_model import LinearRegression
import antropy as ant
from scipy import signal
from scipy.signal import periodogram, welch
import time
import matplotlib.pyplot as plt
import pywt

def normalize(data, base_events=[], baseline=False, mean=True, std=True, sep_baseline=False):
    """
    Normalize the signal.

    Parameters
    ----------
    data : array
        Signal to normalize.
        
    mean : Bool, optional
        mean subtraction. The default is True.
        
    std : Bool, optional
        dividing by standard deviation. The default is True.

    Returns
    -------
    data : array
        Normalized signal.

    """
    if sep_baseline:
        data[0] -= data[1].mean()
        data[0] /= data[1].std()
        data = data[0]
    else:      
        if mean:
            if baseline:
                data -= data[base_events[0]:base_events[1]].mean()
            else:
                data -= data.mean()
        if std:
            data /= data.std()
    
    return data

def extract_feat(sub_list, nirs_clench, mvc_nirs_clench, measure='tsi', side='r', sensor='1'):
    
    task_coef = np.zeros([len(sub_list),15])
    task_intercept = np.zeros([len(sub_list),15])
    task_median = np.zeros([len(sub_list),15])

    rest_coef = np.zeros([len(sub_list),15])
    rest_intercept = np.zeros([len(sub_list),15])
    rest_median = np.zeros([len(sub_list),15])

    mvc_coef = np.zeros(len(sub_list))
    mvc_intercept = np.zeros(len(sub_list))
    mvc_median = np.zeros(len(sub_list))

    pre_rest_coef = np.zeros(len(sub_list))
    pre_rest_intercept = np.zeros(len(sub_list))
    pre_rest_median = np.zeros(len(sub_list))

    mvc_rest_coef = np.zeros(len(sub_list))
    mvc_rest_intercept = np.zeros(len(sub_list))
    mvc_rest_median = np.zeros(len(sub_list))
    
    mvc_len = np.zeros(len(sub_list))

    for j,sub in enumerate(sub_list):
        
        mvc_len[j] = len(np.array(mvc_nirs_clench.loc[sub][side+"_"+measure+sensor+"_mvc"]))/50
        y = np.array(mvc_nirs_clench.loc[sub][side+"_"+measure+sensor+"_mvc"])[100:]
        x = np.array(range(len(y))).reshape(-1,1)
        reg = LinearRegression().fit(x, y)
        mvc_coef[j] = reg.coef_[0]
        mvc_intercept[j] = reg.intercept_
        mvc_median[j] = np.median(y)
        
        y = np.array(nirs_clench.loc[sub][side+"_"+measure+sensor+"_pre_rest2"])
        x = np.array(range(len(y))).reshape(-1,1)
        reg = LinearRegression().fit(x, y)
        pre_rest_coef[j] = reg.coef_[0]
        pre_rest_intercept[j] = reg.intercept_
        pre_rest_median[j] = np.median(y)
        
        y = np.array(nirs_clench.loc[sub][side+"_"+measure+sensor+"_mvc_rest1"])[:250]
        x = np.array(range(len(y))).reshape(-1,1)
        reg = LinearRegression().fit(x, y)
        mvc_rest_coef[j] = reg.coef_[0]
        mvc_rest_intercept[j] = reg.intercept_
        mvc_rest_median[j] = np.median(y)
        
        for i in range(1,16):
            y = np.array(nirs_clench.loc[sub][side+"_"+measure+sensor+"_task_"+str(i)])[250:1250]
            x = np.array(range(len(y))).reshape(-1,1)
            reg = LinearRegression().fit(x, y)
            task_coef[j,i-1] = reg.coef_[0]
            task_intercept[j,i-1] = reg.intercept_
            task_median[j,i-1] = np.median(y)
            
            y = np.array(nirs_clench.loc[sub][side+"_"+measure+sensor+"_rest_"+str(i)])[250:1250]
            x = np.array(range(len(y))).reshape(-1,1)
            reg_rest = LinearRegression().fit(x, y)
            rest_coef[j,i-1] = reg_rest.coef_[0]
            rest_intercept[j,i-1] = reg_rest.intercept_
            rest_median[j,i-1] = np.median(y)
            
         
    intercept_diff = task_intercept - rest_intercept
    median_diff = task_median - rest_median
    median_diff_mvc = mvc_median - mvc_rest_median

         
    ##### fit a linear model to differences
    diff_coef = np.zeros(len(sub_list))
    diff_intercept = np.zeros(len(sub_list))
    all_task_coef = np.zeros(len(sub_list))
    all_task_intercept = np.zeros(len(sub_list))
    all_rest_coef = np.zeros(len(sub_list))
    all_rest_intercept = np.zeros(len(sub_list))

    for j,sub in enumerate(sub_list):
        y = median_diff[j,:16]
        x = np.array(range(len(y))).reshape(-1,1)
        reg = LinearRegression().fit(x, y)
        diff_coef[j] = reg.coef_[0]
        diff_intercept[j] = reg.intercept_
        
        y = task_median[j,:16]
        x = np.array(range(len(y))).reshape(-1,1)
        reg = LinearRegression().fit(x, y)
        all_task_coef[j] = reg.coef_[0]
        all_task_intercept[j] = reg.intercept_
        
        y = rest_median[j,:16]
        x = np.array(range(len(y))).reshape(-1,1)
        reg = LinearRegression().fit(x, y)
        all_rest_coef[j] = reg.coef_[0]
        all_rest_intercept[j] = reg.intercept_
        
    ##### count zero scrossing
    zero_cross = np.zeros(len(sub_list))
    diff_std = np.zeros(len(sub_list))

    for i in range(len(sub_list)):
        zero_cross[i] = ((median_diff[i,:][:-1] * median_diff[i,:][1:]) < 0).sum()
        diff_std[i] = intercept_diff[i,:].std()
        
    return diff_coef, diff_intercept, zero_cross, mvc_len, mvc_coef, mvc_rest_coef, median_diff_mvc, mvc_median, median_diff, task_median, pre_rest_median, all_task_coef, all_rest_coef

def extraxct_corr(sub_list, nirs_clench):
    task_coef = np.zeros([len(sub_list),15])
    task_intercept = np.zeros([len(sub_list),15])
    task_median = np.zeros([len(sub_list),15])

    rest_coef = np.zeros([len(sub_list),15])
    rest_intercept = np.zeros([len(sub_list),15])
    rest_median = np.zeros([len(sub_list),15])

    mvc_coef = np.zeros(len(sub_list))
    mvc_intercept = np.zeros(len(sub_list))
    mvc_median = np.zeros(len(sub_list))

    pre_rest_coef = np.zeros(len(sub_list))
    pre_rest_intercept = np.zeros(len(sub_list))
    pre_rest_median = np.zeros(len(sub_list))

    mvc_rest_coef = np.zeros(len(sub_list))
    mvc_rest_intercept = np.zeros(len(sub_list))
    mvc_rest_median = np.zeros(len(sub_list))
    
    mvc_len = np.zeros(len(sub_list))

    for j,sub in enumerate(sub_list):
        
        mvc_len[j] = len(np.array(mvc_nirs_clench.loc[sub][side+"_"+measure+sensor+"_mvc"]))/50
        y = np.array(mvc_nirs_clench.loc[sub][side+"_"+measure+sensor+"_mvc"])[100:]
        x = np.array(range(len(y))).reshape(-1,1)
        reg = LinearRegression().fit(x, y)
        mvc_coef[j] = reg.coef_[0]
        mvc_intercept[j] = reg.intercept_
        mvc_median[j] = np.median(y)
        
        y = np.array(nirs_clench.loc[sub][side+"_"+measure+sensor+"_pre_rest2"])
        x = np.array(range(len(y))).reshape(-1,1)
        reg = LinearRegression().fit(x, y)
        pre_rest_coef[j] = reg.coef_[0]
        pre_rest_intercept[j] = reg.intercept_
        pre_rest_median[j] = np.median(y)
        
        y = np.array(nirs_clench.loc[sub][side+"_"+measure+sensor+"_mvc_rest1"])[:250]
        x = np.array(range(len(y))).reshape(-1,1)
        reg = LinearRegression().fit(x, y)
        mvc_rest_coef[j] = reg.coef_[0]
        mvc_rest_intercept[j] = reg.intercept_
        mvc_rest_median[j] = np.median(y)
        
        for i in range(1,16):
            y = np.array(nirs_clench.loc[sub][side+"_"+measure+sensor+"_task_"+str(i)])[250:1250]
            x = np.array(range(len(y))).reshape(-1,1)
            reg = LinearRegression().fit(x, y)
            task_coef[j,i-1] = reg.coef_[0]
            task_intercept[j,i-1] = reg.intercept_
            task_median[j,i-1] = np.median(y)
            
            y = np.array(nirs_clench.loc[sub][side+"_"+measure+sensor+"_rest_"+str(i)])[250:1250]
            x = np.array(range(len(y))).reshape(-1,1)
            reg_rest = LinearRegression().fit(x, y)
            rest_coef[j,i-1] = reg_rest.coef_[0]
            rest_intercept[j,i-1] = reg_rest.intercept_
            rest_median[j,i-1] = np.median(y)

def entropy(sub_list, emg_clench, mvc_emg_clench, side='r'):
    
    task_ent = np.zeros([len(sub_list),15])
    rest_ent = np.zeros([len(sub_list),15])
    mvc_ent = np.zeros(len(sub_list))
    pre_rest_ent = np.zeros(len(sub_list))
    mvc_rest_ent = np.zeros(len(sub_list))
    sf = 2048

    for j,sub in enumerate(sub_list):      
        print(sub)
        start = time.time()
        
        pre_rest = filt_notch(filt(np.array(emg_clench.loc[sub][side+"_pre_rest2"])))
        # pre_rest_ent[j] = spectral_entropy(pre_rest, sf=sf, method='fft', normalize=True)
        pre_rest_ent[j] = ant.sample_entropy(pre_rest, order=2)
        
        y = filt_notch(filt(np.array(mvc_emg_clench.loc[sub][side+"_mvc"])))[2*sf:]
        # mvc_ent[j] = spectral_entropy(y, sf=sf, method='fft', normalize=True)
        mvc_ent[j] = ant.sample_entropy(y, order=2)
        
        # y = filt_notch(filt(np.array(emg_clench.loc[sub][side+"_mvc_rest1"])))[5*sf:25*sf]
        # # mvc_rest_ent[j] = spectral_entropy(y, sf=sf, method='fft', normalize=True)
        # mvc_rest_ent[j] = ant.sample_entropy(y, order=2)
        
        for i in range(1,16):
            y = filt_notch(filt(np.array(emg_clench.loc[sub][side+"_task_"+str(i)])))[5*sf:25*sf]
            # task_ent[j,i-1] = spectral_entropy(y, sf=sf, method='fft', normalize=True)
            task_ent[j,i-1] = ant.sample_entropy(y, order=2)
            
            y = filt_notch(filt(np.array(emg_clench.loc[sub][side+"_rest_"+str(i)])))[5*sf:25*sf]
            # rest_ent[j,i-1] = spectral_entropy(y, sf=sf, method='fft', normalize=True)
            rest_ent[j,i-1] = ant.sample_entropy(y, order=2)
            
        end = time.time()
        print(sub + ' took ' + str(round(end-start, 3)) + ' seconds\n')
            
    ##### fit a linear model to entropy values
    ent_coef_task = np.zeros(len(sub_list))
    ent_std_task = np.zeros(len(sub_list))
    ent_coef_rest = np.zeros(len(sub_list))
    ent_std_rest = np.zeros(len(sub_list))
    
    ent_diff = task_ent - rest_ent
    
    ent_diff_coef = np.zeros(len(sub_list))
    ent_diff_intercept = np.zeros(len(sub_list))
    ent_zero_cross = np.zeros(len(sub_list))

    for j,sub in enumerate(sub_list):
        y_task = task_ent[j,:]
        x = np.array(range(len(y_task))).reshape(-1,1)/100
        reg_task = LinearRegression().fit(x, y_task)
        ent_coef_task[j] = reg_task.coef_[0]
        ent_std_task[j] = task_ent[j,:].std()
        
        y_rest = rest_ent[j,:]
        reg_rest = LinearRegression().fit(x, y_rest)
        ent_coef_rest[j] = reg_rest.coef_[0]
        ent_std_rest[j] = rest_ent[j,:].std()
        
        y = ent_diff[j,:]
        x = np.array(range(len(y))).reshape(-1,1)/100
        reg = LinearRegression().fit(x, y)
        ent_diff_coef[j] = reg.coef_[0]
        ent_diff_intercept[j] = reg.intercept_
        
        ent_zero_cross[j] = ((ent_diff[j,:][:-1] * ent_diff[j,:][1:]) < 0).sum()
            
    return task_ent, rest_ent, mvc_ent, pre_rest_ent, ent_coef_task, ent_std_task, ent_coef_rest, ent_std_rest, ent_diff_coef, ent_diff_intercept, ent_zero_cross


def extraxt_freq_power(sub_list, emg_clench, mvc_emg_clench, side='r'):
    
    task_fp = np.zeros([len(sub_list),15])
    rest_fp = np.zeros([len(sub_list),15])
    mvc_fp = np.zeros(len(sub_list))
    pre_rest_fp = np.zeros(len(sub_list))

    sf = 2048

    for j,sub in enumerate(sub_list):      
        print(sub)
        start = time.time()
        
        pre_rest = np.array(emg_clench.loc[sub][side+"_pre_rest2"])
        # pre_rest_ent[j] = spectral_entropy(pre_rest, sf=sf, method='fft', normalize=True)
        pre_rest_fp[j] = freq_power(pre_rest)
        
        y = np.array(mvc_emg_clench.loc[sub][side+"_mvc"])[2*sf:]
        # mvc_ent[j] = spectral_entropy(y, sf=sf, method='fft', normalize=True)
        mvc_fp[j] = freq_power(y)
        
        # y = filt_notch(filt(np.array(emg_clench.loc[sub][side+"_mvc_rest1"])))[5*sf:25*sf]
        # # mvc_rest_ent[j] = spectral_entropy(y, sf=sf, method='fft', normalize=True)
        # mvc_rest_ent[j] = ant.sample_entropy(y, order=2)
        
        for i in range(1,16):
            y = filt_notch(filt(np.array(emg_clench.loc[sub][side+"_task_"+str(i)])))[5*sf:25*sf]
            # task_ent[j,i-1] = spectral_entropy(y, sf=sf, method='fft', normalize=True)
            task_fp[j,i-1] = freq_power(y)
            
            y = filt_notch(filt(np.array(emg_clench.loc[sub][side+"_rest_"+str(i)])))[5*sf:25*sf]
            # rest_ent[j,i-1] = spectral_entropy(y, sf=sf, method='fft', normalize=True)
            rest_fp[j,i-1] = freq_power(y)
            
        end = time.time()
        print(sub + ' took ' + str(round(end-start, 3)) + ' seconds\n')
            
    ##### fit a linear model to entropy values
    fp_coef_task = np.zeros(len(sub_list))
    fp_std_task = np.zeros(len(sub_list))
    fp_coef_rest = np.zeros(len(sub_list))
    fp_std_rest = np.zeros(len(sub_list))
    
    fp_diff = task_fp - rest_fp
    
    fp_diff_coef = np.zeros(len(sub_list))
    fp_diff_intercept = np.zeros(len(sub_list))
    fp_zero_cross = np.zeros(len(sub_list))

    for j,sub in enumerate(sub_list):
        y_task = task_fp[j,:]
        x = np.array(range(len(y_task))).reshape(-1,1)/100
        reg_task = LinearRegression().fit(x, y_task)
        fp_coef_task[j] = reg_task.coef_[0]
        fp_std_task[j] = task_fp[j,:].std()
        
        y_rest = rest_fp[j,:]
        reg_rest = LinearRegression().fit(x, y_rest)
        fp_coef_rest[j] = reg_rest.coef_[0]
        fp_std_rest[j] = rest_fp[j,:].std()
        
        y = fp_diff[j,:]
        x = np.array(range(len(y))).reshape(-1,1)/100
        reg = LinearRegression().fit(x, y)
        fp_diff_coef[j] = reg.coef_[0]
        fp_diff_intercept[j] = reg.intercept_
        
        fp_zero_cross[j] = ((fp_diff[j,:][:-1] * fp_diff[j,:][1:]) < 0).sum()
            
    return task_fp, rest_fp, mvc_fp, pre_rest_fp, fp_coef_task, fp_std_task, fp_coef_rest, fp_std_rest, fp_diff_coef, fp_diff_intercept, fp_zero_cross

def extraxt_wt_power(sub_list, emg_clench, mvc_emg_clench, side='r'):
    
    task_wt = np.zeros([len(sub_list),9,15])
    rest_wt = np.zeros([len(sub_list),9,15])
    # mvc_wt = np.zeros([len(sub_list),9])
    # pre_rest_wt = np.zeros([len(sub_list),9])

    sf = 2048

    for j,sub in enumerate(sub_list):      
        print(sub)
        start = time.time()
        
        # pre_rest = np.array(emg_clench.loc[sub][side+"_pre_rest2"])
        # # pre_rest_ent[j] = spectral_entropy(pre_rest, sf=sf, method='fft', normalize=True)
        # pre_rest_wt[j] = wt_power(pre_rest)
        
        # y = np.array(mvc_emg_clench.loc[sub][side+"_mvc"])[2*sf:]
        # # mvc_ent[j] = spectral_entropy(y, sf=sf, method='fft', normalize=True)
        # mvc_wt[j] = wt_power(y)
        
        # y = filt_notch(filt(np.array(emg_clench.loc[sub][side+"_mvc_rest1"])))[5*sf:25*sf]
        # # mvc_rest_ent[j] = spectral_entropy(y, sf=sf, method='fft', normalize=True)
        # mvc_rest_ent[j] = ant.sample_entropy(y, order=2)
        
        for i in range(1,16):
            y = filt_notch(filt(np.array(emg_clench.loc[sub][side+"_task_"+str(i)])))[5*sf:25*sf]
            # task_ent[j,i-1] = spectral_entropy(y, sf=sf, method='fft', normalize=True)
            task_wt[j,:,i-1] = wt_power(y)
            
            y = filt_notch(filt(np.array(emg_clench.loc[sub][side+"_rest_"+str(i)])))[5*sf:25*sf]
            # rest_ent[j,i-1] = spectral_entropy(y, sf=sf, method='fft', normalize=True)
            rest_wt[j,:,i-1] = wt_power(y)
            
        end = time.time()
        print(sub + ' took ' + str(round(end-start, 3)) + ' seconds\n')
            
    ##### fit a linear model to entropy values
    wt_coef_task = np.zeros([len(sub_list),9])
    wt_std_task = np.zeros([len(sub_list),9])
    wt_coef_rest = np.zeros([len(sub_list),9])
    wt_std_rest = np.zeros([len(sub_list),9])
    
    wt_diff = task_wt - rest_wt
    
    wt_diff_coef = np.zeros([len(sub_list),9])
    wt_diff_intercept = np.zeros([len(sub_list),9])
    wt_zero_cross = np.zeros([len(sub_list),9])

    for j,sub in enumerate(sub_list):
        for i in range(9):
            y_task = task_wt[j,i,:]
            x = np.array(range(len(y_task))).reshape(-1,1)/100
            reg_task = LinearRegression().fit(x, y_task)
            wt_coef_task[j,i] = reg_task.coef_[0]
            wt_std_task[j,i] = task_wt[j,:].std()
            
            y_rest = rest_wt[j,i,:]
            reg_rest = LinearRegression().fit(x, y_rest)
            wt_coef_rest[j,i] = reg_rest.coef_[0]
            wt_std_rest[j,i] = rest_wt[j,:].std()
            
            y = wt_diff[j,i,:]
            x = np.array(range(len(y))).reshape(-1,1)/100
            reg = LinearRegression().fit(x, y)
            wt_diff_coef[j,i] = reg.coef_[0]
            wt_diff_intercept[j,i] = reg.intercept_
            
            wt_zero_cross[j,i] = ((wt_diff[j,i,:][:-1] * wt_diff[j,i,:][1:]) < 0).sum()
            
    return task_wt, rest_wt, wt_coef_task, wt_std_task, wt_coef_rest, wt_std_rest, wt_diff_coef, wt_diff_intercept, wt_zero_cross


def filt(data, filtord=3, cutoff=[1,512], ftype='bandpass', fs=2048):
    """
    Filtering the signal.

    Parameters
    ----------
    data : array
        signal to filter.
        
    filtord : int, optional
        Filter order. The default is 3.
        
    cutoff : double, optional
        Cuttoff frequency in Hz. The default is 5.
        
    ftype : str, optional
        Filter type. 'lp' for low-pass, 'hp' for high-pass, 'bandpass' for bandpass, and 'bandstop'.  The default is 'lp'.
        
    fs : int, optional
        Sampling frequency. The default is 50.

    Returns
    -------
    data : array
        Filtered signal.

    """
    # b, a = signal.butter(filtord, cutoff, ftype, fs=fs)
    # data = signal.filtfilt(b, a, data)
    return data

def filt_notch(data, Q=60, fs=2048):
    """
    Filtering the signal.

    Parameters
    ----------
    data : array
        signal to filter.
        
    filtord : int, optional
        Filter order. The default is 3.
        
    cutoff : double, optional
        Cuttoff frequency in Hz. The default is 5.
        
    ftype : str, optional
        Filter type. 'lp' for low-pass, 'hp' for high-pass, 'bandpass' for bandpass, and 'bandstop'.  The default is 'lp'.
        
    fs : int, optional
        Sampling frequency. The default is 50.

    Returns
    -------
    data : array
        Filtered signal.

    """
    # for i in [60, 120, 180]:
    #     b, a = signal.iirnotch(w0=i, Q=Q, fs=fs)
    #     data = signal.filtfilt(b, a, data)
    return data

def freq_power(x, sf=2048, freq_band=[15,230]):
    x = np.asarray(x)
    f, Pxx_den = signal.welch(filt_notch(filt(normalize(x))), sf, nperseg=1024)
    power_x = np.sum(Pxx_den[int(freq_band[0]/2):int(freq_band[1]/2)])
    
    return power_x

def wt_power(x, wavelet='db5'):
    x = np.asarray(x)
    list_coeff = pywt.wavedec(x, wavelet)
    power = []
    for i in range(4,13):
        power.append(np.sum(list_coeff[i]**2))
        
    power = np.array(power)
    
    return power

def spectral_entropy(x, sf, method="fft", nperseg=None, normalize=False, axis=-1):
    """Spectral Entropy.

    Parameters
    ----------
    x : list or np.array
        1D or N-D data.
    sf : float
        Sampling frequency, in Hz.
    method : str
        Spectral estimation method:

        * ``'fft'`` : Fourier Transform (:py:func:`scipy.signal.periodogram`)
        * ``'welch'`` : Welch periodogram (:py:func:`scipy.signal.welch`)
    nperseg : int or None
        Length of each FFT segment for Welch method.
        If None (default), uses scipy default of 256 samples.
    normalize : bool
        If True, divide by log2(psd.size) to normalize the spectral entropy
        between 0 and 1. Otherwise, return the spectral entropy in bit.
    axis : int
        The axis along which the entropy is calculated. Default is -1 (last).

    Returns
    -------
    se : float
        Spectral Entropy

    Notes
    -----
    Spectral Entropy is defined to be the Shannon entropy of the power
    spectral density (PSD) of the data:

    .. math:: H(x, sf) =  -\\sum_{f=0}^{f_s/2} P(f) \\log_2[P(f)]

    Where :math:`P` is the normalised PSD, and :math:`f_s` is the sampling
    frequency.

    References
    ----------
    - Inouye, T. et al. (1991). Quantification of EEG irregularity by
      use of the entropy of the power spectrum. Electroencephalography
      and clinical neurophysiology, 79(3), 204-210.

    - https://en.wikipedia.org/wiki/Spectral_density

    - https://en.wikipedia.org/wiki/Welch%27s_method

    Examples
    --------
    Spectral entropy of a pure sine using FFT

    >>> import numpy as np
    >>> import antropy as ant
    >>> sf, f, dur = 100, 1, 4
    >>> N = sf * dur # Total number of discrete samples
    >>> t = np.arange(N) / sf # Time vector
    >>> x = np.sin(2 * np.pi * f * t)
    >>> np.round(ant.spectral_entropy(x, sf, method='fft'), 2)
    0.0

    Spectral entropy of a random signal using Welch's method

    >>> np.random.seed(42)
    >>> x = np.random.rand(3000)
    >>> ant.spectral_entropy(x, sf=100, method='welch')
    6.98004566237139

    Normalized spectral entropy

    >>> ant.spectral_entropy(x, sf=100, method='welch', normalize=True)
    0.9955526198316073

    Normalized spectral entropy of 2D data

    >>> np.random.seed(42)
    >>> x = np.random.normal(size=(4, 3000))
    >>> np.round(ant.spectral_entropy(x, sf=100, normalize=True), 4)
    array([0.9464, 0.9428, 0.9431, 0.9417])

    Fractional Gaussian noise with H = 0.5

    >>> import stochastic.processes.noise as sn
    >>> rng = np.random.default_rng(seed=42)
    >>> x = sn.FractionalGaussianNoise(hurst=0.5, rng=rng).sample(10000)
    >>> print(f"{ant.spectral_entropy(x, sf=100, normalize=True):.4f}")
    0.9505

    Fractional Gaussian noise with H = 0.9

    >>> rng = np.random.default_rng(seed=42)
    >>> x = sn.FractionalGaussianNoise(hurst=0.9, rng=rng).sample(10000)
    >>> print(f"{ant.spectral_entropy(x, sf=100, normalize=True):.4f}")
    0.8477

    Fractional Gaussian noise with H = 0.1

    >>> rng = np.random.default_rng(seed=42)
    >>> x = sn.FractionalGaussianNoise(hurst=0.1, rng=rng).sample(10000)
    >>> print(f"{ant.spectral_entropy(x, sf=100, normalize=True):.4f}")
    0.9248
    """
    x = np.asarray(x)
    # Compute and normalize power spectrum
    if method == "fft":
        _, psd = periodogram(x, sf, axis=axis, nfft=2048)
    elif method == "welch":
        _, psd = welch(x, sf, nperseg=nperseg, axis=axis)
    psd_norm = psd / psd.sum(axis=axis, keepdims=True)
    # psd_norm = psd_norm[1024:10240]
    se = -_xlogx(psd_norm).sum(axis=axis)
    if normalize:
        se /= np.log2(psd_norm.shape[axis])
    return se

def _xlogx(x, base=2):
    """Returns x log_b x if x is positive, 0 if x == 0, and np.nan
    otherwise. This handles the case when the power spectrum density
    takes any zero value.
    """
    x = np.asarray(x)
    xlogx = np.zeros(x.shape)
    xlogx[x < 0] = np.nan
    valid = x > 0
    xlogx[valid] = x[valid] * np.log(x[valid]) / np.log(base)
    return xlogx

def smooth(y, box_pts=5):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def plot_wavelet(time, signal, scales, 
                 waveletname = 'cgau5', 
                 cmap = plt.cm.seismic, 
                 title = 'Wavelet Transform (Power Spectrum) of signal', 
                 ylabel = 'Frequency', 
                 xlabel = 'Time'):
    
    dt = time[1] - time[0]
    [coefficients, frequencies] = pywt.cwt(signal, scales, waveletname, dt)
    power = (abs(coefficients)) ** 2
    period = frequencies
    levels = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8]
    contourlevels = np.log2(levels)
    
    fig, ax = plt.subplots(figsize=(15, 10))
    im = ax.contourf(time, np.log2(period), np.log2(power), contourlevels, extend='both',cmap=cmap)
    
    ax.set_title(title, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.set_xlabel(xlabel, fontsize=18)
    
    yticks = 2**np.arange(np.ceil(np.log2(period.min())), np.ceil(np.log2(period.max())))
    ax.set_yticks(np.log2(yticks))
    ax.set_yticklabels(yticks)
    # ax.invert_yaxis()
    # ylim = ax.get_ylim()
    # ax.set_ylim(ylim[0], -1)
    # plt.show()
    
    cbar_ax = fig.add_axes([0.95, 0.5, 0.03, 0.25])
    fig.colorbar(im, cax=cbar_ax, orientation="vertical")
    plt.show()