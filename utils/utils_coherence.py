"""
Power spectral utilisation functions
"""

import pandas as pd
from scipy import signal
import numpy as np
import sys
import pickle
from scipy.ndimage import gaussian_filter1d

# inserting the lib folder to the compiler
sys.path.insert(0, './lib')
sys.path.insert(0, './utils/')

import utils_io, utils_psd, utils_misc

from lib_ECoG import ECoG
from lib_LFP import LFP
from lib_data import DATA_IO

fs = 2048
    
def measure_absolute_coherence(x, y):
    
    wlength  = int(fs/4) # 250 ms window length
    noverlap = wlength/2 # 50% overlap
    f, Cxy   = signal.coherence(x, y, nperseg=wlength, noverlap=noverlap, nfft=fs, fs=fs)
    Cxy      = Cxy[(f>=4) & (f<=100)]
    f        = f[(f>=4) & (f<=100)]
    
    return f, Cxy

def extract_baseline_coherence_between_ECOG_LFP_channels():

    print("ECOG...")
    BASELINE_ECOG               = utils_io.load_baseline_recordings(recording_type="ECoG")
    print("LFP...")
    BASELINE_LFP                = utils_io.load_baseline_recordings(recording_type="LFP")
    baseline_ECOG_LFP_coherence = pd.DataFrame(columns=["patient", "ECOG_hemisphere", "LFP_hemisphere", "ECOG_channel", "LFP_channel", "coherence"])
    
    for patient in BASELINE_ECOG.keys():
        
        for h_ECOG in ["right", "left"]:
            for h_LFP in ["right", "left"]:

                try:
                    channels_ECOG = BASELINE_ECOG[patient][h_ECOG].keys()
                except:
                    channels_ECOG = []

                try:
                    channels_LFP  = BASELINE_LFP[patient][h_LFP].keys()
                except:
                    channels_LFP  = []
   
                
                for c_ECOG in channels_ECOG:
                    for c_LFP in channels_LFP:
        
                        
                        # measure the baseline coherence between two channels 
                        ECOG_recording = BASELINE_ECOG[patient][h_ECOG][c_ECOG]
                        LFP_recording  = BASELINE_LFP[patient][h_LFP][c_LFP]
                        
                        # in case of nan values, replace them with 0
                        ECOG_recording[np.isnan(ECOG_recording)] = 0
                        LFP_recording[np.isnan(LFP_recording)] = 0

                        # f, Cxy = measure_absolute_coherence(ECOG_recording, LFP_recording)

                        ##############################################################################################
                        # get 2-second segments of the baseline recordings and get the average coherence
                        segment_length          = (fs*2)
                        segment_overlap         = segment_length / 2
                        
                        # get the start indices of each segments in baseline recording
                        start_indices           = np.arange(0, len(ECOG_recording) - segment_length + 1, segment_overlap).astype(int)

                        try:
                            # get baseline segments
                            ECOG_recording_segments = np.array([ECOG_recording[i:i+segment_length] for i in start_indices])
                            ECOG_recording_segments = np.array(ECOG_recording_segments).reshape(np.shape(ECOG_recording_segments)[0], 1, np.shape(ECOG_recording_segments)[1])
                            LFP_recording_segments  = np.array([LFP_recording[i:i+segment_length] for i in start_indices])
                            LFP_recording_segments  = np.array(LFP_recording_segments).reshape(np.shape(LFP_recording_segments)[0], 1, np.shape(LFP_recording_segments)[1])
    
                            Cxy_array = []
                            for i in range(len(ECOG_recording_segments)):
                                f, Cxy = measure_absolute_coherence(ECOG_recording_segments[i][0], LFP_recording_segments[i][0])
                                Cxy_array.append(Cxy)
                        

                            Cxy = np.nanmean(Cxy_array, axis=0)
                            ##############################################################################################
            
                            # save to the dataframe
                            row                    = {}
                            row["patient"]         = patient
                            row["ECOG_hemisphere"] = h_ECOG
                            row["LFP_hemisphere"]  = h_LFP
                            row["ECOG_channel"]    = c_ECOG
                            row["LFP_channel"]     = c_LFP
                            row["coherence"]       = Cxy
    
                            if(~all(row["coherence"])): #if coherence array is not completely np.nan
                                baseline_ECOG_LFP_coherence.loc[len(baseline_ECOG_LFP_coherence)] = row

                        except:
                            print(f">>> >>> >>>Issue at {patient} -  [ECOG {h_ECOG}{c_ECOG} & LFP {h_LFP}{c_LFP}] channel baseline average coherence estimation!..")
        print(f">>> Patient {patient} baseline average coherence estimation is completed...")
        
                        
    return baseline_ECOG_LFP_coherence

def measure_LFP_ECOG_channel_pair_coherence(dataset_LFP, dataset_ECOG, baseline_coherence_LFP_ECOG,
                                            patient, hemisphere_LFP, hemisphere_ECOG, channel_ECOG, channel_LFP):
    # select the channel events in ECoG and LFP recordings
    channel_events_LFP  = dataset_LFP[(dataset_LFP.patient==patient) & (dataset_LFP.LFP_hemisphere==hemisphere_LFP) & (dataset_LFP.LFP_channel==channel_LFP)]
    channel_events_ECOG = dataset_ECOG[(dataset_ECOG.patient==patient) & (dataset_ECOG.ECoG_hemisphere==hemisphere_ECOG) & (dataset_ECOG.ECoG_channel==channel_ECOG)]
    
    # update recording column names
    channel_events_LFP['LFP_pre_event_recording']    = channel_events_LFP['pre_event_recording']
    channel_events_LFP['LFP_event_recording']        = channel_events_LFP['event_recording']
    channel_events_LFP['LFP_post_event_recording']   = channel_events_LFP['post_event_recording']
    channel_events_ECOG['ECOG_pre_event_recording']  = channel_events_ECOG['pre_event_recording']
    channel_events_ECOG['ECOG_event_recording']      = channel_events_ECOG['event_recording']
    channel_events_ECOG['ECOG_post_event_recording'] = channel_events_ECOG['post_event_recording']
    
    channel_events_LFP  = channel_events_LFP[['patient', 'event_no', 'event_start_time', 'dyskinesia_arm', 
                                              'LFP_pre_event_recording', 'LFP_event_recording', 'LFP_post_event_recording','LFP_hemisphere',	'LFP_channel', ]]
    channel_events_ECOG = channel_events_ECOG[['patient', 'event_no', 'event_start_time', 'dyskinesia_arm', 
                                               'ECOG_pre_event_recording', 'ECOG_event_recording', 'ECOG_post_event_recording','ECoG_hemisphere', 'ECoG_channel']]
    
    # merge dataframes of ECoG and LFP recordings based on same/unique event number to match LFP and ECoG recordings belonging the same event
    channel_events_LFP_ECOG = pd.merge(left=channel_events_LFP, right=channel_events_ECOG,
                                       how='left',left_on=['patient', 'event_no', 'event_start_time', 'dyskinesia_arm'], 
                                       right_on=['patient', 'event_no', 'event_start_time', 'dyskinesia_arm'])
    channel_events_LFP_ECOG.dropna(inplace=True)
    
    # measure coherence 
    channel_events_LFP_ECOG["pre_event_coherence"]  = ""
    channel_events_LFP_ECOG["event_coherence"]      = ""
    channel_events_LFP_ECOG["post_event_coherence"] = ""
    
    for index, row in channel_events_LFP_ECOG.iterrows():

        try:
            # pre_event coherence
            f, event_coherence = measure_absolute_coherence(row.LFP_pre_event_recording, row.ECOG_pre_event_recording)
            baseline_coherence = baseline_coherence_LFP_ECOG[(baseline_coherence_LFP_ECOG.patient==patient) & 
                                                             (baseline_coherence_LFP_ECOG.ECOG_hemisphere==hemisphere_ECOG) & 
                                                             (baseline_coherence_LFP_ECOG.LFP_hemisphere==hemisphere_LFP) & 
                                                             (baseline_coherence_LFP_ECOG.ECOG_channel==channel_ECOG) & 
                                                             (baseline_coherence_LFP_ECOG.LFP_channel==channel_LFP)].iloc[0].coherence
            event_coherence_normalized = (event_coherence - baseline_coherence) / baseline_coherence * 100
            event_coherence_normalized[(f>35) & (f<60)] = 0
            channel_events_LFP_ECOG.at[index, 'pre_event_coherence'] = event_coherence_normalized    
        except:
            channel_events_LFP_ECOG.at[index, 'pre_event_coherence'] = np.linspace(4,100,97) * np.nan

        try:
    
            f, event_coherence = measure_absolute_coherence(row.LFP_event_recording, row.ECOG_event_recording)
            event_coherence    = event_coherence * (len(row.LFP_event_recording)/(fs*2)) # scaling of coherence based on event duration by 2 seconds
            
            #f, event_coherence = measure_absolute_coherence(row.LFP_event_recording, row.ECOG_event_recording)
            baseline_coherence = baseline_coherence_LFP_ECOG[(baseline_coherence_LFP_ECOG.patient==patient) & 
                                                             (baseline_coherence_LFP_ECOG.ECOG_hemisphere==hemisphere_ECOG) & 
                                                             (baseline_coherence_LFP_ECOG.LFP_hemisphere==hemisphere_LFP) & 
                                                             (baseline_coherence_LFP_ECOG.ECOG_channel==channel_ECOG) & 
                                                             (baseline_coherence_LFP_ECOG.LFP_channel==channel_LFP)].iloc[0].coherence
    
            #event_coherence = gaussian_filter1d(event_coherence, sigma=2)
            event_coherence_normalized = (event_coherence - baseline_coherence) / baseline_coherence * 100
            event_coherence_normalized[(f>35) & (f<60)] = 0
            channel_events_LFP_ECOG.at[index, 'event_coherence'] = event_coherence_normalized
        except:
            channel_events_LFP_ECOG.at[index, 'event_coherence'] = np.linspace(4,100,97) * np.nan

        try:
            # post_event coherence 
            f, event_coherence = measure_absolute_coherence(row.LFP_post_event_recording, row.ECOG_post_event_recording)
            baseline_coherence = baseline_coherence_LFP_ECOG[(baseline_coherence_LFP_ECOG.patient==patient) & 
                                                             (baseline_coherence_LFP_ECOG.ECOG_hemisphere==hemisphere_ECOG) & 
                                                             (baseline_coherence_LFP_ECOG.LFP_hemisphere==hemisphere_LFP) & 
                                                             (baseline_coherence_LFP_ECOG.ECOG_channel==channel_ECOG) & 
                                                             (baseline_coherence_LFP_ECOG.LFP_channel==channel_LFP)].iloc[0].coherence
            event_coherence_normalized = (event_coherence - baseline_coherence) / baseline_coherence * 100
            event_coherence_normalized[(f>35) & (f<60)] = 0
            channel_events_LFP_ECOG.at[index, 'post_event_coherence'] = event_coherence_normalized
        except:
            channel_events_LFP_ECOG.at[index, 'post_event_coherence'] = np.linspace(4,100,97) * np.nan
        
    channel_events_LFP_ECOG = channel_events_LFP_ECOG[['patient', 'event_no', 'event_start_time', 'dyskinesia_arm', 
                                                       'pre_event_coherence', 'event_coherence', 'post_event_coherence',
                                                       'LFP_hemisphere', 'LFP_channel','ECoG_hemisphere', 'ECoG_channel']]
    
    return channel_events_LFP_ECOG

def measure_ECOG_LFP_coherence(dataset_LFP, dataset_ECOG, dataset_baseline_LFP_ECOG_coherence, mode="ipsilateral"):
    
    LFP_ECOG_coherence = pd.DataFrame()
    patients           = np.intersect1d(dataset_ECOG.patient.unique(), dataset_LFP.patient.unique())
    
    print ("ECOG-LFP Channel Coherence Measurement Started...")
    
    for patient in patients:
    
        print("---> Patient: " + patient)
        
        ECOG_hemispheres = dataset_ECOG[dataset_ECOG.patient == patient].ECoG_hemisphere.unique()
        LFP_hemispheres  = dataset_LFP[dataset_LFP.patient == patient].LFP_hemisphere.unique()

        for ECOG_h in ECOG_hemispheres:
            for LFP_h in LFP_hemispheres:
    
                ECOG_channels = dataset_ECOG[(dataset_ECOG.patient == patient) & (dataset_ECOG.ECoG_hemisphere == ECOG_h)].ECoG_channel.unique()
                LFP_channels  = dataset_LFP[(dataset_LFP.patient == patient) & (dataset_LFP.LFP_hemisphere == LFP_h)].LFP_channel.unique()

                if((mode=="ipsilateral") & (ECOG_h==LFP_h)):
                    for ECOG_c in ECOG_channels:
                        for LFP_c in LFP_channels:
        
                            print ("     LFP: " + LFP_h + " hemisphere - " + LFP_c + " | ECOG: " + ECOG_h + " hemisphere - " + ECOG_c)
                            
                            channel_pair_coherence = measure_LFP_ECOG_channel_pair_coherence(dataset_LFP                =dataset_LFP,
                                                                                             dataset_ECOG               =dataset_ECOG,
                                                                                             baseline_coherence_LFP_ECOG=dataset_baseline_LFP_ECOG_coherence,
                                                                                             patient                    =patient,
                                                                                             hemisphere_LFP             =LFP_h, 
                                                                                             hemisphere_ECOG            =ECOG_h,
                                                                                             channel_LFP                =LFP_c,
                                                                                             channel_ECOG               =ECOG_c)
        
                            if(len(LFP_ECOG_coherence) == 0):
                                LFP_ECOG_coherence = channel_pair_coherence
                            else:
                                LFP_ECOG_coherence = pd.concat([LFP_ECOG_coherence, channel_pair_coherence], ignore_index=True)
                                
                elif((mode=="controlateral") & (ECOG_h!=LFP_h)):
                    for ECOG_c in ECOG_channels:
                        for LFP_c in LFP_channels:
        
                            print ("     LFP: " + LFP_h + " hemisphere - " + LFP_c + " | ECOG: " + ECOG_h + " hemisphere - " + ECOG_c)
                            
                            channel_pair_coherence = measure_LFP_ECOG_channel_pair_coherence(dataset_LFP                =dataset_LFP,
                                                                                             dataset_ECOG               =dataset_ECOG,
                                                                                             baseline_coherence_LFP_ECOG=dataset_baseline_LFP_ECOG_coherence,
                                                                                             patient                    =patient,
                                                                                             hemisphere_LFP             =LFP_h, 
                                                                                             hemisphere_ECOG            =ECOG_h,
                                                                                             channel_LFP                =LFP_c,
                                                                                             channel_ECOG               =ECOG_c)
        
                            if(len(LFP_ECOG_coherence) == 0):
                                LFP_ECOG_coherence = channel_pair_coherence
                            else:
                                LFP_ECOG_coherence = pd.concat([LFP_ECOG_coherence, channel_pair_coherence], ignore_index=True)
                            
    LFP_ECOG_coherence = LFP_ECOG_coherence[['patient', 'event_no', 'event_start_time', 'dyskinesia_arm', 'pre_event_coherence', 'event_coherence',
                                             'post_event_coherence', 'LFP_hemisphere', 'LFP_channel', 'ECoG_hemisphere', 'ECoG_channel']]
    
    return LFP_ECOG_coherence

def measure_mean_coherence_by_frequency_band(dataset):

    freq  = np.linspace(4, 100, 97)
    bands = {"theta"     : np.where((freq >= 4) & (freq <= 8))[0],
             "alpha"     : np.where((freq >= 8) & (freq <= 12))[0],
             "beta_low"  : np.where((freq >= 12) & (freq <= 20))[0],
             "beta_high" : np.where((freq >= 20) & (freq <= 35))[0],
             "gamma"     : np.where((freq >= 60) & (freq <= 90))[0],
             "gamma_III" : np.where((freq >= 80) & (freq <= 90))[0]}

    def compute_band_means(row):
        results = {}
        for phase in ["pre_event_coherence", "event_coherence", "post_event_coherence"]:
            values      = np.array(row[phase])
            phase_short = phase.replace("_coherence", "")  # remove "coherence"
            for band, idx in bands.items():
                results[f"{phase_short}_{band}_mean"] = values[idx].mean()
        return pd.Series(results)

    # Apply and merge results
    band_means = dataset.apply(compute_band_means, axis=1)
    return pd.concat([dataset, band_means], axis=1)

