"""
Utilisation function for time/frequency plotting
"""

import numpy as np
import pandas as pd
from mne.time_frequency import tfr_array_morlet, tfr_array_multitaper
from mne.stats import permutation_cluster_test
from mne.stats.cluster_level import _setup_adjacency
import mne

# inserting the lib folder to the compiler
import sys
sys.path.insert(0, './lib')


def measure_normalized_multitaper_spectrograms_for_channels(dataset, fs, baseline_recordings):
    
    df_TF = pd.DataFrame(columns=["patient","hemisphere","channel","severity","onset_aligned_normalized","offset_aligned_normalized"])

    for patient in list(dataset.patient.unique()): # iterate across patients
        pat_hemispheres = list(dataset[dataset.patient==patient].hemisphere.unique())
    
        for hemisphere in pat_hemispheres: # iterate across patients, hemispheres
            pat_channels = list(dataset[(dataset.patient==patient) & (dataset.hemisphere==hemisphere)].channel.unique())
        
            for channel in pat_channels: # iterate across patients, hemispheres, channels
                pat_channel_severity = list(dataset[(dataset.patient==patient) & (dataset.hemisphere==hemisphere) & (dataset.channel==channel)].dyskinesia_arm.unique())
    
                for severity in pat_channel_severity: # iterate across patients, hemispheres, channels, dyskinesia severities
    
                    print("patient " + patient + " - " + hemisphere + " - " + channel + " - " + severity)
                    
                    event_LFPs = dataset[(dataset.patient==patient) & (dataset.hemisphere==hemisphere) & 
                                         (dataset.channel==channel) & (dataset.dyskinesia_arm==severity)]
    
                    try:
                        row               = {}
                        row["patient"]    = patient
                        row["hemisphere"] = hemisphere
                        row["channel"]    = channel
                        
                        onset_aligned  = list(event_LFPs[event_LFPs["recording_onset_aligned"].apply(lambda x: isinstance(x, list))].recording_onset_aligned)
                        onset_aligned  = np.reshape(onset_aligned, (len(onset_aligned), 1, len(onset_aligned[0])))
                        offset_aligned = list(event_LFPs[event_LFPs["recording_offset_aligned"].apply(lambda x: isinstance(x, list))].recording_offset_aligned)
                        offset_aligned = np.reshape(offset_aligned, (len(offset_aligned), 1, len(offset_aligned[0])))
                
                        # get time-frequency representation of onset and offset aligned events
                        onset_spectograms  = tfr_array_multitaper(onset_aligned , sfreq=fs, n_cycles=np.linspace(4,90,87) / 1, 
                                                                  freqs=np.linspace(4,90,87), output='power', verbose=False)
                        offset_spectograms = tfr_array_multitaper(offset_aligned, sfreq=fs, n_cycles=np.linspace(4,90,87) / 1, 
                                                                  freqs=np.linspace(4,90,87), output='power', verbose=False)
                        
                        onset_tfr_average  = np.nanmean(onset_spectograms, axis=0)[0]
                        offset_tfr_average = np.nanmean(offset_spectograms, axis=0)[0]
                
                        # get the time-frequency representation of onset and offset aligned channel baseline
                        channel_baseline   = baseline_recordings[patient][hemisphere][channel]
                        channel_baseline   = channel_baseline[~np.isnan(channel_baseline)] # remove np.nan elements
                        channel_baseline   = np.reshape(channel_baseline, (1,1, len(channel_baseline)))
                        baseline_tfr       = tfr_array_multitaper(channel_baseline , sfreq=fs, n_cycles=np.linspace(4,90,87) / 1, 
                                                                  freqs=np.linspace(4,90,87), output='power', verbose=False)
                        baseline_tfr_avg   = np.mean(baseline_tfr, axis=3)[0][0]
                        baseline_tfr_avg   = baseline_tfr_avg[:, np.newaxis]  # shape (87, 1)
                        
                        # Normalize event spectrogram based on baseline spectrograms
                        onset_tfr_norm     = ((onset_tfr_average - baseline_tfr_avg) /  (baseline_tfr_avg)) * 100
                        offset_tfr_norm    = ((offset_tfr_average - baseline_tfr_avg) / (baseline_tfr_avg)) * 100
                
                        row["onset_aligned_normalized"]   = onset_tfr_norm
                        row["offset_aligned_normalized"]  = offset_tfr_norm
                        row["severity"]                   = severity
                        
                    except:
                        pass
                    
                    df_TF.loc[len(df_TF)]  = row

    return df_TF


def measure_normalized_multitaper_spectrograms_for_taps(dataset, fs, baseline_recordings):
    
    df_TF = pd.DataFrame(columns=["patient","hemisphere","severity","event_no","onset_aligned_normalized","offset_aligned_normalized"])

    for patient in list(dataset.patient.unique()): # iterate across patients
        pat_hemispheres      = list(dataset[dataset.patient==patient].hemisphere.unique())

        print ("Patient " + patient)
        # measure TFR of each channel baseline for the selected patient
        channel_baseline_tfr = measure_electrophysiological_channels_baseline_tfr(dataset, baseline_recordings, patient, fs)

        print ("---> baseline TFR measurement is completed...")
    
        for hemisphere in pat_hemispheres: # iterate across patients, hemispheres
            pat_severity = list(dataset[(dataset.patient==patient) & (dataset.hemisphere==hemisphere)].severity.unique())

            for severity in pat_severity: # iterate across patients, hemispheres, dyskinesia severities
                pat_taps = list(dataset[(dataset.patient==patient) & (dataset.hemisphere==hemisphere) & (dataset.severity==severity)].event_no.unique())
                
                for event_no in pat_taps: # iterate across patients, hemispheres, dyskinesia severities and taps

                    print("------> " + hemisphere + " - " + severity + " - " + event_no)  

                    event_LFPs        = dataset[(dataset.patient==patient) & (dataset.hemisphere==hemisphere) & (dataset.severity==severity) & (dataset.event_no==event_no)]
                    event_LFPs_onset  = event_LFPs[event_LFPs.recording_onset_aligned.notna()]
                    event_LFPs_offset = event_LFPs[event_LFPs.recording_offset_aligned.notna()]
                    
                    ############################################################
                    # ONSET ALIGNED SPECTROGRAMS ###############################
                    ############################################################
                    
                    if(len(event_LFPs_onset)!=0): # if the tapping activity has at least one valid onset aligned signal across channels
                        onset_aligned      = list(event_LFPs_onset[event_LFPs_onset["recording_onset_aligned"].apply(lambda x: isinstance(x, list))].recording_onset_aligned)
                        onset_aligned      = np.reshape(onset_aligned, (len(onset_aligned), 1, len(onset_aligned[0])))
    
                        # get time-frequency representation of onset aligned recordings across all channels for the same tapping event
                        onset_spectograms  = tfr_array_multitaper(onset_aligned , sfreq=fs, n_cycles=np.linspace(4,90,87) / 1, 
                                                                  freqs=np.linspace(4,90,87), output='power', verbose=False)
    
                        # get channel-based normalized spectrograms of the same tapping event
                        onset_spectograms_norm  = []
    
                        for i in range(len(event_LFPs_onset)):
                            
                            channel                = event_LFPs_onset.iloc[i].channel
                            tap_onset_tfr_channel  = onset_spectograms[i]

                            try:
                                onset_tfr_norm         = normalize_event_spectogram_by_channel_baseline(tap_onset_tfr_channel, channel_baseline_tfr[hemisphere][channel], fs)
                                onset_spectograms_norm.append(onset_tfr_norm)
                            except KeyError as e:
                                pass # the baseline of the channel does not have enough time points to get robust TFR

                        onset_spectograms_norm = np.stack(onset_spectograms_norm)
                        # measure the mean spectrogram per tapping event
                        onset_tfr_average      = np.nanmean(onset_spectograms_norm, axis=0)[0]

                    else:
                        onset_tfr_average      = np.nan
                    
                    ################################################################### 
                    # OFFSET ALIGNED SPECTROGRAMS ##################################### 
                    ###################################################################  

                    if(len(event_LFPs_offset)!=0): # if the tapping activity has at least one valid offset aligned signal across channels
                        offset_aligned     = list(event_LFPs_offset[event_LFPs_offset["recording_offset_aligned"].apply(lambda x: isinstance(x, list))].recording_offset_aligned)
                        offset_aligned     = np.reshape(offset_aligned, (len(offset_aligned), 1, len(offset_aligned[0])))
    
                        # get time-frequency representation of offset aligned recordings across all channels for the same tapping event
                        offset_spectograms = tfr_array_multitaper(offset_aligned, sfreq=fs, n_cycles=np.linspace(4,90,87) / 1, 
                                                                  freqs=np.linspace(4,90,87), output='power', verbose=False)
                        
                        # get channel-based normalized spectrograms of the same tapping event
                        offset_spectograms_norm = []
                                    
                        for i in range(len(event_LFPs_offset)):
                            
                            channel                = event_LFPs_offset.iloc[i].channel
                            tap_offset_tfr_channel = offset_spectograms[i]
                            
                            try:
                                offset_tfr_norm        = normalize_event_spectogram_by_channel_baseline(tap_offset_tfr_channel,channel_baseline_tfr[hemisphere][channel], fs)
                                offset_spectograms_norm.append(offset_tfr_norm)
                            except KeyError as e:
                                pass # the baseline of the channel does not have enough time points to get robust TFR
                                
                        offset_spectograms_norm    = np.stack(offset_spectograms_norm)       
                        # measure the mean spectrogram per tapping event
                        offset_tfr_average         = np.nanmean(offset_spectograms_norm, axis=0)[0]
                
                    else:
                        offset_tfr_average         = np.nan
                
        
                    row                              = {}
                    row["patient"]                   = patient
                    row["hemisphere"]                = hemisphere
                    row["severity"]                  = severity
                    row["event_no"]                  = event_no
                    row["onset_aligned_normalized"]  = onset_tfr_norm
                    row["offset_aligned_normalized"] = offset_tfr_norm
                    df_TF.loc[len(df_TF)]            = row

    return df_TF

def downsample_spectrogram(spectrogram, n_bins=40):

    n_freqs, n_time = spectrogram.shape
    bin_size        = n_time / n_bins
    downsampled     = np.zeros((n_freqs, n_bins))
    
    for b in range(n_bins):
        
        # compute start and end indices for this bin
        start = int(round(b * bin_size))
        end   = int(round((b + 1) * bin_size))
        # Ensure valid slice
        if(start >= n_time):
            downsampled[:, b] = np.nan
        else:
            segment           = spectrogram[:, start:end]
            downsampled[:, b] = np.nanmean(segment, axis=1)
    
    return downsampled
    
def measure_normalized_multitaper_spectrograms_for_taps_and_channels(dataset, fs, baseline_recordings):

    # to be stored in  DataFrame
    rows = []
    
    for patient in list(dataset.patient.unique()): # iterate across patients
        pat_hemispheres      = list(dataset[dataset.patient==patient].hemisphere.unique())

        print ("Patient " + patient)
        # measure TFR of each channel baseline for the selected patient
        channel_baseline_tfr = measure_electrophysiological_channels_baseline_tfr(dataset, baseline_recordings, patient, fs)

        print ("---> baseline TFR measurement is completed...")
    
        for hemisphere in pat_hemispheres: # iterate across patients, hemispheres
            pat_severity = list(dataset[(dataset.patient==patient) & (dataset.hemisphere==hemisphere)].severity.unique())

            for severity in pat_severity: # iterate across patients, hemispheres, dyskinesia severities
                pat_taps = list(dataset[(dataset.patient==patient) & (dataset.hemisphere==hemisphere) & (dataset.severity==severity)].event_no.unique())
                
                for event_no in pat_taps: # iterate across patients, hemispheres, dyskinesia severities and taps

                    print("------> " + hemisphere + " - " + severity + " - " + event_no)  

                    event_LFPs        = dataset[(dataset.patient==patient) & (dataset.hemisphere==hemisphere) & (dataset.severity==severity) & (dataset.event_no==event_no)]
                    event_LFPs_onset  = event_LFPs[event_LFPs.recording_onset_aligned.notna()]
                    event_LFPs_offset = event_LFPs[event_LFPs.recording_offset_aligned.notna()]
                    
                    
                    
                    ############################################################
                    # ONSET ALIGNED SPECTROGRAMS ###############################
                    ############################################################
                    
                    if(len(event_LFPs_onset)!=0): # if the tapping activity has at least one valid onset aligned signal across channels
                        onset_aligned      = list(event_LFPs_onset[event_LFPs_onset["recording_onset_aligned"].apply(lambda x: isinstance(x, list))].recording_onset_aligned)
                        onset_aligned      = np.reshape(onset_aligned, (len(onset_aligned), 1, len(onset_aligned[0])))
    
                        # get time-frequency representation of onset aligned recordings across all channels for the same tapping event
                        onset_spectograms  = tfr_array_multitaper(onset_aligned , sfreq=fs, n_cycles=np.linspace(4,90,87) / 1, 
                                                                  freqs=np.linspace(4,90,87), output='power', verbose=False)
    
                        # get channel-based normalized spectrograms of the same tapping event
                        for i in range(len(event_LFPs_onset)):
                            
                            channel                = event_LFPs_onset.iloc[i].channel
                            tap_onset_tfr_channel  = onset_spectograms[i]

                            try:
                                onset_tfr_norm         = normalize_event_spectogram_by_channel_baseline(tap_onset_tfr_channel, channel_baseline_tfr[hemisphere][channel], fs)[0]
                                onset_tfr_norm_down    = downsample_spectrogram(onset_tfr_norm, n_bins=40) # 100 ms

                                assert onset_tfr_norm_down.shape[1] == 40, f"Unexpected shape {onset_tfr_norm_down.shape}"
           
                                rows.append({"patient": patient, "hemisphere": hemisphere, "channel": channel, "severity": severity, "event_no": event_no,
                                             "alignment_type": "onset", "spectrogram": onset_tfr_norm_down})
                                
                            except KeyError as e:
                                pass # the baseline of the channel does not have enough time points to get robust TFR
                    
                    ################################################################### 
                    # OFFSET ALIGNED SPECTROGRAMS ##################################### 
                    ###################################################################  

                    if(len(event_LFPs_offset)!=0): # if the tapping activity has at least one valid offset aligned signal across channels
                        offset_aligned     = list(event_LFPs_offset[event_LFPs_offset["recording_offset_aligned"].apply(lambda x: isinstance(x, list))].recording_offset_aligned)
                        offset_aligned     = np.reshape(offset_aligned, (len(offset_aligned), 1, len(offset_aligned[0])))
    
                        # get time-frequency representation of offset aligned recordings across all channels for the same tapping event
                        offset_spectograms = tfr_array_multitaper(offset_aligned, sfreq=fs, n_cycles=np.linspace(4,90,87) / 1, 
                                                                  freqs=np.linspace(4,90,87), output='power', verbose=False)
                                    
                        for i in range(len(event_LFPs_offset)):
                            
                            channel                = event_LFPs_offset.iloc[i].channel
                            tap_offset_tfr_channel = offset_spectograms[i]
                            
                            try:
                                offset_tfr_norm        = normalize_event_spectogram_by_channel_baseline(tap_offset_tfr_channel,channel_baseline_tfr[hemisphere][channel], fs)[0]
                                offset_tfr_norm_down   = downsample_spectrogram(offset_tfr_norm, n_bins=40) # 100 ms
                                
                                assert offset_tfr_norm_down.shape[1] == 40, f"Unexpected shape {offset_tfr_norm_down.shape}"
                                
                                rows.append({"patient": patient, "hemisphere": hemisphere, "channel": channel, "severity": severity, "event_no": event_no,
                                             "alignment_type": "offset", "spectrogram": offset_tfr_norm_down})
                                
                            except KeyError as e:
                                pass # the baseline of the channel does not have enough time points to get robust TFR

                    
    df_TF = pd.DataFrame(rows)
    
    return df_TF

def measure_electrophysiological_channels_baseline_tfr(dataset, baseline_recordings, patient, fs):

    patient_channel_baseline_tfr = {key:{} for key in dataset[dataset.patient == patient].hemisphere.unique()} 
    
    for hemisphere in patient_channel_baseline_tfr.keys():
        for channel in dataset[(dataset.patient == patient) & (dataset.hemisphere == hemisphere)].channel.unique():

            channel_baseline  = baseline_recordings[patient][hemisphere][channel]
            channel_baseline  = channel_baseline[~np.isnan(channel_baseline)] # remove np.nan elements
            channel_baseline  = np.reshape(channel_baseline, (1,1, len(channel_baseline)))

            try:
                patient_channel_baseline_tfr[hemisphere][channel] = tfr_array_multitaper(channel_baseline , sfreq=fs, n_cycles=np.linspace(4,90,87) / 1, 
                                                                                         freqs=np.linspace(4,90,87), output='power', verbose=False)
            except:
                # there is not enough signal length in the channel to measure robust TFR at baseline period
                print (">> " + hemisphere + " hemisphere : " + channel + " channel is excluded!") 

    return patient_channel_baseline_tfr

def normalize_event_spectogram_by_channel_baseline(events_tfr, baseline_tfr, fs):
    
    # get average value for frequency
    baseline_tfr_avg   = np.mean(baseline_tfr, axis=3)[0][0]
    baseline_tfr_avg   = baseline_tfr_avg[:, np.newaxis]  # shape (87, 1)

    # Normalize event spectrogram based on baseline spectrograms
    events_tfr_norm    = ((events_tfr - baseline_tfr_avg) /  (baseline_tfr_avg)) * 100
    return events_tfr_norm

def get_patient_mean_spectrogram_for_severity(dataset, alignment, severity, random_sampling=False, event_threshold=20):
    
    dataset_severity     = dataset[dataset.severity == severity]
    patient_spectrograms = []


    # without augmentation / original patient and severity-wise mean spectrograms
    for patient, group_df in dataset_severity.groupby("patient"):  
        
        if(alignment=="onset"):
            stacked = np.stack(group_df["onset_aligned_normalized"].values)
        else:
            stacked = np.stack(group_df["offset_aligned_normalized"].values)
                
        mean_spectrogram = stacked.mean(axis=0).squeeze()  # [freq, time]
        patient_spectrograms.append(mean_spectrogram)

    # with augmentation - within-patient bootstrapped sampling
    if(random_sampling==True):

        # setting seed for reproducibility
        np.random.seed(0)  
        
        for patient, group_df in dataset_severity.groupby("patient"):
            
            # Select the aligned spectrograms
            if alignment == "onset":
                spectrograms = group_df["onset_aligned_normalized"].values
            else:
                spectrograms = group_df["offset_aligned_normalized"].values
            
            # stack into an array: [num_events, freq, time]
            stacked  = np.stack(spectrograms)
            n_events = stacked.shape[0]
    
            # If less than 10 events, skip
            if n_events < event_threshold:
                continue
    
            # number of bootstrap samples = floor(n_events / event_threshold)
            n_samples = n_events // event_threshold 
    
            for _ in range(n_samples):
                idx          = np.random.choice(n_events, size=int(event_threshold/2), replace=True)
                sampled_mean = stacked[idx].mean(axis=0).squeeze()  # [time, freq]
                patient_spectrograms.append(sampled_mean)

    return np.stack(patient_spectrograms)

def downsample_time_axis_for_spectrogram(spectrogram, target_time_points=256):
    
    # ensure the input data is in the shape [n_subjects, n_freqs, n_times]
    if len(spectrogram.shape) != 3:
        raise ValueError(f"input data must be a 3D array, but got shape {spectrogram.shape}")
    
    n_subjects, n_freqs, n_times = spectrogram.shape
    
    if target_time_points >= n_times:  # no downsampling needed if the target time points are greater or equal
        return spectrogram  

    # compute the downsampling factor
    factor      = n_times // target_time_points
    # reshape and mean-pool over the time axis
    downsampled = spectrogram[:, :, :factor * target_time_points].reshape(n_subjects, n_freqs, target_time_points, factor).mean(axis=3)
    
    return downsampled

def run_permutation_cluster_with_downsampling(group_A, group_B, target_time_points=256, n_permutations=1000, tail=0, n_jobs=1):

    # check that both groups have the correct shape
    if len(group_A.shape) != 3 or len(group_B.shape) != 3:
        raise ValueError("Both groups must have shape [n_subjects, n_freqs, n_times]")

    # downsample the time axis for both groups (downsampling from fs*4 to selected target_time_points
    group_A_downsampled       = downsample_time_axis_for_spectrogram(group_A, target_time_points)
    group_B_downsampled       = downsample_time_axis_for_spectrogram(group_B, target_time_points)
    
    # get the number of frequency bins and time points after downsampling
    _, n_freqs, n_times       = group_A_downsampled.shape
    
    # compute the adjacency matrix for time-frequency data (2D adjacency)
    adjacency                 = mne.stats.combine_adjacency(n_freqs, n_times)

    # Reshape the data to (n_subjects, time * freq)
    X_A                       = group_A_downsampled.reshape(group_A_downsampled.shape[0], -1)  # [n_subjects, time * freq]
    X_B                       = group_B_downsampled.reshape(group_B_downsampled.shape[0], -1)  # [n_subjects, time * freq]

    X                         = [X_A, X_B] # stack the groups
    T_obs, clusters, pvals, _ = permutation_cluster_test(X, n_permutations=n_permutations, tail=tail, n_jobs=n_jobs, adjacency=adjacency)
    T_obs                     = T_obs.reshape(n_freqs, n_times)

    return T_obs, clusters, pvals

################################################################################################################################################################
################################################################################################################################################################
################################################################################################################################################################
    
def measure_TFR_for_patient_LFP_baseline(patient, baseline, fs):

    # create an empty dataframe
    df_channel_baseline_spectogram = pd.DataFrame(columns=["patient", "LFP_hemisphere", "LFP_channel", "baseline_frequency_mean", "baseline_frequency_std"])
        
    for hemisphere in baseline[patient].keys():
        for channel in baseline[patient][hemisphere].keys():

            # for the selected hemisphere and the channel for patient, get the baseline recordings.
            channel_baseline          = baseline[patient][hemisphere][channel]

            # Since the baseline recordings correspond to 5 minutes, we will divide the baseline into 4 seconds segments with
            # 50% overlap. For each segment, we will estimate the MWT and measure the average  spectrogram for 4 seconds for baseline (N x fs*4).
            # Furthermore, we also take the average of all columns to represent baseline MWT as (M X 1) vector. Then we will use either this vector
            # mean and std values of each frequency in average baseline MWT. So, we will save both, the baseline MWT column, mean, and std values for each
            # frequency.

            segment_length            = (fs*4)
            segment_overlap           = segment_length / 2

            # get the start indices of each segment in baseline recording
            start_indices             = np.arange(0, len(channel_baseline) - segment_length + 1, segment_overlap).astype(int)
            
            # get baseline segments
            channel_baseline_segments = np.array([channel_baseline[i:i+segment_length] for i in start_indices])

            # reshape the data
            channel_baseline_segments = np.array(channel_baseline_segments).reshape(np.shape(channel_baseline_segments)[0], 1, 
                                                                                    np.shape(channel_baseline_segments)[1])
            # measure the spectogram for each segment
            baseline_spectogram       = tfr_array_morlet(channel_baseline_segments, sfreq=fs, n_cycles=5, freqs=np.linspace(4,90,87), output='power', verbose=False) 

            # measure the average spectrogram across all segments (N x fs*4)
            baseline_spectogram       = np.nanmean(baseline_spectogram, axis=0)[0]

            # get the mean value for each frequency (from 4-90Hz)
            baseline_frequency_mean   = np.nanmean(baseline_spectogram, axis=1)

            # get the std value for each frequency (from 4-90Hz)
            baseline_frequency_std    = np.std(baseline_spectogram, axis=1)
                
            row                       = {} 
            row["patient"]            = patient
            row["LFP_hemisphere"]     = hemisphere
            row["LFP_channel"]        = channel
            row["frequency_mean"]     = baseline_frequency_mean
            row["frequency_std"]      = baseline_frequency_std
            
            df_channel_baseline_spectogram.loc[len(df_channel_baseline_spectogram)] = row 
    
    return df_channel_baseline_spectogram


def measure_TFR_for_patient_LFP_channel_baseline(baseline, patient, hemisphere, channel, fs):

    # for the selected hemisphere and the channel for patient, get the baseline recordings.
    channel_baseline          = baseline[patient][hemisphere][channel]

    # since the baseline recordings corresponds to 5 minutes, we will divide the baseline into 4 seconds segments with
    # 50% overalap. And for each segment, we will estimate the MWT and measure and average  spectogram for 4 seconds for baseline (N x fs*4).
    # Furthermore, we also take the average of all columns to represent baseline MWT as (M X 1) vector. Then we will use either this vector
    # mean and std values of each frequency in average baseline MWT. So, we will save both, baseline MWT column, mean and std values for each
    # frequency.

    segment_length            = (fs*4)
    segment_overlap           = segment_length / 2

    # get the start indices of each segments in baseline recording
    start_indices             = np.arange(0, len(channel_baseline) - segment_length + 1, segment_overlap).astype(int)
            
    # get baseline segments
    channel_baseline_segments = np.array([channel_baseline[i:i+segment_length] for i in start_indices])

    # reshape the data
    channel_baseline_segments = np.array(channel_baseline_segments).reshape(np.shape(channel_baseline_segments)[0], 1, 
                                                                                    np.shape(channel_baseline_segments)[1])
    # measure the spectogram for each segment
    baseline_spectogram       = tfr_array_morlet(channel_baseline_segments, sfreq=fs, n_cycles=5, freqs=np.linspace(4,90,87), output='power', verbose=False) 

    # measure the average spectogram across all segments (N x fs*4)
    baseline_spectogram       = np.nanmean(baseline_spectogram, axis=0)[0]

    # get the mean value for each frequency (from 4-90Hz)
    baseline_frequency_mean   = np.nanmean(baseline_spectogram, axis=1)

    # get the std value for each frequency (from 4-90Hz)
    baseline_frequency_std    = np.std(baseline_spectogram, axis=1)

    # reshape mean and std arrays
    baseline_frequency_mean   = baseline_frequency_mean.reshape(len(baseline_frequency_mean), 1)
    baseline_frequency_std    = baseline_frequency_std.reshape(len(baseline_frequency_std), 1)
    
    return baseline_frequency_mean, baseline_frequency_std

def measure_TFR_for_patient_LFP_channel_events(dataset, patient, hemisphere, channel, fs):

    # get the correspondings events of patient with particular hemisphere and LFP channel
    dataset_patient_channel_events = dataset[(dataset.patient==patient) & (dataset.LFP_hemisphere==hemisphere) & (dataset.LFP_channel==channel)].copy()
    dataset_patient_channel_events.reset_index(drop=True, inplace=True)
    
    event_spectograms = []
    
    for index, row in dataset_patient_channel_events.iterrows():
        
        event_recording  = np.array(row.event_recording_onset_alingned)
        event_recording  = event_recording.reshape(1, 1, len(event_recording))
    
        # get morlet spectogram of the event
        event_spectogram = tfr_array_morlet(event_recording, sfreq=fs, n_cycles=5, freqs=np.linspace(4,90,87), output='power', verbose=False)[0]
        # pad the spectogram with np.nan until it reaches to 4 seconds
        event_spectogram = np.pad(event_spectogram, ((0, 0), (0, 0), (0, fs*4 - event_recording.shape[2])), constant_values=np.nan)
    
        event_spectograms.append(event_spectogram[0])

    return event_spectograms


def get_normalized_event_spectogram_by_channel(dataset_events, dataset_baseline, patient, hemisphere, channel, fs):
    
    # get event spectograms in the channel
    channel_event_spectograms         = measure_TFR_for_patient_LFP_channel_events(dataset = dataset_events,
                                                                                   patient = patient,
                                                                                   hemisphere = hemisphere,
                                                                                   channel = channel,
                                                                                   fs = fs)
    
    # get event baseline spectogram of the the channel
    freq_mean, freq_std               = measure_TFR_for_patient_LFP_channel_baseline(baseline = dataset_baseline,
                                                                                     patient = patient,
                                                                                     hemisphere = hemisphere,
                                                                                     channel = channel,
                                                                                     fs = fs)
    
    return channel_event_spectograms, freq_mean, freq_std

def get_LFP_channel_coefficient_of_variation(dataset_events, dataset_baseline, fs, stn_area):

    df_channel_cv = pd.DataFrame()
    
    for patient in dataset_events.patient.unique():
        print("Patient " + patient + " - STN " + stn_area + " area: LFP channel cv measurement started...")
        for hemisphere in dataset_events[dataset_events.patient==patient].LFP_hemisphere.unique():
            for channel in dataset_events[(dataset_events.patient==patient)&(dataset_events.LFP_hemisphere==hemisphere)].LFP_channel.unique():
                
                event_spectograms, baseline_mean, baseline_std = get_normalized_event_spectogram_by_channel(dataset_events = dataset_events,
                                                                                                            dataset_baseline = dataset_baseline, 
                                                                                                            patient=patient, 
                                                                                                            hemisphere=hemisphere, 
                                                                                                            channel=channel, 
                                                                                                            fs=fs)
                
                
                # get average spectogram and then normalize to baseline frequency mean
                event_spectogram_avg  = np.nanmean(event_spectograms, axis=0)
                event_spectogram_norm = (event_spectogram_avg - baseline_mean) / baseline_mean

                # instead of measuring cv between -2 to 2 second around event onset, measure the cv for between -1 to 1 for more robust estimation
                event_spectogram_norm = event_spectogram_norm[:, fs:3*fs]
                
                # get the mean value for each frequency (from 4-90Hz)
                spectogram_freq_mean  = np.nanmean(event_spectogram_norm, axis=1)
                
                # get the std value for each frequency (from 4-90Hz)
                spectogram_freq_std   = np.std(event_spectogram_norm, axis=1)
                
                # reshape mean and std arrays
                channel_cv            = spectogram_freq_std / spectogram_freq_mean
                channel_cv            = channel_cv.reshape(len(channel_cv),1)
                
                df_patient_channel_cv               = pd.DataFrame()
                df_patient_channel_cv["frequency"]  = np.linspace(4,90,87)
                df_patient_channel_cv["patient"]    = patient
                df_patient_channel_cv["hemisphere"] = hemisphere
                df_patient_channel_cv["channel"]    = channel
                df_patient_channel_cv["cv"]         = channel_cv
                df_patient_channel_cv["stn_area"]   = stn_area

                if(len(df_channel_cv)==0):
                    df_channel_cv = df_patient_channel_cv
                else:
                    df_channel_cv = pd.concat([df_channel_cv, df_patient_channel_cv], ignore_index=True)
                print("---> " + hemisphere + " hemisphere - " + channel + " channel is completed.")

    return df_channel_cv
