"""
Power spectral utilisation functions
"""

import pandas as pd
from scipy import signal
import numpy as np
import sys
import pickle

import mne
from mne_connectivity import spectral_connectivity_epochs

# inserting the lib folder to the compiler
sys.path.insert(0, './lib')
sys.path.insert(0, './utils/')

import utils_io

def add_onset_aligned_recording(df, fs=2048):
    
    target_len = fs * 2  # 2 seconds

    def make_onset_window(row):
        pre_event = np.array(row['pre_event_recording'])
        event     = np.array(row['event_recording'])

        # truncate if too long
        if(len(pre_event) > target_len) : pre_event = pre_event[:target_len]
        if(len(event) > target_len)     : event = event[:target_len]

        # merge
        onset_window = np.concatenate([pre_event, event])
        return onset_window

    df = df.copy()
    df['onset_aligned_recording'] = df.apply(make_onset_window, axis=1)
    return df


def add_offset_aligned_recording(df, fs=2048):

    target_len = fs * 2  # 2 seconds

    def make_offset_window(row):
        event      = np.array(row['event_recording'])
        post_event = np.array(row['post_event_recording'])

        # truncate if too long
        if(len(event) > target_len)      : event = event[-target_len:]  # take last samples
        if(len(post_event) > target_len) : post_event = post_event[:target_len]  # take first samples

        # merge
        offset_window = np.concatenate([event, post_event])
        return offset_window

    df = df.copy()
    df['offset_aligned_recording'] = df.apply(make_offset_window, axis=1)
    return df

def truncate_pre_event(lfp_df, ecog_df):
   
    lfp_df  = lfp_df.copy()
    ecog_df = ecog_df.copy()

    # find the shortest pre-event segment length both in STN and cortex
    all_lengths = pd.concat([lfp_df['pre_event_recording'].apply(len), 
                             ecog_df['pre_event_recording'].apply(len)])
    min_len     = int(all_lengths.min())

    # truncate each recording (keep the end)
    lfp_df['pre_event_recording']  = lfp_df['pre_event_recording'].apply(lambda x: x[-min_len:])
    ecog_df['pre_event_recording'] = ecog_df['pre_event_recording'].apply(lambda x: x[-min_len:])

    return lfp_df, ecog_df

def truncate_event(lfp_df, ecog_df):

    # find the shortest event segment length both in STN and cortex
    min_len      = min(lfp_df['event_recording'].apply(len).min(), ecog_df['event_recording'].apply(len).min())

    # truncate each recording (keep the beginning)
    lfp_df     = lfp_df.copy()
    ecog_df    = ecog_df.copy()
    lfp_df['event_recording']  = lfp_df['event_recording'].apply(lambda x: x[:min_len])
    ecog_df['event_recording'] = ecog_df['event_recording'].apply(lambda x: x[:min_len])

    return lfp_df, ecog_df

def truncate_post_event(lfp_df, ecog_df):

    # find the shortest post-event segment length both in STN and cortex
    min_len   = min(lfp_df['post_event_recording'].apply(len).min(), ecog_df['post_event_recording'].apply(len).min())

    # truncate each recording (keep the beginning)
    lfp_df  = lfp_df.copy()
    ecog_df = ecog_df.copy()
    lfp_df['post_event_recording']  = lfp_df['post_event_recording'].apply(lambda x: x[:min_len])
    ecog_df['post_event_recording'] = ecog_df['post_event_recording'].apply(lambda x: x[:min_len])

    return lfp_df, ecog_df

def check_recording_lengths(lfp_df, ecog_df, segment):
    lfp_lengths  = lfp_df[f"{segment}_recording"].apply(len)
    ecog_lengths = ecog_df[f"{segment}_recording"].apply(len)
    all_lengths  = pd.concat([lfp_lengths, ecog_lengths])  # we expect all the segment lengths across patients to be the same, 
    return all_lengths.nunique() == 1

def check_missing_recordings(lfp_df, ecog_df, segment):
    lfp_missing_rows  = lfp_df[lfp_df[f"{segment}_recording"].apply(lambda x: x is None or len(x)==0)]
    ecog_missing_rows = ecog_df[ecog_df[f"{segment}_recording"].apply(lambda x: x is None or len(x)==0)]
    print(f">>> >>> LFP missing recordings: {len(lfp_missing_rows)}")
    print(f">>> >>> ECoG missing recordings: {len(ecog_missing_rows)}")
    return lfp_missing_rows, ecog_missing_rows
    
def truncate_segment(lfp_df, ecog_df, segment):

    # remove missing recordings, if exists
    lfp_df  = lfp_df[lfp_df[f"{segment}_recording"].apply(lambda x: x is not None and len(x) > 0)].reset_index(drop=True)
    ecog_df = ecog_df[ecog_df[f"{segment}_recording"].apply(lambda x: x is not None and len(x) > 0)].reset_index(drop=True)

    if(segment=="pre_event"):
        lfp_df, ecog_df       = truncate_pre_event(lfp_df, ecog_df)
        pre_event_valitation  = check_recording_lengths(lfp_df=lfp_df, ecog_df=ecog_df, segment=segment)
        if(pre_event_valitation==False) : check_missing_recordings(lfp_df=lfp_df, ecog_df=ecog_df, segment=segment)
        print(f">>> pre-event recordings have consistent length  : {pre_event_valitation}")
        
    elif(segment=="event"):      
        lfp_df, ecog_df       = truncate_event(lfp_df, ecog_df)
        event_valitation      = check_recording_lengths(lfp_df=lfp_df, ecog_df=ecog_df, segment=segment)
        if(event_valitation==False) : check_missing_recordings(lfp_df=lfp_df, ecog_df=ecog_df, segment=segment)
        print(f">>> event recordings have consistent length      : {event_valitation}")

    elif(segment=="post_event"): 
        lfp_df, ecog_df       = truncate_post_event(lfp_df, ecog_df)
        post_event_valitation = check_recording_lengths(lfp_df=lfp_df, ecog_df=ecog_df, segment="post_event")
        if(post_event_valitation==False) : check_missing_recordings(lfp_df=lfp_df, ecog_df=ecog_df, segment=segment)
        print(f">>> post event recordings have consistent length : {post_event_valitation}")
        
    return lfp_df, ecog_df

def keep_complete_events(dataset):

    # keep events only when we have recordings in all channels, if one event lacks recording for some channels, we will remove this 
    # event from connectivity analysis
    
    if("LFP_channel" in dataset.columns)    : channel_column = "LFP_channel"
    elif("ECoG_channel" in dataset.columns) : channel_column = "ECoG_channel"
    
    all_channels    = set(dataset[channel_column].unique())
    complete_events = []

    for event_no, group in dataset.groupby("event_no"):
        if set(group[channel_column].unique()) == all_channels:
            complete_events.append(event_no)

    dataset_filtered = dataset[dataset["event_no"].isin(complete_events)].reset_index(drop=True)
    return dataset_filtered

def create_data_vector_GC_estimation(lfp_df, ecog_df, segment):
    """
    This method creates the data matrix with shape (n_trials, n_channels, n_times) independent of the directionality of connectivity analysis.
    We basically stack all the taps across all the LFP(first) and ECoG(second) channels in this data matrix. The directionality will be handled with
    the connectivity matrix.
    """
    
    # common unique tap events between STN and cortex (in case of artifact removal in one recording type)
    events      = sorted(set(lfp_df['event_no'].unique()).intersection(set(ecog_df['event_no'].unique()))) # always the same order for event_no with sorting
    data_vector = []  # shape: (n_trials, n_channels, n_times)
    
    for event_no in events:
        lfp_rows           = lfp_df[lfp_df['event_no'] == event_no]
        ecog_rows          = ecog_df[ecog_df['event_no'] == event_no]
        segment_recordings = [] # (n_channels, n_times)
        
        # LFP channels
        for arr in lfp_rows[segment+"_recording"]:
            segment_recordings.append(np.array(arr))
        
        # ECoG channels
        for arr in ecog_rows[segment+"_recording"]:
            segment_recordings.append(np.array(arr))
        
        segment_recordings = np.vstack(segment_recordings)
        data_vector.append(segment_recordings)
    
    # final array: (n_trials, n_channels, n_times)
    data_vector = np.stack(data_vector, axis=0)

    # just in case any recordings contained np.nan
    data_vector = np.nan_to_num(data_vector, nan=0)
    
    return data_vector


def create_connectivity_indices(lfp_df, ecog_df, direction):
    
    # channel names and counts / just in case sort the channel names in ascending order
    lfp_names        = list(lfp_df.sort_values("LFP_channel")["LFP_channel"].unique())
    ecog_names       = list(ecog_df.sort_values("ECoG_channel")["ECoG_channel"].unique())
    n_lfp            = len(lfp_names)
    n_ecog           = len(ecog_names)

    # hemisphere information (assuming one hemisphere per recording type)
    assert len(lfp_df.LFP_hemisphere.unique())   == 1, "multiple LFP hemispheres in input!"
    assert len(ecog_df.ECoG_hemisphere.unique()) == 1, "multiple ECoG hemispheres in input!"

    lfp_hemisphere   = lfp_df.LFP_hemisphere.unique()[0]
    ecog_hemisphere  = ecog_df.ECoG_hemisphere.unique()[0]

    # Channel indices in stacked data_vector
    lfp_chs          = list(range(n_lfp))
    ecog_chs         = list(range(n_lfp, n_lfp + n_ecog))
    
    sources, targets = [], []
    rows             = []

    if direction not in ("lfp->ecog", "ecog->lfp", "both"): raise ValueError(f"invalid direction: {direction}")

    if direction in ("lfp->ecog", "both"):
        for lfp_idx, lfp_channel in zip(lfp_chs, lfp_names):
            for ecog_idx, ecog_channel in zip(ecog_chs, ecog_names):
                sources.append([lfp_idx])
                targets.append([ecog_idx])
                rows.append({"source_type": "LFP" , "source_hemisphere": lfp_hemisphere , "source_channel": lfp_channel , "source_index": lfp_idx, 
                             "target_type": "ECOG", "target_hemisphere": ecog_hemisphere, "target_channel": ecog_channel, "target_index": ecog_idx})

    if direction in ("ecog->lfp", "both"):
        for ecog_idx, ecog_channel in zip(ecog_chs, ecog_names):
            for lfp_idx, lfp_channel in zip(lfp_chs, lfp_names):
                sources.append([ecog_idx])
                targets.append([lfp_idx])
                rows.append({"source_type": "ECOG", "source_hemisphere": ecog_hemisphere, "source_channel": ecog_channel, "source_index": ecog_idx,
                             "target_type": "LFP" , "target_hemisphere": lfp_hemisphere , "target_channel": lfp_channel , "target_index": lfp_idx})
    
    df_connections = pd.DataFrame(rows)
    return (sources, targets), df_connections

def create_data_structures_for_connectivity_analyses(lfp_df, ecog_df, state, direction, patient, segment, connectivity_side):

    # get the tap activity of a particular patient in a particular clinical state
    lfp_df   = lfp_df[(lfp_df.severity==state) & (lfp_df.patient==patient)].copy()
    ecog_df  = ecog_df[(ecog_df.severity==state) & (ecog_df.patient==patient)].copy()
    
    if(lfp_df.empty or ecog_df.empty): 
        print(f">>> >>> >>> Patient {patient} does not have LFP or ECOG recordings for {state} state to estimate connectivity...")
        return np.nan, np.nan, np.nan
        
    hemisphere_ECOG     = ecog_df.ECoG_hemisphere.unique()[0] # since we have single hemisphere for ECOG recordings

    # based on the ipsilateral or contralateral connectivity (between the STN and the cortex), select LFP taps
    if(connectivity_side=="ipsilateral") : lfp_df = lfp_df[lfp_df.LFP_hemisphere==hemisphere_ECOG]
    else                                 : lfp_df = lfp_df[lfp_df.LFP_hemisphere!=hemisphere_ECOG]
        
    # if an event does not have recordings across all possible channels, remove this event from the connectivity analysis
    # With this approach, we make sure that stacking the recordings is possible without getting an inconsistent shape error
    lfp_df  = keep_complete_events(lfp_df)
    ecog_df = keep_complete_events(ecog_df)

    # in case ECOG recordings do not have ipsilateral/contralateral recordings, abort the process
    if(len(lfp_df)==0): 
        print(f">>> >>> >>> Patient {patient} does not have {connectivity_side} side LFP recordings with ECOG for {state} state...")
        return np.nan, np.nan, np.nan

    else:
        # truncate the segments to the shortest recording observed for patients (each event segment individually)
        # to get consistent length across tapping trials (mne does not support np.nan in its computation and 
        # 0 padding affects connectivity measurements, particularly at lower frequencies.
        
        lfp_df, ecog_df  = truncate_segment(lfp_df=lfp_df, ecog_df=ecog_df, segment=segment)
        lfp_df           = lfp_df.sort_values("LFP_channel").reset_index(drop=True)
        ecog_df          = ecog_df.sort_values("ECoG_channel").reset_index(drop=True)
        
        # LFP and ECOG channel counts 
        n_LFP_channels   = lfp_df.LFP_channel.nunique() 
        n_ECOG_channels  = ecog_df.ECoG_channel.nunique()
        
        # create a data vector with shape (n_taps, n_channels, n_recording_length)
        data_vector                                   = create_data_vector_GC_estimation(lfp_df=lfp_df, ecog_df=ecog_df, segment=segment) 
        # create a channel connectivity matrix
        connectivity_indices, df_channel_connectivity = create_connectivity_indices(lfp_df=lfp_df, ecog_df=ecog_df, direction=direction)
        df_channel_connectivity["patient"]            = patient
        df_channel_connectivity["state"]              = state
        df_channel_connectivity["segment"]            = segment
        return data_vector, connectivity_indices, df_channel_connectivity

def measure_granger_causality_between_STN_and_cortex(dataset_LFP, dataset_ECOG, state, direction, patient, segment, connectivity_side, fs, n_iteration, verbose=False):

    if(verbose==True):
        print("-------------------------------------------------------------------------------------")
        print("--------------------------------- GRANGER CAUSALITY ---------------------------------")
        print("-------------------------------------------------------------------------------------")
        print(f"clinical state         : {state}")
        print(f"direction              : {direction}")
        print(f"patient                : {patient}")
        print(f"event segment          : {segment}")
        print(f"STN-cortex positioning : {connectivity_side}")
    
    data_vector, conn_indices, df_connectivity = create_data_structures_for_connectivity_analyses(lfp_df            = dataset_LFP, 
                                                                                                  ecog_df           = dataset_ECOG, 
                                                                                                  state             = state, 
                                                                                                  direction         = direction, 
                                                                                                  patient           = patient, 
                                                                                                  segment           = segment, 
                                                                                                  connectivity_side = connectivity_side)
    
    if isinstance(conn_indices, float) and np.isnan(conn_indices):
        return np.nan # then there is no data in one of the recording modality for the selected configuration
        
    else:
        try:
            con                              = spectral_connectivity_epochs(data_vector, method='gc_tr', indices=conn_indices, mode='multitaper',
                                                                            sfreq=fs, fmin=1, fmax=100, faverage=False, mt_adaptive=True, mt_low_bias=True, verbose=verbose)
        except RuntimeError as e:
            print(f">>> >>> >>> time reversed granger causality estimation failed for the channel, skipping...")
            return np.nan
        
        # extract Granger causality 
        freqs                            = np.array(con.freqs)
        gc_data                          = con.get_data()  # shape: (n_channels, n_freqs)
        
        theta_mean                       = np.nanmean(gc_data[:, (freqs >= 4) & (freqs <= 8)], axis=1)
        alpha_mean                       = np.nanmean(gc_data[:, (freqs >= 8) & (freqs <= 12)], axis=1)
        beta_low_mean                    = np.nanmean(gc_data[:, (freqs >= 12) & (freqs <= 20)], axis=1)
        beta_high_mean                   = np.nanmean(gc_data[:, (freqs >= 20) & (freqs <= 25)], axis=1)
        gamma_mean                       = np.nanmean(gc_data[:, (freqs >= 60) & (freqs <= 90)], axis=1)
        gamma_III_mean                   = np.nanmean(gc_data[:, (freqs >= 80) & (freqs <= 90)], axis=1)

        df_connectivity["direction"]     = direction
        df_connectivity['gc_theta']      = theta_mean
        df_connectivity['gc_alpha']      = alpha_mean
        df_connectivity['gc_beta_low']   = beta_low_mean
        df_connectivity['gc_beta_high']  = beta_high_mean
        df_connectivity['gc_gamma']      = gamma_mean
        df_connectivity['gc_gamma_III']  = gamma_III_mean

        ## check the significance of measured state-dependent Granger causality in each frequency band
        null_distributions               = bootstraping(data_vector, conn_indices, fs=fs, n_iteration=n_iteration, fmin=1, fmax=100, mode='multitaper')
        df_connectivity                  = measure_pvalues_for_GC(df_connectivity, null_distributions)
        
        return df_connectivity

def remove_invalid_segments(df, segment_length, segment_column="event_recording"):
    # remove rows where: the segment contains only NaNs and the segment length is less than the expected segment length
    mask_valid = df[segment_column].apply(lambda x: (len(x) == segment_length) and (not np.all(np.isnan(x))))
    return df[mask_valid].reset_index(drop=True)


def measure_pvalues_for_GC(observed_gc, null_distributions):

    bands = ["theta", "alpha", "beta_low", "beta_high", "gamma"]
    
    for band in bands:
        obs_values  = observed_gc[f"gc_{band}"].values                     # shape: (n_connections,)
        null_values = null_distributions[f"gc_{band}"]                     # shape: (n_iterations, n_connections)
        pvals       = np.mean(null_values >= obs_values[None, :], axis=0)  #  connection-wise p-value -> shape: (n_connections,)
        observed_gc[f"pvalue_{band}"] = pvals

    return observed_gc
    
def bootstraping(data_vector, conn_indices, fs, n_iteration=500, fmin=1, fmax=100, mode='multitaper', verbose=True):

    # bootstrapping with trial and time-point shuffling without touching the channel pairing
    
    n_trials, n_channels, n_times = data_vector.shape
    sources, targets              = conn_indices
    unique_src_idx                = sorted(set([s[0] for s in sources]))
    
    bands                         = ["gc_theta","gc_alpha","gc_beta_low","gc_beta_high","gc_gamma"]
    null_distributions            = {band: [] for band in bands}
    freqs_out                     = None

    if verbose:
        print(f">>> >>> bootstrap testing with {n_iteration} iterations")
    
    for iter in range(n_iteration):
        
        surrogate = data_vector.copy()
        
        if(verbose and (iter+1)%100==0): print(f">>> >>> >>> {iter+1} iteration completed...")

        # trial shuffle within channel pair
        for ch in unique_src_idx:
            perm                = np.random.permutation(n_trials)
            surrogate[:, ch, :] = surrogate[perm, ch, :]

        # time-point shuffle of recordings
        for trial in range(n_trials):
            for ch in range(n_channels):
                surrogate[trial, ch, :] = np.random.permutation(surrogate[trial, ch, :])

        # compute GC
        con       = spectral_connectivity_epochs(surrogate, method='gc_tr', indices=conn_indices, mode=mode, sfreq=fs, fmin=fmin, fmax=fmax, 
                                                 mt_adaptive=True, mt_low_bias=True, faverage=False, verbose=False)
        freqs     = np.array(con.freqs)
        gc_data   = con.get_data()  # shape: (n_connections, n_freqs)

        # band averages
        theta     = np.nanmean(gc_data[:, (freqs >= 4) & (freqs <= 8)], axis=1)
        alpha     = np.nanmean(gc_data[:, (freqs >= 8) & (freqs <= 12)], axis=1)
        beta_low  = np.nanmean(gc_data[:, (freqs >= 12) & (freqs <= 20)], axis=1)
        beta_high = np.nanmean(gc_data[:, (freqs >= 20) & (freqs <= 25)], axis=1)
        gamma     = np.nanmean(gc_data[:, (freqs >= 60) & (freqs <= 90)], axis=1)

        # append each iteration as a new row
        null_distributions["gc_theta"].append(theta)
        null_distributions["gc_alpha"].append(alpha)
        null_distributions["gc_beta_low"].append(beta_low)
        null_distributions["gc_beta_high"].append(beta_high)
        null_distributions["gc_gamma"].append(gamma)

    # convert lists of arrays -> 2D arrays (n_iterations, n_connections)
    for band in bands:
        null_distributions[band] = np.stack(null_distributions[band], axis=0)

    return null_distributions
    
def measure_baseline_granger_causality_between_STN_and_cortex(connectivity_side, fs):
    
    gc_rest       = []
    BASELINE_ECOG = utils_io.load_baseline_recordings(recording_type="ECoG")
    BASELINE_LFP  = utils_io.load_baseline_recordings(recording_type="LFP")
    
    for patient in BASELINE_ECOG.keys():
    
        patient_ecog = BASELINE_ECOG[patient].copy()
        patient_lfp  = BASELINE_LFP[patient].copy()
    
        ############################################################################################
        # STEP 1: data preprocessing (length check, nan removal etc)
        ############################################################################################
        
        baseline_lengths = []
        
        for hemisphere in patient_ecog.keys():
            for channel in patient_ecog[hemisphere].keys():
                baseline_lengths.append(len(patient_ecog[hemisphere][channel]))
                
        for hemisphere in patient_lfp.keys():
            for channel in patient_lfp[hemisphere].keys():
                baseline_lengths.append(len(patient_lfp[hemisphere][channel]))
    
        # In this case, some baseline recording lengths are different, and trim the minimum length
        if len(set(baseline_lengths)) != 1:
            
            min_length = min(baseline_lengths)
            
            for hemisphere in patient_ecog.keys():
                for channel in patient_ecog[hemisphere].keys():
                    patient_ecog[hemisphere][channel] = patient_ecog[hemisphere][channel][:min_length]
                
            for hemisphere in patient_lfp.keys():
                for channel in patient_lfp[hemisphere].keys():
                    patient_lfp[hemisphere][channel] = patient_lfp[hemisphere][channel][:min_length]
    
            baseline_length = min_length
    
        ############################################################################################
        # STEP 2: segmentation
        ############################################################################################
        
        # start segmentation of baseline recordings with 2 seconds epochs
        baseline_length  = list(set(baseline_lengths))[0]
        segment_duration = 2  # seconds
        segment_length   = segment_duration * fs
        n_segments       = (baseline_length // segment_length) # number of segments
        start_indices    = np.arange(0, n_segments * segment_length, segment_length).astype(int) # start indices for non-overlapping segments
    
        # Segment ECOG recordings
        ecog_rows = []
        for hemisphere in patient_ecog:
            for channel in patient_ecog[hemisphere]:
                data = patient_ecog[hemisphere][channel]
                for event_no, start in enumerate(start_indices):
                    segment = data[start:start + segment_length]
                    ecog_rows.append({"patient": patient, "event_no": f"rest_s{event_no}", "ECoG_hemisphere": hemisphere, 
                                      "ECoG_channel": channel, "event_recording": segment, "severity": "REST"})
    
        # Segment LFP recordings
        lfp_rows  = []
        for hemisphere in patient_lfp:
            for channel in patient_lfp[hemisphere]:
                data = patient_lfp[hemisphere][channel]
                for event_no, start in enumerate(start_indices):
                    segment = data[start:start + segment_length]
                    lfp_rows.append({"patient": patient, "event_no": f"rest_s{event_no}", "LFP_hemisphere": hemisphere,
                                     "LFP_channel": channel, "event_recording": segment, "severity": "REST"})
    
        # After looping through all patients, create DataFrames
        ecog_df = pd.DataFrame(ecog_rows)
        lfp_df  = pd.DataFrame(lfp_rows)
    
        # in cases where some section of baseline recordings are np.nan, we will remove them. Since we code all segments with event_no and 
        # measure_granger_causality_between_STN_and_cortex uses event_no (it expects to see the event across all patient channels, otherwise
        # it will exclude this event/segment from the analysis, we won't have a synchronicity problem
        lfp_df  = remove_invalid_segments(lfp_df, segment_length=segment_length)
        ecog_df = remove_invalid_segments(ecog_df, segment_length=segment_length)
    
        for direction in ["lfp->ecog", "ecog->lfp"]:
            print(f"processing patient {patient} | {direction} direction | REST state ...")
            patient_gc = measure_granger_causality_between_STN_and_cortex(dataset_LFP=lfp_df, dataset_ECOG=ecog_df, state="REST", direction=direction,
                                                                          patient=patient, segment="event", connectivity_side=connectivity_side, fs=fs)
            gc_rest.append(patient_gc)
    
    gc_rest = pd.concat(gc_rest, ignore_index=True)
    return gc_rest


def normalize_gc_by_baseline(taps_gc, baseline_gc):

    norm_gc     = taps_gc.copy()
    
    for i, row_tap in norm_gc.iterrows():
        
        # select the baseline GC values for the selected row (same patient and connectivity pattern)
        row_baseline = baseline_gc[(baseline_gc.patient==row_tap["patient"]) & (baseline_gc.direction==row_tap["direction"]) & 
                                   (baseline_gc.source_type==row_tap["source_type"]) & (baseline_gc.source_hemisphere==row_tap["source_hemisphere"]) & (baseline_gc.source_channel==row_tap["source_channel"]) & 
                                   (baseline_gc.target_type==row_tap["target_type"]) & (baseline_gc.target_hemisphere==row_tap["target_hemisphere"]) & (baseline_gc.target_channel==row_tap["target_channel"])]

        if(row_baseline.empty):
            norm_gc.at[i, "gc_theta"]     = np.nan
            norm_gc.at[i, "gc_alpha"]     = np.nan
            norm_gc.at[i, "gc_beta_low"]  = np.nan
            norm_gc.at[i, "gc_beta_high"] = np.nan
            norm_gc.at[i, "gc_gamma"]     = np.nan
            #norm_gc.at[i, "gc_gamma_III"] = np.nan
        else:
            assert len(row_baseline)      == 1, f"multiple baseline matches found for row {i}"
            
            row_baseline                  = row_baseline.iloc[0].to_dict() # turn baseline row into dict for easier mathematical operations
            norm_gc.at[i, "gc_theta"]     = (row_tap["gc_theta"] - row_baseline["gc_theta"]) / row_baseline["gc_theta"] * 100
            norm_gc.at[i, "gc_alpha"]     = (row_tap["gc_alpha"] - row_baseline["gc_alpha"]) / row_baseline["gc_alpha"] * 100
            norm_gc.at[i, "gc_beta_low"]  = (row_tap["gc_beta_low"] - row_baseline["gc_beta_low"]) / row_baseline["gc_beta_low"] * 100
            norm_gc.at[i, "gc_beta_high"] = (row_tap["gc_beta_high"] - row_baseline["gc_beta_high"]) / row_baseline["gc_beta_high"] * 100
            norm_gc.at[i, "gc_gamma"]     = (row_tap["gc_gamma"] - row_baseline["gc_gamma"]) / row_baseline["gc_gamma"] * 100
            #norm_gc.at[i, "gc_gamma_III"] = (row_tap["gc_gamma_III"] - row_baseline["gc_gamma_III"] ) / row_baseline["gc_gamma_III"] * 100
    
    norm_gc = norm_gc.drop(columns=["gc_gamma_III"])
    
    return norm_gc