import os
import pandas as pd
import numpy as np
import sys
import json
from utils.utils_fileManagement import load_class_pickle, mergedData

from lib_data import DATA_IO
        
###############################################################################################################################################################
###############################################################################################################################################################
###############################################################################################################################################################
###############################################################################################################################################################
###############################################################################################################################################################

class EVENTS:
        
    def __init__(self, PATH, SUB, DAT_SOURCE):

        #assert DAT_SOURCE in ['ecog_right', 'ecog_left'], f'Please pass ECoG DAT_SOURCE ({DAT_SOURCE})'

        # setting environmental variables
        self.__PATH         = PATH
        self.__SUB          = SUB
        self.__DAT_SOURCE   = DAT_SOURCE
        
        print("EVENT HISTORY: SUB-" + SUB)
        print("... loading started")

        # use DATA_IO to load data structure
        data_IO             = DATA_IO(PATH, SUB, DAT_SOURCE)
        self.__dat          = data_IO.get_data()

        # populate class fields
        self.fs             = self.__dat.fs
        self.times          = self.__dat.data[:,self.__dat.colnames.index('dopa_time')]
        self.task           = self.__dat.data[:,self.__dat.colnames.index('task')]
        self.left_tap       = self.__dat.data[:,self.__dat.colnames.index('left_tap')]
        self.right_tap      = self.__dat.data[:,self.__dat.colnames.index('right_tap')]
        self.left_move      = self.__dat.data[:,self.__dat.colnames.index('left_move')]
        self.right_move     = self.__dat.data[:,self.__dat.colnames.index('right_move')]
        self.bilateral_move = np.array([a & b for a, b in zip(self.left_move.astype(int).tolist(), self.right_move.astype(int).tolist())])
        self.bilateral_tap  = np.array([a & b for a, b in zip(self.left_tap.astype(int).tolist(), self.right_tap.astype(int).tolist())])
        self.no_move        = self.__dat.data[:,self.__dat.colnames.index('no_move')]
        self.period_rest    = (self.task == "rest").astype(int)
        self.period_free    = (self.task == "free").astype(int)
        self.period_tap     = (self.task == "tap").astype(int)
        
        print("... task periods were defined")
        self.__define_events()
        print("... events were categorized")
        self.get_dyskinetia_scores()
        print("... dyskinesia evaluation was collected")
        print("... loading completed")

    def __operator_event_difference(self, array_A, array_B):
        """
        Description
            This method finds events that occurred in array_A but not in array_B. The two arrays were expected to have the same length

        Input
            :param array_A: A binary list represents the existence of event=1 and absence=0.
            :param array_B: A binary list represents the existence of event=1 and absence=0.

        Output
            :return: A binary list with the same length as provided arrays.
        """
        assert len(array_A) == len(array_A), "Please provide two arrays with the same length."
        return [1 if event_A == 1 and event_B == 0 else 0 for event_A, event_B in zip(array_A, array_B)]
    
    def __operator_event_intersection(self, array_A, array_B):
        """
        Description
            This method finds events that occurred both in array_A and array_B (and operator). The two arrays were expected to have the same length

        Input
            :param array_A: A binary list represents the existence of event=1 and absence=0.
            :param array_B: A binary list represents the existence of event=1 and absence=0.

        Output
            :return: A binary list with the same length as provided arrays.
        """
        assert len(array_A) == len(array_A), "Please provide two arrays with the same length."
        return ([a & b for a, b in zip(array_A, array_B)])
    
    def __operator_event_union(self, array_A, array_B):
        """
        Description
            This method finds events that occurred either in array_A or array_B (or operator). The two arrays were expected to have the same length

        Input
            :param array_A: A binary list represents the existence of event=1 and absence=0.
            :param array_B: A binary list represents the existence of event=1 and absence=0.

        Output
            :return: A binary list with the same length as provided arrays.
        """
        assert len(array_A) == len(array_A), "Please provide two arrays with the same length."
        return ([a | b for a, b in zip(array_A, array_B)])

    def __define_events(self):
        """
        Description
            This method defines the types of events that were detected during the recording session. The definition of each event is defined based on the following criteria:
            VOLUNTARY TAPPING: the tapping event observed in tap field + not observed in the move field + observed during the tapping period
            
        Output
            :return: The definitions of events are added as a class field that can be accessible.
        """
        self.left_voluntary_tap          = self.__operator_event_intersection(self.__operator_event_difference(self.left_tap, self.left_move), self.period_tap)
        self.left_involuntary_tap        = self.__operator_event_intersection(self.__operator_event_difference(self.left_tap, self.left_move), self.period_rest)
        self.left_involuntary_movements  = self.__operator_event_difference(self.left_move, self.left_tap)
        
        self.right_voluntary_tap         = self.__operator_event_intersection(self.__operator_event_difference(self.right_tap, self.right_move), self.period_tap)
        self.right_involuntary_tap       = self.__operator_event_intersection(self.__operator_event_difference(self.right_tap, self.right_move), self.period_rest)
        self.right_involuntary_movements = self.__operator_event_difference(self.right_move, self.right_tap)


    def get_dyskinetia_scores(self):
        """
        Description
            This method reads the Excel file containing the CDRS scores (right, left, total) of dyskinesia events and their corresponding timestamps included in different 
            sheets named with "sub-xxx" notation. This CDRS file is expected to be located under the "PATH\data" directory with the name CDRS.xlsx. The sheet of this Excel 
            belonging to the selected patients will be saved into a dataframe structure. To get the timestamp of dyskinesia scores (the same length as self.times field), 
            we first get the registration time of dyskinesia evaluation and corresponding dyskinesia score and fill an empty array with this score until the next 
            evaluation is made.
            
        Output
            :return: The definitions of dyskinesia scores in the right, left, and bilateral hemispheres were added as a field. It also returns a 
                     Python dictionary with three fields:
                     - key: "CDRS_right", value: an integer array
                     - key: "CDRS_left", value: an integer array
                     - key: "CDRS_total", value: an integer array
        """

        # check if the CDRS scores exist in the directory
        PATH_CDRS = self.__PATH +"\\data\\"
        assert os.path.exists(PATH_CDRS +"\\CDRS.xlsx"), f'CDRS.xlsx does not exist in directory: ({PATH_CDRS}) '

        # read the Excell sheet for the given SUB into the dataframe
        CDRS = pd.read_excel(PATH_CDRS +"CDRS.xlsx", sheet_name="sub-"+self.__SUB)
        CDRS = CDRS[['dopa_time','CDRS_total_right', 'CDRS_total_left', 'CDRS_total']]
        CDRS.dropna(inplace=True)

        self.__CDRS_dataframe = CDRS

        
        t_CDRS = CDRS.dopa_time.to_numpy()                  # get the timestamp of evaluation time (an array in minutes) 
        score_right = CDRS["CDRS_total_right"].to_numpy()   # get right hemisphere dyskinesia scores (an array) 
        score_left  = CDRS["CDRS_total_left"].to_numpy()    # get left hemisphere dyskinesia scores (an array) 
        score_total = CDRS["CDRS_total"].to_numpy()         # get total dyskinesia scores (bilateral) (an array) 
        
        # Find the indices where t_CDRS <= t
        # self.times / 60: to represent recording times in minutes instead of seconds
        CDRS_times = np.searchsorted(CDRS.dopa_time, self.times / 60) # find the indices of evaluation times in the time array
        CDRS_times = CDRS_times-1
        
        CDRS_left  = CDRS_times.copy()
        CDRS_right = CDRS_times.copy()
        CDRS_total = CDRS_times.copy()
        
        for index in CDRS.index:
            CDRS_left[CDRS_left==index]   = CDRS.iloc[index].CDRS_total_left
            CDRS_right[CDRS_right==index] = CDRS.iloc[index].CDRS_total_right
            CDRS_total[CDRS_total==index] = CDRS.iloc[index].CDRS_total
            
        # Take values at found indices
        self.CDRS_right = CDRS_right
        self.CDRS_left  = CDRS_left
        self.CDRS_total = CDRS_total 

    def __get_event_indices(self, array):
        """
        Description
            This method finds the indices of the beginning and end of the event in the array. Basically, one of the events (move or tapping) array will be provided as a parameter
            to the function.

        Input
            :param array: A binary list represents the existence=1 and absence=0 of a particular event (move/tapping).

        Output
            :return event_indices: A list containing tupples (start_index, finish_index) of index information for events
        """
        event_indices     = []
        event_started     = False
        event_start_index = None
        
        for i, num in enumerate(array):
            if num == 1:
                if not event_started:
                    event_start_index = i
                    event_started = True
            elif event_started:
                event_indices.append((event_start_index, i - 1))
                event_started = False
        
        # If an event is ongoing at the end of the array
        if event_started:
            event_indices.append((event_start_index, len(array) - 1))
        
        return event_indices

    def __populate_dataframe(self, dataset, patient, hemisphere, event_indices, event_type):

        """
        Description
            This method add a dataframe that contains all events detected for the given hemisphere of the patients. Initially, it acquires all the events contained 
            in _tap and _move fields. The information regarding the task (tapping, free, rest), the category (voluntary tapping, involuntary tapping, involuntary movement),
            event start and finish timestamps, etc are stored in this dataframe.

        Input
            :param dataset: A dataframe contains "patient","hemisphere","event_type", "event_no", "event_start_index", "event_finish_index", "event_start_time", 
                            "event_finish_time", "duration", "task" columns. It can be empty or filled previously.
            :param patient: A string representing the patient code
            :param hemisphere: A string representing the hemisphere information
            :param event_indices: A list containing tuples (start_index, finish_index) of index information for events
            :param event_type: A string represents the type of event that is considered
        
        Output
            :return dataset: The more populated version of the given dataframe structure
        """
        
        counter    = 1
        
        for event in event_indices:
            event_start_i = event[0]
            event_end_i   = event[1]

            if(event_start_i!=event_end_i):
    
                # get the corresponding tasks that were patient conducting during the event
                event_tasks   = self.task[event_start_i:event_end_i].tolist()
                # assing event to specific task [rest, tapping, free] based on the majority of duration passed for a given event
                event_task    = max(set(event_tasks), key=event_tasks.count) 
                # add event to the dataset
                dataset.loc[len(dataset)] = {"patient": patient, 
                                             "hemisphere": hemisphere, 
                                             "event": event_type, 
                                             "task": event_task,
                                             "event_no": "p_" + patient + "_" + hemisphere + "_" + event_type + str(counter),
                                             "event_start_index" : event_start_i, 
                                             "event_finish_index" : event_end_i, 
                                             "event_start_time" : self.times[event_start_i]/60, 
                                             "event_finish_time" : self.times[event_end_i]/60, 
                                             "duration": self.times[event_end_i] - self.times[event_start_i]}
                counter += 1

    def get_event_dataframe(self):

        """
        Description
            This method populates an event dataframe for a given patient, hemisphere, and event type.

        Input
            :param dataset: A dataframe contains "patient","hemisphere","event_type", "event_no", "event_start_index", "event_finish_index", "event_start_time", 
                                                 "event_finish_time", "duration", "task" columns
            :param patient: A string representing the patient code
            :param hemisphere: A string representing the hemisphere information
            :param event_indices: A list containing tuples (start_index, finish_index) of index information for events
            :param event_type: A string represents the type of event that is considered
        
        Output
            :return dataset: The more populated version of the given dataframe structure
        """
        # get start and finish indices of different event types in different hemispheres
        left_moves        = self.__get_event_indices(self.left_move.tolist())
        left_tapping      = self.__get_event_indices(self.left_tap.tolist())
        right_moves       = self.__get_event_indices(self.right_move.tolist())
        right_tapping     = self.__get_event_indices(self.right_tap.tolist())
        bilateral_moves   = self.__get_event_indices(self.bilateral_move.tolist())
        bilateral_tapping = self.__get_event_indices(self.bilateral_tap.tolist())

        # create an empty event dataframe
        events = pd.DataFrame(columns=["patient", "hemisphere", "event", "task", "event_no", "event_start_index", "event_finish_index", 
                                       "event_start_time", "event_finish_time", "duration"])   

        # populate the empty dataframe
        self.__populate_dataframe(events, self.__SUB, "left"     , left_moves       , "move")
        self.__populate_dataframe(events, self.__SUB, "left"     , left_tapping     , "tap")
        self.__populate_dataframe(events, self.__SUB, "right"    , right_moves      , "move")
        self.__populate_dataframe(events, self.__SUB, "right"    , right_tapping    , "tap")
        self.__populate_dataframe(events, self.__SUB, "bilateral", bilateral_moves  , "move" )
        self.__populate_dataframe(events, self.__SUB, "bilateral", bilateral_tapping, "tap" )
    
        # define dyskinesia_score column
        events["dyskinesia_score"] = 0
        # for each index check the hemisphere and get the corresponding dyskinesia score on the event onset
        for index in events.index:
            if(events.iloc[index].hemisphere == "right")       : events.loc[index, ['dyskinesia_score']] = self.CDRS_right[events.iloc[index].event_start_index]
            elif(events.iloc[index].hemisphere == "left")      : events.loc[index, ['dyskinesia_score']] = self.CDRS_left[events.iloc[index].event_start_index]
            elif(events.iloc[index].hemisphere == "bilateral") : events.loc[index, ['dyskinesia_score']] = self.CDRS_total[events.iloc[index].event_start_index]

        # define if the movement is voluntary or not as a column
        events["is_voluntary"]  = False
        events.loc[(events.event == "tap") & (events.task == "tap"), 'is_voluntary'] = True
        
        # add event categories to the dataframe
        events = self.__define_event_categories(events)
        return events
    
    def get_CDRS_dataframe(self):
        return self.__CDRS_dataframe

    def __define_event_categories(self, dataset):
    
        def categorize_event(row):
            if (row['event'] == 'tap' and row['task'] == 'tap') : return 'voluntary_tapping'
            elif (row['event'] == 'tap' and row['task'] != 'tap'): return 'involuntary_tapping'
            else : return 'involuntary_movement'
            
            
    
        # Apply the function to create the new column 'event_category'
        dataset['event_category'] = dataset.apply(categorize_event, axis=1)
        dataset                   = dataset[['patient', 'hemisphere', 'event_no', 'event', 'task', 'is_voluntary', 'event_category', 'event_start_index', 
                                             'event_finish_index', 'event_start_time', 'event_finish_time', 'duration', 'dyskinesia_score']]

        return dataset
        
###############################################################################################################################################################
###############################################################################################################################################################
###############################################################################################################################################################
###############################################################################################################################################################
###############################################################################################################################################################

class KINEMATIC_DATA:

    def __init__(self, PATH, SUB):
        
        #assert DAT_SOURCE in ['acc_left', 'acc_right'], f'Please pass accelerometer DAT_SOURCE ({DAT_SOURCE})'

        # use DATA_IO to load data structure
        data_IO_r           = DATA_IO(PATH, SUB, 'acc_right')
        data_IO_l           = DATA_IO(PATH, SUB, 'acc_left')
        self.__dat_r        = data_IO_r.get_data()
        self.__dat_l        = data_IO_l.get_data()

        # populate class fields
        self.fs             = self.__dat_r.fs
        self.times          = self.__dat_r.data[:,self.__dat_r.colnames.index('dopa_time')]
        
        # load right accelerometer data
        self.ACC_R_X = self.__dat_r.data[:,self.__dat_r.colnames.index('ACC_R_X')]
        self.ACC_R_Y = self.__dat_r.data[:,self.__dat_r.colnames.index('ACC_R_Y')]
        self.ACC_R_Z = self.__dat_r.data[:,self.__dat_r.colnames.index('ACC_R_Z')]

        # load left accelerometer data
        self.ACC_L_X = self.__dat_l.data[:,self.__dat_l.colnames.index('ACC_L_X')]
        self.ACC_L_Y = self.__dat_l.data[:,self.__dat_l.colnames.index('ACC_L_Y')]
        self.ACC_L_Z = self.__dat_l.data[:,self.__dat_l.colnames.index('ACC_L_Z')]

    def extract_accelerometer_events(self, event_dataset, hemisphere="", event_type="", event_category="", dyskinesia_score="", alignment="onset"):

        fs             = self.fs
        
        if(hemisphere!=""):
            # check if the selected hemisphere is valid
            assert hemisphere in ["right", "left", "bilateral"], f'Please choose hemisphere as "right", "left", "bilateral"'
            
        if(event_type!=""):
            # check if the selected event type is valid
            assert event_category in event_dataset.event.unique().tolist(), f'Please enter valid event type as "move", "tap"'

        if(event_category!=""): 
            # check if the event category is valid
            assert event_category in event_dataset.event_category.unique().tolist(), f'Please enter valid event category, not ({alignment})'
            
        # check if event alignment strategy is valid
        assert alignment in ["onset", "offset"], f'Please choose alignment as "onset", "offset"'
       
        #################################################################################################################################
        dataset = event_dataset[event_dataset['hemisphere']==hemisphere] if hemisphere != "" else event_dataset # select hemisphere
        dataset = dataset[dataset['event']==event_type] if event_type != "" else dataset                        # select event type
        dataset = dataset[dataset['event_category']==event_category] if event_category != "" else dataset       # select event category
        dataset = dataset[dataset['dyskinesia_score']==dyskinesia_score] if dyskinesia_score != "" else dataset # select dyskinesia score
        #################################################################################################################################
        
        # create empty arrays for storing accelerometer data for selected event category
        acc_x = []
        acc_y = []
        acc_z = []
    
        # iterate across dataframe rows
        for _, row in dataset.iterrows():
    
            # If events aligned based on their onset
            if(alignment=="onset"):
                start_index    = row['event_start_index'] - fs  # 1 sec before event onset
                finish_index   = row['event_finish_index']      # event offset
            # If events aligned based on their offset
            else:
                start_index    = row['event_start_index']       # event onset
                finish_index   = row['event_finish_index'] + fs # 1 sec after event offset
    
            # get hemisphere information
            hemisphere     = row['hemisphere']
            
            if(hemisphere == "right"):
                acc_data_x = self.ACC_R_X[start_index:finish_index].tolist()
                acc_data_y = self.ACC_R_Y[start_index:finish_index].tolist()
                acc_data_z = self.ACC_R_Z[start_index:finish_index].tolist()
            else:
                acc_data_x = self.ACC_L_X[start_index:finish_index].tolist()
                acc_data_y = self.ACC_L_Y[start_index:finish_index].tolist()
                acc_data_z = self.ACC_L_Z[start_index:finish_index].tolist()
            
            acc_x.append(acc_data_x)
            acc_y.append(acc_data_y)
            acc_z.append(acc_data_z)

        # store selected events in 3 axis of accelerometer data into a dictionary file
        acc_events      = {}
        acc_events["X"] = acc_x
        acc_events["Y"] = acc_y
        acc_events["Z"] = acc_z
        
        return acc_events
            
            





    