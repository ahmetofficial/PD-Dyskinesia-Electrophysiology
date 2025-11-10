"""
Power spectral utilisation functions
"""

import pandas as pd
import numpy as np
import pyvista as pv
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

def create_3D_grid_around_anatomical_structure(structure, n_bins):
    
    n_bins += 1
    
    # define the grid boundaries for given 3D anatomical structure (pyvista mesh)
    xmin, xmax, ymin, ymax, zmin, zmax = structure.bounds
    
    # calculate spacing based on bounds and the desired number of bins
    x_spacing = (xmax - xmin) / (n_bins - 1)
    y_spacing = (ymax - ymin) / (n_bins - 1)
    z_spacing = (zmax - zmin) / (n_bins - 1)
    
    # create a grid of points
    x_points = np.linspace(xmin, xmax, n_bins)
    y_points = np.linspace(ymin, ymax, n_bins)
    z_points = np.linspace(zmin, zmax, n_bins)
    
    # create the lines for each axis to show full grid lines
    lines = []
    
    # generate lines along the z-axis at each x-y slice
    for x in x_points:
        for y in y_points:
            points = np.array([[x, y, z] for z in z_points])
            line = pv.Line(points[0], points[-1], resolution=n_bins-1)
            lines.append(line)
    
    # generate lines along the x-axis at each y-z slice
    for y in y_points:
        for z in z_points:
            points = np.array([[x, y, z] for x in x_points])
            line = pv.Line(points[0], points[-1], resolution=n_bins-1)
            lines.append(line)
    
    # generate lines along the y-axis at each x-z slice
    for x in x_points:
        for z in z_points:
            points = np.array([[x, y, z] for y in y_points])
            line = pv.Line(points[0], points[-1], resolution=n_bins-1)
            lines.append(line)
    
    # create a single PolyData object from the lines
    grid = pv.PolyData()
    for line in lines:
        grid = grid.merge(line)

    return grid


def find_grid_cell_index_for_contact(point, grid):
    
    # extract bounds from the grid lines PolyData
    xmin, xmax, ymin, ymax, zmin, zmax = grid.bounds
    
    # calculate the grid dimensions based on the lines created
    nx        = len(np.unique(grid.points[:, 0]))  # unique x-coordinates
    ny        = len(np.unique(grid.points[:, 1]))  # unique y-coordinates
    nz        = len(np.unique(grid.points[:, 2]))  # unique z-coordinates
    
    # calculate the cell size
    x_spacing = (xmax - xmin) / (nx - 1)
    y_spacing = (ymax - ymin) / (ny - 1)
    z_spacing = (zmax - zmin) / (nz - 1)

    # calculate the cell index along each axis
    x_index = int((point[0] - xmin) / x_spacing)
    y_index = int((point[1] - ymin) / y_spacing)
    z_index = int((point[2] - zmin) / z_spacing)
    
    # ensure the indices are within the grid bounds
    x_index = min(max(x_index, 0), nx - 1)
    y_index = min(max(y_index, 0), ny - 1)
    z_index = min(max(z_index, 0), nz - 1)
    
    return (x_index, y_index, z_index)

def assign_recording_channels_to_grid_cells(df_MNI_coordinates, grid):
    
    grid_bin_x = []
    grid_bin_y = []
    grid_bin_z = []
    
    for index, row in df_MNI_coordinates.iterrows():
        channel_coordinates = [row.x, row.y, row.z]
        cell_index          = find_grid_cell_index_for_contact(channel_coordinates, grid)
        grid_bin_x.append(cell_index[0])
        grid_bin_y.append(cell_index[1])
        grid_bin_z.append(cell_index[2])
    
    df_MNI_coordinates["grid_bin_x"] = grid_bin_x
    df_MNI_coordinates["grid_bin_y"] = grid_bin_y
    df_MNI_coordinates["grid_bin_z"] = grid_bin_z

    return df_MNI_coordinates

def extract_grid_cell_centers(grid, n_bins):
    
    # extract bounds from the grid lines PolyData
    xmin, xmax, ymin, ymax, zmin, zmax = grid.bounds
    
    # compute bin widths
    dx = (xmax - xmin) / n_bins
    dy = (ymax - ymin) / n_bins
    dz = (zmax - zmin) / n_bins
    
    # compute centers along each axis
    x_centers = xmin + (np.arange(n_bins) + 0.5) * dx
    y_centers = ymin + (np.arange(n_bins) + 0.5) * dy
    z_centers = zmin + (np.arange(n_bins) + 0.5) * dz

    # generate 3D grid of centers and cell IDs
    x_grid, y_grid, z_grid = np.meshgrid(x_centers, y_centers, z_centers, indexing='ij')

    # create dataframe
    data = {'grid_bin_center_x': x_grid.ravel(), 'grid_bin_center_y': y_grid.ravel(),'grid_bin_center_z': z_grid.ravel()}
    data = pd.DataFrame(data)

    # get the assign grid cell for each center point
    x_id = []
    y_id = []
    z_id = []

    for i, row in data.iterrows():
        point                     = [row.grid_bin_center_x, row.grid_bin_center_y, row.grid_bin_center_z]
        x_index, y_index, z_index = find_grid_cell_index_for_contact(point, grid)
        x_id.append(x_index)
        y_id.append(y_index)
        z_id.append(z_index)

    data["grid_bin_x"] = x_id
    data["grid_bin_y"] = y_id
    data["grid_bin_z"] = z_id
    
    return data

###############################################################################################################################################
## SINGLE AXES ################################################################################################################################
###############################################################################################################################################

def get_intercept_definitions_for_severity_levels_single(result, MNI_axis):

    # Define severity levels
    severity_numeric = [0, 1, 2]  # numeric encoding in dataset
    severity_names   = ["MED-OFF", "MED-ON", "LID"]

    # Get names of all fixed effects in the model
    names = result.model.exog_names
    def idx(name): return names.index(name)

    results = []

    for sev_num, sev_name in zip(severity_numeric, severity_names):
        contrast = np.zeros(len(names))

        # Main intercept
        contrast[idx("Intercept")] = 1

        # Add main effect for severity (if not baseline)
        if sev_num != severity_numeric[0]:
            term_name = f"C(severity_numeric)[T.{sev_num}]"
            if term_name in names:
                contrast[idx(term_name)] = 1

        # perform t-test
        test            = result.t_test(contrast.reshape(1, -1))
        est             = test.effect[0]
        ci_low, ci_high = test.conf_int()[0]
        pval            = test.pvalue

        results.append({"severity": sev_name, "intercept": est, "ci_lower": ci_low, "ci_upper": ci_high, "p_value": pval})

    return pd.DataFrame(results)
    
def extract_spatial_dataset_single(dataset, event_segment, frequency_band, MNI_axis):
    
    if(event_segment == "pre")     : event_prefix = "pre_event_"
    elif(event_segment == "event") : event_prefix = "event_"
    elif(event_segment == "post")  : event_prefix = "post_event_"
        
    dataset_spatial                = dataset[["patient", "hemisphere", "channel", event_prefix + frequency_band + "_mean", MNI_axis, "severity_numeric"]]
    dataset_spatial["channel_id"]  = dataset["patient"] + "_" + dataset["hemisphere"] + "_" + dataset["channel"]
    dataset_spatial                = dataset_spatial[["patient", "channel_id", MNI_axis, event_prefix + frequency_band + "_mean", "severity_numeric"]]
    dataset_spatial                = dataset_spatial.rename(columns={event_prefix + frequency_band + "_mean": frequency_band})

    return dataset_spatial

def get_slope_definitions_for_severity_levels_single(result, MNI_axis):
    
    names = result.model.exog_names
    def idx(name): return names.index(name)
    
    # Define contrasts
    contrast_0 = np.zeros(len(names)); contrast_0[idx(MNI_axis)] = 1
    contrast_1 = np.zeros(len(names)); contrast_1[idx(MNI_axis)] = 1; contrast_1[idx(f"{MNI_axis}:C(severity_numeric)[T.1]")] = 1
    contrast_2 = np.zeros(len(names)); contrast_2[idx(MNI_axis)] = 1; contrast_2[idx(f"{MNI_axis}:C(severity_numeric)[T.2]")] = 1
    
    severities = ["MED-OFF", "MED-ON", "LID"]
    contrasts  = [contrast_0, contrast_1, contrast_2]
    
    results = []
    for severity, contrast in zip(severities, contrasts):
        test            = result.t_test(contrast.reshape(1, -1))
        est             = test.effect[0]
        ci_low, ci_high = test.conf_int()[0]
        pval            = test.pvalue
        results.append({"severity": severity, "slope": est, "ci_lower": ci_low, "ci_upper": ci_high,"p_value": pval})
    
    return pd.DataFrame(results)

def slope_differences_between_severity_levels_single(result, MNI_axis):
    
    # fixed effect names for indexing
    names = result.fe_params.index.tolist()
    idx   = lambda name: names.index(name)
    
    # Differences
    contrast_01                                    = np.zeros(len(names))  # slope_0 - slope_1 = - z:C(severity_numeric)[T.1]
    contrast_01[idx(f"{MNI_axis}:C(severity_numeric)[T.1]")] = -1

    contrast_12                                    = np.zeros(len(names))  # slope_1 - slope_2 = z:C(severity_numeric)[T.1] - z:C(severity_numeric)[T.2]
    contrast_12[idx(f"{MNI_axis}:C(severity_numeric)[T.1]")] = 1
    contrast_12[idx(f"{MNI_axis}:C(severity_numeric)[T.2]")] = -1

    contrast_02                                    = np.zeros(len(names))  # slope_0 - slope_2 = - z:C(severity_numeric)[T.2]
    contrast_02[idx(f"{MNI_axis}:C(severity_numeric)[T.2]")] = -1

    results                                        = []

    for name, contrast in zip(["noLID-noDOPA vs noLID-DOPA", "noLID-DOPA vs LID", "noLID-noDOPA vs LID"],[contrast_01, contrast_12, contrast_02]):
    #for name, contrast in zip(["noLID-noDOPA vs noLID-DOPA", "noLID-DOPA vs LID"],[contrast_01, contrast_12]): # only focus on two comparison
        test            = result.t_test(contrast.reshape(1, -1))  # Ensure 2D shape
        est             = test.effect[0]
        ci_low, ci_high = test.conf_int()[0]
        pval            = test.pvalue.item()

        results.append({"comparison": name, "slope_diff": est, "ci_lower": ci_low, "ci_upper": ci_high,"p_value": pval})

    return pd.DataFrame(results)

def fit_LME_for_spatial_dynamics_single(dataset, frequency_band, event_segment, MNI_axis):

    # STEP 1: get the spatial data for selected parameters
    dataset_spatial = extract_spatial_dataset_single(dataset, event_segment=event_segment, frequency_band=frequency_band, MNI_axis=MNI_axis)

    # STEP 2: LME modelling: random intercept + fixed slopes
        
    model     = smf.mixedlm(formula    = f"{frequency_band} ~ {MNI_axis} * C(severity_numeric)", 
                            data       = dataset_spatial, 
                            groups     = dataset_spatial["patient"], 
                            vc_formula = {"channel": "0 + C(channel_id)"})  # random intercept per channel
    result    = model.fit()

    # STEP 4: get slope definitions for severity levels 
    intercept_by_severity                  = get_intercept_definitions_for_severity_levels_single(result, MNI_axis)
    intercept_by_severity["frequency"]     = frequency_band
    intercept_by_severity["axis"]          = MNI_axis
    intercept_by_severity["event_segment"] = event_segment
    intercept_by_severity["p_value"]       = multipletests(intercept_by_severity.p_value.to_list(), method="holm")[1]
    
    # STEP 4: get slope definitions for severity levels 
    slopes_definitions_by_severity                  = get_slope_definitions_for_severity_levels_single(result, MNI_axis)
    slopes_definitions_by_severity["frequency"]     = frequency_band
    slopes_definitions_by_severity["axis"]          = MNI_axis
    slopes_definitions_by_severity["event_segment"] = event_segment
    slopes_definitions_by_severity["p_value"]       = multipletests(slopes_definitions_by_severity.p_value.to_list(), method="holm")[1]

    # STEP 5: get slope differences between severity levels 
    slopes_differences_between_severity                  = slope_differences_between_severity_levels_single(result, MNI_axis)
    slopes_differences_between_severity["frequency"]     = frequency_band
    slopes_differences_between_severity["axis"]          = MNI_axis
    slopes_differences_between_severity["event_segment"] = event_segment
    slopes_differences_between_severity["p_value"]       = multipletests(slopes_differences_between_severity.p_value.to_list(), method="holm")[1]

    return intercept_by_severity, slopes_definitions_by_severity, slopes_differences_between_severity
    
###############################################################################################################################################
## MULTIPLE AXES ##############################################################################################################################
###############################################################################################################################################

def extract_spatial_dataset(dataset, event_segment, frequency_band):
    
    if(event_segment == "pre")     : event_prefix = "pre_event_"
    elif(event_segment == "event") : event_prefix = "event_"
    elif(event_segment == "post")  : event_prefix = "post_event_"
        
    dataset_spatial                = dataset[["patient", "hemisphere", "channel", event_prefix + frequency_band + "_mean", "x", "y", "z", "severity_numeric"]]
    dataset_spatial["channel_id"]  = dataset["hemisphere"] + "_" + dataset["channel"]
    dataset_spatial["channel_id"]  = dataset_spatial["channel_id"].astype("category")
    dataset_spatial                = dataset_spatial[["patient", "channel_id", "x", "y", "z", event_prefix + frequency_band + "_mean", "severity_numeric"]]
    dataset_spatial                = dataset_spatial.rename(columns={event_prefix + frequency_band + "_mean": frequency_band})

    return dataset_spatial

def get_intercept_definitions_for_severity_levels(result):

    # extract intercepts (predicted beta_low values at x=y=z=0) for each severity level from a fitted MixedLM model.

    # Define severity levels
    severity_numeric = [0, 1, 2]           # numeric encoding in dataset
    severity_names   = ["MED-OFF", "MED-ON", "LID"]

    # Get names of all fixed effects in the model
    names = result.model.exog_names
    def idx(name): return names.index(name)  # helper to find coefficient index

    results = []

    # loop over each severity level
    for sev_num, sev_name in zip(severity_numeric, severity_names):

        # initialize contrast vector
        contrast = np.zeros(len(names))

        # always include the main intercept
        contrast[idx("Intercept")] = 1

        # if not baseline severity, add its main effect term
        if sev_num != severity_numeric[0]:
            term_name = f"C(severity_numeric)[T.{sev_num}]"
            if term_name in names:
                contrast[idx(term_name)] = 1

        # perform t-test on this contrast
        test = result.t_test(contrast.reshape(1, -1))

        # extract effect, confidence interval, and p-value
        est             = test.effect[0]
        ci_low, ci_high = test.conf_int()[0]
        pval            = test.pvalue

        results.append({"severity": sev_name, "intercept": est, "ci_lower": ci_low, "ci_upper": ci_high, "p_value": pval})

    return pd.DataFrame(results)
    
def get_slope_definitions_for_severity_levels(result):

    # Extract slopes for multiple spatial axes (x, y, z) per severity level from a fitted MixedLM model.

    # Define axes and severity groups
    axes             = ["x", "y", "z"]
    severity_numeric = [0, 1, 2]           # numeric encoding of states in dataset
    severity_names   = ["MED-OFF", "MED-ON", "LID"]
    
    # Get names of all fixed effects in the model
    names = result.model.exog_names
    def idx(name): return names.index(name)  # helper to get index of a coefficient
    
    results = []

    # loop over each axis and each severity level
    for axis in axes:
        
        for sev_num, sev_name in zip(severity_numeric, severity_names):
            
            # initialize contrast vector (zeros for all coefficients)
            contrast            = np.zeros(len(names))
            
            # always include the main effect of the axis
            contrast[idx(axis)] = 1
            
            # if severity is not baseline, include interaction term
            if(sev_num != severity_numeric[0]):
                
                term_name = f"{axis}:C(severity_numeric)[T.{sev_num}]"
                
                if(term_name in names):
                    contrast[idx(term_name)] = 1
            
            # perform t-test on the linear combination defined by contrast
            test            = result.t_test(contrast.reshape(1, -1))
            
            # extract slope estimate, confidence interval, and p-value
            est             = test.effect[0]
            ci_low, ci_high = test.conf_int()[0]
            pval            = test.pvalue
            
            results.append({"axis": axis, "severity": sev_name, "slope": est, "ci_lower": ci_low, "ci_upper": ci_high, "p_value": pval})
    
    # Return as a tidy DataFrame
    return pd.DataFrame(results)


def slope_differences_between_severity_levels(result):

    # Compute pairwise slope differences between severity levels for multiple axes from a fitted MixedLM model.

    # assumes severity_numeric has three levels: 0 (baseline), 1, 2.
    # pairwise comparisons: 0 vs 1, 1 vs 2, 0 vs 2.
    # interaction terms must follow statsmodels naming: e.g., 'x:C(severity_numeric)[T.1]'.
    
    axes             = ["x", "y", "z"]
    severity_numeric = [0, 1, 2]
    comparisons      = [("0 vs 1", 0, 1), ("1 vs 2", 1, 2)]
    severity_names   = ["MED-OFF", "MED-ON", "LID"]

    # get names of all fixed effects
    names            = result.fe_params.index.tolist()
    idx              = lambda name: names.index(name)
    results          = []

    for axis in axes:
        
        for comp_name, group1, group2 in comparisons:

            contrast = np.zeros(len(names))
            
            # determine slope difference formula
            if(group1 == 0):
                term_name              = f"{axis}:C(severity_numeric)[T.{group2}]"
                if(term_name in names) : contrast[idx(term_name)] = -1
                    
            elif(group2 == 0):
                term_name              = f"{axis}:C(severity_numeric)[T.{group1}]"
                if(term_name in names) : contrast[idx(term_name)] = 1
                    
            else:
                # difference between two non-baseline severity groups
                term_a             = f"{axis}:C(severity_numeric)[T.{group1}]"
                term_b             = f"{axis}:C(severity_numeric)[T.{group2}]"
                
                if term_a in names : contrast[idx(term_a)] = 1
                if term_b in names : contrast[idx(term_b)] = -1
            
            # t-test on contrast
            test            = result.t_test(contrast.reshape(1, -1))
            est             = test.effect[0]
            ci_low, ci_high = test.conf_int()[0]
            pval            = test.pvalue.item()
            
            results.append({"axis": axis, "comparison": comp_name, "slope_diff": est, "ci_lower": ci_low, "ci_upper": ci_high, "p_value": pval})
    
    return pd.DataFrame(results)


def fit_LME_for_spatial_dynamics(dataset, frequency_band, event_segment):

    # STEP 1: get the spatial data for selected parameters
    dataset_spatial = extract_spatial_dataset(dataset, event_segment=event_segment, frequency_band=frequency_band)
    
    # STEP 2: LME modelling: random intercept + fixed slopes
            
    model     = smf.mixedlm(formula    = f"{frequency_band} ~ (x + y + z) * C(severity_numeric)",
                            data       = dataset_spatial,
                            groups     = dataset_spatial["patient"], 
                            vc_formula = {"channel": "0 + C(channel_id)"})
    try:
        result    = model.fit()
    except:
        print(frequency_band + " - " + event_segment)
    
    # STEP 4: get slope differences between severity levels 
    intercepts_severity                                  = get_intercept_definitions_for_severity_levels(result)
    intercepts_severity["frequency"]                     = frequency_band
    intercepts_severity["event_segment"]                 = event_segment
    intercepts_severity["p_value"]                       = multipletests(intercepts_severity.p_value.to_list(), method="holm")[1]
    
    # STEP 4: get slope definitions for severity levels 
    slopes_definitions_by_severity                       = get_slope_definitions_for_severity_levels(result)
    slopes_definitions_by_severity["frequency"]          = frequency_band
    slopes_definitions_by_severity["event_segment"]      = event_segment
    slopes_definitions_by_severity["p_value"]            = multipletests(slopes_definitions_by_severity.p_value.to_list(), method="holm")[1]
    
    # STEP 5: get slope differences between severity levels 
    slopes_differences_between_severity                  = slope_differences_between_severity_levels(result)
    slopes_differences_between_severity["frequency"]     = frequency_band
    slopes_differences_between_severity["event_segment"] = event_segment
    slopes_differences_between_severity["p_value"]       = multipletests(slopes_differences_between_severity.p_value.to_list(), method="holm")[1]
    
    return intercepts_severity, slopes_definitions_by_severity, slopes_differences_between_severity
