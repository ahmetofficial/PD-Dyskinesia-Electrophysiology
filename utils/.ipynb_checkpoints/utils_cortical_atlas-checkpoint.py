"""
cortical atlas utilisation functions
"""
import os
import pandas as pd
import numpy as np
import pyvista as pv 
from scipy.interpolate import Rbf
from scipy.spatial import cKDTree
from scipy.spatial import KDTree

def map_ECOG_channels_on_cortical_surface(MNI_ECoG_channels, cortex_mesh):
    
    x_mapped = []
    y_mapped = []
    z_mapped = []
    
    for index, row in MNI_ECoG_channels.iterrows():
        
        hemisphere_mesh = cortex_mesh[row.hemisphere + "_hemisphere"]
        
        # extract surface points from the PolyData (cortex surface)
        surface_points  = hemisphere_mesh.points  # (N, 3) point cloud
    
        # build KDTree from surface points for fast nearest-neighbor search
        kdtree          = cKDTree(surface_points)
    
        # extract coordinates of the  ECoG channel
        coordinates     = [[row.x, row.y, row.z]]  # create a (1, 3) array 
        
        # find  nearest surface points for each ECoG channel coordinate
        dist, indices   = kdtree.query(coordinates)
    
        # get the mapped points (the nearest surface points)
        mapped_points   = surface_points[indices]
        
        # Step 6: Add new columns for the mapped coordinates in the DataFrame
        x_mapped.append(mapped_points[:, 0][0])
        y_mapped.append(mapped_points[:, 1][0])
        z_mapped.append(mapped_points[:, 2][0])
    
    MNI_ECoG_channels["x_mapped"] = x_mapped
    MNI_ECoG_channels["y_mapped"] = y_mapped
    MNI_ECoG_channels["z_mapped"] = z_mapped

    return MNI_ECoG_channels
    
def parcellate_ECoG_channels_to_cortical_areas(AAL3_object, AAL3_labels, MNI_ECoG_channels):
    
    AAL3_data          = AAL3_object.get_fdata()
    AAL3_affine_matrix = AAL3_object.affine  # Get the affine matrix

    #####################################################################################################
    non_zero_voxels = np.column_stack(np.where(AAL3_data > 0))  # Get all voxel indices with valid labels
    kd_tree         = KDTree(non_zero_voxels)
    def find_closest_valid_voxel(coord):
        """Find the closest voxel with a valid label if the coordinate is outside the atlas."""
        _, closest_index = kd_tree.query(coord)  # Find the closest non-zero labeled voxel
        return non_zero_voxels[closest_index]
    #####################################################################################################
    
    # find the coordinates of the corresponding voxel of each ECoG channel
    voxel_coordinates = []
    for index, row in MNI_ECoG_channels.iterrows():
        coordinates = [row.x, row.y, row.z]
        voxel_coordinates.append(np.round(np.linalg.inv(AAL3_affine_matrix).dot(np.append(coordinates, 1))[:3]).astype(int))
    
    # find the voxel ids of given voxel coordinates
    voxel_ids = []
    for coordinate in voxel_coordinates:
        
        x, y, z  = coordinate
        voxel_id = int(AAL3_data[x, y, z])
        
        if (voxel_id!=0): # the ECoG channel within the AAL3 atlas
            voxel_ids.append(voxel_id)
        else: # the ECoG channel is outside of the AAL3 atlas, find the closest voxel
            closest_voxel = find_closest_valid_voxel(coordinate)
            x, y, z = closest_voxel
            voxel_ids.append(int(AAL3_data[x, y, z]))  # Append closest voxel's ID
    
    # find the anatomical label of each voxel id that represents an ECoG channel
    cortical_parcellation = []
    for id in voxel_ids:
        voxel = AAL3_labels[AAL3_labels.voxel_no == id]
        if(len(voxel)==0): # ECoG contact is outside of the AAL3 atlas definition
            cortical_parcellation.append(np.nan)
        else:
            cortical_parcellation.append(voxel.anatomical_description.values[0])
    
    MNI_ECoG_channels["AAL3_voxel_id"]     = voxel_ids
    MNI_ECoG_channels["AAL3_parcellation"] = cortical_parcellation
    
    gyrus_to_functional_cortex_mapping = {
        'Precentral gyrus': 'Motor cortex',
        'Postcentral gyrus': 'Sensory cortex',
        'Middle frontal gyrus': 'Prefrontal cortex',
        'Superior parietal gyrus': 'Parietal cortex',
        'Superior frontal gyrus, dorsolateral': 'Prefrontal cortex',
        'Inferior parietal gyrus, excluding supramarginal and angular gyri': 'Parietal cortex'
    }

    gyrus_to_functional_cortex_mapping = {
        'Precentral gyrus': 'Motor cortex',
        'Rolandic operculum': 'Motor cortex',
        'Supplementary motor area': 'Motor cortex',
        'Postcentral gyrus': 'Sensory cortex',
        'Middle frontal gyrus': 'Prefrontal cortex',
        'Superior frontal gyrus, dorsolateral': 'Prefrontal cortex',
        'Inferior frontal gyrus, opercular part': 'Prefrontal cortex',
        'Inferior frontal gyrus, triangular part': 'Prefrontal cortex',
        'Superior parietal gyrus': 'Parietal cortex',
        'Inferior parietal gyrus, excluding supramarginal and angular gyri': 'Parietal cortex'
    }
    
    # create the new column by mapping the values
    MNI_ECoG_channels['AAL3_cortex'] = MNI_ECoG_channels['AAL3_parcellation'].map(gyrus_to_functional_cortex_mapping)
    
    return MNI_ECoG_channels


def flip_ECoG_channels_left_to_right_hemisphere(dataset):
    
    # flip the x-coordinate where the hemisphere is 'left'
    dataset.loc[dataset['hemisphere'] == 'left', 'x'] = -dataset.loc[dataset['hemisphere'] == 'left', 'x']
    
    # update the 'hemisphere' column to 'right_flipped' for these channels
    dataset.loc[dataset['hemisphere'] == 'left', 'hemisphere'] = 'right_flipped'
    
    return dataset
    
#############################################################################################################



def measure_mean_psd_activity_for_ECoG_channel(df_ECOG_PSD, df_ECOG_channel_coordinates, feature_set):
    
    ECOG_dynamics_dict = {}

    # iterate across all features in feature set
    for feature in feature_set:
        
        channel_dynamics = df_ECOG_channel_coordinates.copy()
        
        #iterate across all dyskinesia severity conditions
        for severity in list(df_ECOG_PSD.keys()):
            
            dyskinesia_group = df_ECOG_PSD[severity]
            values           = []
            
            # iterate across all ECoG channels
            for index, row in channel_dynamics.iterrows():
                values.append(np.nanmedian(dyskinesia_group[(dyskinesia_group.patient == row.patient) & 
                                                          (dyskinesia_group.ECoG_hemisphere == row.hemisphere) & 
                                                          (dyskinesia_group.ECoG_channel == row.channel)][feature]))
            
            channel_dynamics[severity] = values
            
        # map left_hemisphere coordinates into right hemisphere
        channel_dynamics['x']  = channel_dynamics['x'].apply(lambda x: -x if x < 0 else x) 
        ECOG_dynamics_dict[feature] = channel_dynamics

    return ECOG_dynamics_dict
    
def map_ecog_activity_to_cortex(ecog_activity, cortex_mesh, feature, severity):
    """
    Maps ECoG data onto a brain surface using interpolation.

    Parameters:
    - ecog_activity: DataFrame containing position of ECoG channels in MNI space ['x', 'y', 'z'] and corresponding
                     electrophysiological features observed in these channels in standard frequency bands
    - cortex_mesh  : PyVista PolyData object representing the cortical atlas mesh.

    Returns:
    - cortex_mesh  : PyVista PolyData object representing the cortical atlas mesh with interpolated feature values.
    """
    # extract points from cortex mesh
    cortex_points = cortex_mesh.points

    # extract ECoG points and values
    ecog_activity_filtered   = ecog_activity[feature][['x', 'y', 'z', severity]].dropna(subset=[severity])
    ecog_channel_coordinates = ecog_activity_filtered[['x', 'y', 'z']].values
    ecog_feature_values      = ecog_activity_filtered[severity].values
    
    # interpolate linearly the feature values using RBF
    rbf = Rbf(ecog_channel_coordinates[:, 0], ecog_channel_coordinates[:, 1], ecog_channel_coordinates[:, 2], 
              ecog_feature_values, function='gaussian')

    # map the ECoG feature values onto the cortical surface
    cortex_mesh[feature] = rbf(cortex_points[:, 0], cortex_points[:, 1], cortex_points[:, 2])

    return cortex_mesh


def map_ecog_activity_to_cortex_by_radius(ecog_activity, cortex_mesh, feature, severity, radius):
    """
    Maps ECoG data onto a brain surface using interpolation,
    limited to areas within a specified radius from ECoG coordinates.

    Parameters:
    - ecog_activity: DataFrame containing position of ECoG channels in MNI space ['x', 'y', 'z'] and corresponding
                     electrophysiological features observed in these channels in standard frequency bands.
    - cortex_mesh: PyVista PolyData object representing the cortical atlas mesh.
    - feature: The specific feature to map onto the cortex.
    - severity: The column indicating the severity or intensity of the feature values.
    - radius: The maximum distance from ECoG points for interpolation.

    Returns:
    - cortex_mesh: PyVista PolyData object representing the cortical atlas mesh with interpolated feature values.
    """
    # Extract points from cortex mesh
    cortex_points = cortex_mesh.points

    # Extract ECoG points and values
    ecog_activity_filtered = ecog_activity[feature][['x', 'y', 'z', severity]].dropna(subset=[severity])
    ecog_channel_coordinates = ecog_activity_filtered[['x', 'y', 'z']].values
    ecog_feature_values = ecog_activity_filtered[severity].values

    # Create a mask for points within the radius
    mask = np.zeros(cortex_points.shape[0], dtype=bool)

    for ecog_coord in ecog_channel_coordinates:
        # Calculate distance from the current ECoG coordinate to all cortex points
        distances = np.linalg.norm(cortex_points - ecog_coord, axis=1)
        mask |= (distances < radius)

    # Interpolate only for the points within the mask
    rbf = Rbf(ecog_channel_coordinates[:, 0], ecog_channel_coordinates[:, 1], ecog_channel_coordinates[:, 2], 
              ecog_feature_values, function='gaussian')
    
    # Map ECoG feature values onto the cortical surface
    cortex_mesh[feature] = np.zeros(cortex_points.shape[0])  # Initialize with zeros or any other default value
    cortex_mesh[feature][mask] = rbf(cortex_points[mask, 0], cortex_points[mask, 1], cortex_points[mask, 2])

    return cortex_mesh
    
def map_ecog_activity_to_cortex_by_radius_v2(ecog_activity, cortex_mesh, feature, severity, radius):
    """
    Maps ECoG data onto a brain surface using interpolation,
    limited to areas within a specified radius from ECoG coordinates,
    and scales interpolated values to match the original range.

    Parameters:
    - ecog_activity: DataFrame containing position of ECoG channels in MNI space ['x', 'y', 'z'] and corresponding
                     electrophysiological features observed in these channels in standard frequency bands.
    - cortex_mesh: PyVista PolyData object representing the cortical atlas mesh.
    - feature: The specific feature to map onto the cortex.
    - severity: The column indicating the severity or intensity of the feature values.
    - radius: The maximum distance from ECoG points for interpolation.

    Returns:
    - cortex_mesh: PyVista PolyData object representing the cortical atlas mesh with interpolated feature values.
    """
    # Extract points from cortex mesh
    cortex_points = cortex_mesh.copy().points

    # Extract ECoG points and values
    ecog_activity_filtered   = ecog_activity.copy()[feature][['x', 'y', 'z', severity]].dropna(subset=[severity])
    ecog_channel_coordinates = ecog_activity_filtered[['x', 'y', 'z']].values
    ecog_feature_values      = ecog_activity_filtered[severity].values

    # Create a mask for points within the radius
    mask = np.zeros(cortex_points.shape[0], dtype=bool)

    for ecog_coord in ecog_channel_coordinates:
        # Calculate distance from the current ECoG coordinate to all cortex points
        distances = np.linalg.norm(cortex_points - ecog_coord, axis=1)
        mask |= (distances < radius)

    # Interpolate only for the points within the mask
    rbf = Rbf(ecog_channel_coordinates[:, 0], ecog_channel_coordinates[:, 1], ecog_channel_coordinates[:, 2], 
              ecog_feature_values, function='linear')
    
    # Initialize the feature in the cortex mesh with zeros
    cortex_mesh[feature] = np.zeros(cortex_points.shape[0])  # or any other default value

    # Map ECoG feature values onto the cortical surface for the points within the mask
    interpolated_values        = rbf(cortex_points[mask, 0], cortex_points[mask, 1], cortex_points[mask, 2])
    cortex_mesh[feature][mask] = interpolated_values

    # Get min and max from the original ECoG data
    min_val = ecog_feature_values.min()
    max_val = ecog_feature_values.max()

    # Scale interpolated values to the original min and max
    if interpolated_values.max() != interpolated_values.min():  # Avoid division by zero
        cortex_mesh[feature][mask] = min_val + (interpolated_values - interpolated_values.min()) * (max_val - min_val) / (interpolated_values.max() - interpolated_values.min())
    else:
        cortex_mesh[feature][mask] = np.full(interpolated_values.shape, min_val)  # If all interpolated values are the same

    return cortex_mesh