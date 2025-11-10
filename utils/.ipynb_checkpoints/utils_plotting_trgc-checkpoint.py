"""
Utilisation function for plotting
"""

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import griddata
from pyvistaqt import BackgroundPlotter
import pyvista as pv
from scipy.interpolate import Rbf
from scipy.spatial import cKDTree
import os

# inserting the lib folder to the compiler
import sys
sys.path.insert(0, './lib')

from lib_data import DATA_IO
import utils_io, utils_plotting

def plot_gc_connectivity_for_ECOG_channels(df, state, segment, frequency_band, vmin, vmax, cortical_area=None, only_significant=False):

    # plot the average 

    #######################################################################################
    # STEP 1: filter the dataset
    #######################################################################################
    
    df_filtered    = df[(df.state==state) & (df.segment==segment) & (df.direction=="ecog->lfp")].copy()
    # filter by significance / if indicated
    if(only_significant == True): df_filtered = df_filtered[df_filtered[f"pvalue_{frequency_band}"] <= 0.05].copy()

    #######################################################################################
    # STEP 2: get the mean connectivity for each ECOG channel across LFP channels 
    #######################################################################################
    
    # group by patient, state, source channel
    group_cols          = ["patient","state","segment","source_type","source_hemisphere","source_channel"]

    # taking the mean across LFP channels for the ECOG channels
    df_filtered_mean    = df_filtered.groupby(group_cols)[f"gc_{frequency_band}"].mean().reset_index()

    # read ECOG channel coordinates
    MNI_ECoG_channels   = pd.read_pickle(DATA_IO.path_coordinates + "MNI_ECoG_channels.pkl")
    MNI_ECoG_channels.x = MNI_ECoG_channels.x.abs()
    mni_channels        = MNI_ECoG_channels.rename(columns={"hemisphere": "source_hemisphere","channel": "source_channel"})
    mni_channels        = mni_channels[["patient","source_hemisphere","source_channel","x","y","z","AAL3_cortex"]]
    
    # merge to data frame by mapping hemisphere + ECOG hemisphere + ECOG channel
    df_plot_data        = df_filtered_mean.merge(mni_channels, on=["patient", "source_hemisphere", "source_channel"], how="left")

    if(cortical_area is not None):
        df_plot_data        = df_plot_data[df_plot_data.AAL3_cortex==cortical_area]

    #######################################################################################
    # STEP 3: plot the connectivity on the surface of cortex 
    #######################################################################################

    coords                            = df_plot_data[['x', 'y', 'z']].to_numpy()
    values                            = df_plot_data[f"gc_{frequency_band}"].to_numpy()
    
    # load meshes
    cortex_mesh                       = utils_io.load_cortical_atlas_meshes()
    cortex_right                      = cortex_mesh["right_hemisphere"] # we mapped all left hemisphere recordings to right
    
    # cortical mesh vertices
    radius                            = 20
    mesh_vertices                     = cortex_right.points
    
    # RBF interpolation
    rbf                               = Rbf(coords[:,0], coords[:,1], coords[:,2], values, function='multiquadric', smooth=2)
    values_interp                     = rbf(mesh_vertices[:,0], mesh_vertices[:,1], mesh_vertices[:,2])
    
    # mask vertices outside radius
    tree                              = cKDTree(coords)
    distances, _                      = tree.query(mesh_vertices)
    values_interp[distances > radius] = np.nan
    cortex_right[frequency_band]      = values_interp
    cortex_smoothed                   = cortex_right.smooth(n_iter=100, relaxation_factor=0.01)
    
    # Plot
    plotter = BackgroundPlotter()
    plotter.add_mesh(cortex_smoothed, color='white', scalars=frequency_band, cmap='Reds', opacity=1, 
                     clim=(vmin, vmax), scalar_bar_args={"color": "black", "n_labels": 3,"fmt": "%.3f", "width": 0.5, "height": 0.1}, smooth_shading=True)
    
    # Add electrodes
    for coor in coords:
        plotter.add_mesh(pv.Sphere(radius=2, center=[coor[0], coor[1], coor[2]+5]), color='lightgrey', smooth_shading=True)
    
    plotter.background_color = 'white'
    plotter.add_text(f'{state} - {frequency_band} - {segment}', position='upper_right', font_size=16, color='black')
    plotter.view_vector((0,0,1))
    plotter.add_light(pv.Light(position=(0,0,1), color='white', intensity=0.6))

    path = DATA_IO.path_figure + f"granger_causality/{state}/{frequency_band}-{segment}.png"
    plotter.screenshot(path, transparent_background=False)
    plotter.show()

def plot_net_gc_connectivity_for_ECOG_channels(df, state, segment, frequency_band, vmin, vmax, cortical_area=None, only_significant=False):

    # plot the average 

    #######################################################################################
    # STEP 1: filter the dataset
    #######################################################################################
    
    df_filtered    = df[(df.state==state) & (df.segment==segment)].copy()
    # filter by significance / if indicated
    if(only_significant == True): df_filtered = df_filtered[df_filtered[f"pvalue_{frequency_band}"] <= 0.05].copy()

    #######################################################################################
    # STEP 2: get the mean connectivity for each ECOG channel across LFP channels 
    #######################################################################################
    
    # group by patient, state, source channel
    group_cols          = ["patient","state","segment","source_type","source_hemisphere","source_channel"]

    # taking the mean across LFP channels for the ECOG channels
    df_filtered_mean    = df_filtered.groupby(group_cols)[f"gc_{frequency_band}"].mean().reset_index()

    # read ECOG channel coordinates
    MNI_ECoG_channels   = pd.read_pickle(DATA_IO.path_coordinates + "MNI_ECoG_channels.pkl")
    MNI_ECoG_channels.x = MNI_ECoG_channels.x.abs()
    mni_channels        = MNI_ECoG_channels.rename(columns={"hemisphere": "source_hemisphere","channel": "source_channel"})
    mni_channels        = mni_channels[["patient","source_hemisphere","source_channel","x","y","z","AAL3_cortex"]]
    
    # merge to data frame by mapping hemisphere + ECOG hemisphere + ECOG channel
    df_plot_data        = df_filtered_mean.merge(mni_channels, on=["patient", "source_hemisphere", "source_channel"], how="left")

    if(cortical_area is not None):
        df_plot_data        = df_plot_data[df_plot_data.AAL3_cortex==cortical_area]

    #######################################################################################
    # STEP 3: plot the connectivity on the surface of cortex 
    #######################################################################################

    coords                            = df_plot_data[['x', 'y', 'z']].to_numpy()
    values                            = df_plot_data[f"gc_{frequency_band}"].to_numpy()
    
    # load meshes
    cortex_mesh                       = utils_io.load_cortical_atlas_meshes()
    cortex_right                      = cortex_mesh["right_hemisphere"] # we mapped all left hemisphere recordings to right
    
    # cortical mesh vertices
    radius                            = 20
    mesh_vertices                     = cortex_right.points
    
    # RBF interpolation
    rbf                               = Rbf(coords[:,0], coords[:,1], coords[:,2], values, function='multiquadric', smooth=2)
    values_interp                     = rbf(mesh_vertices[:,0], mesh_vertices[:,1], mesh_vertices[:,2])
    
    # mask vertices outside radius
    tree                              = cKDTree(coords)
    distances, _                      = tree.query(mesh_vertices)
    values_interp[distances > radius] = np.nan
    cortex_right[frequency_band]      = values_interp
    cortex_smoothed                   = cortex_right.smooth(n_iter=100, relaxation_factor=0.01)
    
    # Plot
    plotter = BackgroundPlotter()
    plotter.add_mesh(cortex_smoothed, color='white', scalars=frequency_band, cmap='bwr', opacity=1, 
                     clim=(vmin, vmax), scalar_bar_args={"color": "black", "n_labels": 3,"fmt": "%.3f", "width": 0.5, "height": 0.1}, smooth_shading=True)
    
    # Add electrodes
    for coor in coords:
        plotter.add_mesh(pv.Sphere(radius=2, center=[coor[0], coor[1], coor[2]+5]), color='lightgrey', smooth_shading=True)
    
    plotter.background_color = 'white'
    plotter.add_text(f'{state} - {frequency_band} - {segment}', position='upper_right', font_size=16, color='black')
    plotter.view_vector((0,0,1))
    plotter.add_light(pv.Light(position=(0,0,1), color='white', intensity=0.1))

    path = DATA_IO.path_figure + f"granger_causality/net/{state}/{frequency_band}-{segment}.png"
    plotter.screenshot(path, transparent_background=False)
    plotter.show()


def plot_stats_between_states(data, event_segment, reference_group, comparison_group, axis):

    dataset          = data[(data.reference_group==reference_group) & (data.comparison_group==comparison_group)].copy()
    dataset['color'] = np.where(dataset['pvalue_corrected'] > 0.05,'#ccccccff',  # non-significant
                                np.where(dataset['coefficient'] > 0,
                                         '#99e2b4ff',  # significant positive
                                         '#ef3c2dff'   # significant negative
                                        )
                               )
    data_subset      = dataset[dataset.segment==event_segment]
    axis             = sns.scatterplot(data=data_subset, y="frequency_band", x='z_score', 
                                       c=data_subset.color.to_list(), s=15, ax=axis)
    axis.set_xlim([-10,10])
    axis.set_yticklabels(axis.get_yticklabels(), fontsize=utils_plotting.LABEL_SIZE, rotation=0)
    axis.set_title(f"{event_segment} | {comparison_group} - {reference_group}", fontsize=utils_plotting.LABEL_SIZE)
    axis.set_ylabel("")
    utils_plotting.set_axis(axis)

    return axis