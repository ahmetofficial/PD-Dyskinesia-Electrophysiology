import pyvista as pv 
import pyvista
from matplotlib.colors import LinearSegmentedColormap, to_rgb
import pandas as pd
import numpy as np
import sys

import warnings
warnings.filterwarnings("ignore")

# inserting the lib folder to the compiler
sys.path.insert(0, './lib')
sys.path.insert(0, './utils/')
import utils_io


def get_gradient_transparent(mesh, position, color, fade_power=1.75):


    if not hasattr(mesh, "points"):
        raise ValueError("`mesh` must have a `.points` attribute (Nx3 array).")

    valid_positions = {"dorsal", "ventral", "anterior", "posterior", "medial", "lateral", "full"}
    positions = [p.strip().lower() for p in position.split("+")]
    for p in positions:
        if p not in valid_positions:
            raise ValueError(f"Invalid position '{p}'. Must be one of {valid_positions}.")

    # Extract coordinates
    x = mesh.points[:, 0]
    y = mesh.points[:, 1]
    z = mesh.points[:, 2]

    # Normalize safely
    def normalize(arr):
        arr_min, arr_max = arr.min(), arr.max()
        if arr_max == arr_min:
            return np.zeros_like(arr)
        return (arr - arr_min) / (arr_max - arr_min)

    x_norm = normalize(x)
    y_norm = normalize(y)
    z_norm = normalize(z)

    # Start fully opaque
    opacity = np.ones_like(z_norm)

    # Apply each fade progressively (multiplicative blending)
    for p in positions:
        if p == "dorsal":        # bottom → top
            opacity *= z_norm ** fade_power
        elif p == "ventral":     # top → bottom
            opacity *= (1 - z_norm) ** fade_power
        elif p == "anterior":    # left → right
            opacity *= y_norm ** fade_power
        elif p == "posterior":   # right → left
            opacity *= (1 - y_norm) ** fade_power
        elif p == "lateral":     # left → right
            opacity *= x_norm ** fade_power
        elif p == "medial":      # right → left
            opacity *= (1 - x_norm) ** fade_power
        elif p == "full":
            opacity *= 1.0  # no change

    # Normalize opacity again to 0–1 range
    min_opacity, max_opacity = 0.0, 1.0
    opacity = np.clip(opacity, 0, 1)
    opacity = min_opacity + (max_opacity - min_opacity) * opacity

    return to_rgb(color), opacity



###############################################################################################################################################
###############################################################################################################################################
###############################################################################################################################################

# load basal ganglia nuclei meshes
plane           = "xy"


# color codes for basal ganglia nuclei
colors              = {}
colors["stn"]       = "lightgray"
colors["theta"]     = "#ea698bff"
colors["alpha"]     = "#b458c6ff"
colors["beta_low"]  = "#4ec1dfff"
colors["beta_high"] = "#1a659eff"
colors["gamma"]     = "#ff0a54ff"

colors["beta_high"] = "lightgray"
colors["gamma"]     = "lightgray"


# Define bands and their gradient positions
bands               = [("beta_low", "anterior"), ("beta_high", "full"),("gamma", "full")]
camera_positions    = {"xy": (0.0, 0.0, 1.0), "yz": (-0.6, -0.6, 1.0), "xz": (0.0, -1.0, 0.0)}



plotter = pv.Plotter(shape=(1, 3), border=False)

for i, (band, grad_position) in enumerate(bands):
    
    
    cortex_mesh  = utils_io.load_cortical_atlas_meshes()
    cortex_right = cortex_mesh["right_hemisphere"]
    
    if(band!="gamma"):
        color, opacity = get_gradient_transparent(cortex_right, position=grad_position, color=colors[band], fade_power=1.5)
    else:
        color, opacity = get_gradient_transparent(cortex_right, position=grad_position, color=colors[band])
    
    plotter.subplot(0, i)
    
    # Base STN and SM meshes (semi-transparent)
    plotter.add_mesh(cortex_right, color=colors["stn"], opacity=1)
    
    # Gradient overlay
    plotter.add_mesh(cortex_right, color=color, opacity=opacity, show_scalar_bar=False)

    # Set camera for the chosen plane
    plotter.camera_position = camera_positions[plane]

plotter.set_background("white")
plotter.show_axes()
plotter.show()
