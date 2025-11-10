import pyvista as pv 
import pyvista
import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore")

import utils_MER


colors              = {}
colors["stn"]       = "lightgray"
colors["theta"]     = "#ea698bff"
colors["alpha"]     = "#b458c6ff"
colors["beta_low"]  = "#4ec1dfff"
colors["beta_high"] = "#1a659eff"
colors["gamma"]     = "#ff0a54ff"



STN_meshes     = utils_MER.load_STN_meshes()
STN_mesh       = STN_meshes["right"]["stn"]
STN_SM_mesh    = STN_meshes["right"]["stn_SM"]
    

plotter      = pv.Plotter()
    
# Base STN and SM meshes (semi-transparent)
plotter.add_mesh(STN_mesh, color=colors["stn"], opacity=1)
#plotter.add_mesh(STN_SM_mesh, color=colors["stn"], opacity=1)

plotter.view_xz()

plotter.set_background("white")
plotter.show_axes()
plotter.show()
