#%%
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from matplotlib.colors import Normalize
from matplotlib import colors
from matplotlib.cm import get_cmap
from scipy.interpolate import griddata
import numpy as np
import matplotlib.patches as patches
from matplotlib.path import Path
from shapely.geometry import Point, Polygon, MultiPoint
from matplotlib.cm import ScalarMappable
from scipy.io import savemat
from multiprocessing import Pool, cpu_count
import multiprocessing
#%% Load the pmesh object
# points are the coordinates
# facets are the indices of the points that make up the the interfaces between the polygons
# The regions attribute contains the area (2D) that enclose the polygons.
with open('specifications_2D.pkl', 'rb') as f:
    specifications = pickle.load(f)    

size = specifications["fig_size"]
dpi = specifications["dpi"]
num_samples = specifications["num_samples"]
x = np.linspace(0, specifications["side_length"], size)
y = np.linspace(0, specifications["side_length"], size)
X, Y = np.meshgrid(x,y)

def angle_grid(ii): 
    # %% Load the pmesh object 
    with open(f"{specifications['data_path']}/sample_{ii}/polymesh_{ii}.pkl", "rb") as f:
        pmesh = pickle.load(f)
    with open(f"{specifications['data_path']}/sample_{ii}/trimesh_mat_angle_{ii}.pkl", "rb") as f:
        angles = pickle.load(f)    
    points = np.array(pmesh.points)  
    facets = pmesh.facets  
    regions = pmesh.regions
    # %% Points are the coordinates
    # Facets are the indices of the points that make up the the interfaces between the polygons
    # The regions attribute contains the area (2D) that enclose the polygons
    region_polygons = []
    for region in regions:
        region_points = []
        for idx in region:
            # get the facet from the list of facets
            facet = facets[idx]
            # get the start and end point of the line segment
            start_point = points[facet[0]]
            end_point = points[facet[1]]
            region_points.append(start_point)
            region_points.append(end_point)
        region_points = np.array(region_points)
        # Create a Polygon as the convex hull of the region points
        polygon = MultiPoint(region_points).convex_hull
        region_polygons.append(polygon)
    # %% Plot polygons with angles. This must be compared with the grain orientation plot
    # It makes sure that the ange assignment is correct
    norm = Normalize(vmin=0, vmax=np.pi)
    colormap = plt.cm.bwr
    fig, ax = plt.subplots()
    plt.sca(ax)
    for polygon, angle in zip(region_polygons, angles):
        # Create a patch for the polygon
        patch = patches.Polygon(np.array(polygon.exterior.coords), fill=True)
        # Set the color of the patch using the colormap and the normalized angle
        patch.set_facecolor(colormap(norm(angle)))
        ax.add_patch(patch)
   # %% Create a grid to interpolate the angles
    angle_grid = np.empty((size, size))
    # For each pixel, find the region it belongs to and assign the angle
    for i in range(size):
        for j in range(size):
            point = Point(X[i, j], Y[i, j])
            for region_index, polygon in enumerate(region_polygons):
                if polygon.contains(point):
                    angle_grid[i,j] = angles[region_index]
                    break
               
    # After the loop, set the boundary values
    # Make the first column the same as the second
    angle_grid[:, 0] = angle_grid[:, 1]
    # Make the last column the same as the second last
    angle_grid[:, -1] = angle_grid[:, -2]
    # Make the first row the same as the second
    angle_grid[0, :] = angle_grid[1, :]
    # Make the last row the same as the second last
    angle_grid[-1, :] = angle_grid[-2, :]
    #!!!!!! Save flipped because mesh does not have the origin at the bottom left corner !!!!!!!!!!!!
    # ---------------------------------------------------------------------------------------------
    angle_grid = np.flip(angle_grid,axis=0)
    # ---------------------------------------------------------------------------------------------
    savemat(f"{specifications['data_path']}/sample_{ii}/angle_grid_{ii}.mat", {'angle_grid': angle_grid})
    with open(f"{specifications['data_path']}/sample_{ii}/angle_grid_{ii}.pkl", 'wb') as f:
        pickle.dump(angle_grid, f)
        
    print(f"Angle grid {ii}")

# Get number of available CPUs
num_cpus = cpu_count()
# Create a pool of processes
if __name__ == '__main__':
    num_cpus = multiprocessing.cpu_count()
    with Pool(num_cpus) as p:
        # Map the function to the list of inputs
        results = p.map(angle_grid, range(num_samples))        
