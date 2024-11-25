import pandas as pd
import matplotlib.pyplot as plt
import os 
from PIL import Image
import numpy as np
from scipy.interpolate import griddata
import pickle
from matplotlib.colors import Normalize
from matplotlib import colors
from matplotlib.cm import ScalarMappable
from matplotlib import gridspec
from io import BytesIO
from skimage.color import rgba2rgb, rgb2gray
from skimage.io import imsave
import imageio
from joblib import Parallel, delayed
import re

def compact_formatter(x, _):
    return f"{x:.2e}"

def plot_stress_field(X, Y, data, label, output_path, specifications, pmesh, compact_formatter):
        dpi = specifications["dpi"]
        size = specifications["fig_size"]
        width, height = size / dpi, size / dpi
        fig = plt.figure(figsize=(width, height), dpi=dpi)
        gs = gridspec.GridSpec(1, 2, width_ratios=[9, 1])
        ax = plt.subplot(gs[0])
        pcm = ax.pcolormesh(X, Y, data, cmap=plt.cm.plasma)
        plt.axis("image")
        # pmesh.plot(edgecolor="k", facecolors="white", alpha=0.1)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        cbar_ax = plt.subplot(gs[1])
        cbar = plt.colorbar(pcm, cax=cbar_ax, format=plt.FuncFormatter(compact_formatter))
        cbar.set_label(label)
        plt.savefig(output_path, bbox_inches='tight', dpi=dpi)
        plt.close(fig)

def save_grayscale_figure(fig, file_path, dpi=100):
    # Save the figure to a buffer
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=dpi)
    buf.seek(0)
    image = Image.open(buf).convert('L')

    # Save the grayscale image to disk
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    image.save(file_path)

    # Close the buffer
    buf.close()

def process_sample(i,specifications):
    # Load data from pmesh.pkl and create pmesh object
    with open(f"{specifications['data_path']}/sample_{i}/polymesh_{i}.pkl", "rb") as f:
        pmesh = pickle.load(f)
    with open(f"{specifications['data_path']}/sample_{i}/trimesh_mat_angle_{i}.pkl", "rb") as f:
        angles = pickle.load(f)
    size = specifications["fig_size"]
    dpi = specifications["dpi"]
    # Plot the grains with orientation
    fig, ax = plt.subplots()
    plt.sca(ax)
    colormap = plt.cm.bwr
    norm = Normalize(vmin=0, vmax=np.pi)
    poly_colors = [colormap(norm(angle)) for angle in angles]
    pmesh.plot(facecolors=poly_colors, edgecolor='none')
    plt.axis("image")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set_frame_on(False)
    ax.margins(x=0)
    # fig.set_size_inches(size/dpi, size/dpi)  
    fig.savefig(f"{specifications['fig_path']}/sample_{i}/polymesh_grains_orientation_{i}_plain.png", bbox_inches='tight',  pad_inches=0, dpi=dpi)
    # Save the figure as one channel grayscale  
    file_path = f"{specifications['fig_path']}/sample_{i}/polymesh_grains_orientation_{i}_gray.png"
    save_grayscale_figure(fig, file_path, dpi=dpi)
    # Add colorbar
    sm = ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array(angles)
    cbar = fig.colorbar(sm, ax=ax)
    # Set the colorbar tick labels
    ticks = np.linspace(0, np.pi, 5)
    ticks = np.linspace(np.min(angles), np.max(angles), 5)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(["0", "π/4", "π/2", "3π/4", "π"])
    # Save the figure
    plt.savefig(f"{specifications['fig_path']}/sample_{i}/polymesh_grains_orientation_{i}.png")
    plt.close(fig)
    
    # Plot the angle grid and make sure it is correct
    with open(f"{specifications['data_path']}/sample_{i}/angle_grid_{i}.pkl", "rb") as f:
        angle_grid = pickle.load(f)
    plt.figure(figsize=(size/dpi,size/dpi),frameon=False)
    plt.imshow(angle_grid, cmap='bwr')
    plt.xticks([]) 
    plt.yticks([]) 
    plt.savefig(f"{specifications['fig_path']}/sample_{i}/polymesh_grains_orientation_{i}_grid.png")
    plt.close(fig)
    
    # Read the abaqus CSV generated files
    df_disp_coords = pd.read_csv(f"{specifications['data_path']}/sample_{i}/coords_disp_{i}.csv")
    df_stress = pd.read_csv(f"{specifications['data_path']}/sample_{i}/stress_int_bound_{i}.csv")
    df_disp = pd.read_csv(f"{specifications['data_path']}/sample_{i}/disp_{i}.csv")
    # Create a grid of x and y values where the stress field will be interpolated
    x = np.linspace(df_stress['X'].min(), df_stress['X'].max(), 1000)
    y = np.linspace(df_stress['Y'].min(), df_stress['Y'].max(), 1000)
    x_u = np.linspace(df_disp_coords['X_U'].min(), df_disp_coords['X_U'].max(), 1000)
    y_u = np.linspace(df_disp_coords['Y_U'].min(), df_disp_coords['Y_U'].max(), 1000)
    X, Y = np.meshgrid(x, y)
    X_U, Y_U = np.meshgrid(x_u, y_u)
    
    # Plot the mesh with the stress at the nodes
    fig, ax = plt.subplots()
    plt.sca(ax)
    pmesh.plot(edgecolor="k", facecolors = "#626567")
    plt.scatter(df_stress["X"], df_stress["Y"], c = df_stress['S11'], cmap="inferno",s=3)
    plt.axis("image")
    cbar = plt.colorbar()
    cbar.set_label(r'$S_{11}$')
    plt.savefig(f"{specifications['fig_path']}/sample_{i}/polymesh_S11_node_{i}.png")
    plt.close(fig)

    # Interpolate stress values at grid points 
    points = np.column_stack((df_stress['X'], df_stress['Y']))
    points_u = np.column_stack((df_disp_coords['X_U'], df_disp_coords['Y_U']))
    S11 = griddata(points, df_stress['S11'], (X, Y), method='linear')
    S22 = griddata(points, df_stress['S22'], (X, Y), method='linear')
    # S33 = griddata(points, df_stress['S33'], (X, Y), method='linear')
    S12 = griddata(points, df_stress['S12'], (X, Y), method='linear')
    Mises = griddata(points, df_stress['Mises'], (X, Y), method='linear')
    u_points = np.column_stack((df_disp_coords['X_U'], df_disp_coords['Y_U']))
    disp = griddata(points_u, np.sqrt(df_disp['U_x']**2 + df_disp['U_y']**2), (X, Y), method='linear')

    # Plot interpolated displacement field
    fig, ax = plt.subplots()
    pcm = ax.pcolormesh(X_U, Y_U, disp, cmap=plt.cm.plasma)
    plt.axis("image")
    pmesh.plot(edgecolor="k", facecolors = "white",alpha=0.1)
    cbar = plt.colorbar(pcm,format=plt.FuncFormatter(compact_formatter))
    cbar.set_label(r'$U$')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    # plt.show()
    plt.savefig(f"{specifications['fig_path']}/sample_{i}/polymesh_disp_{i}.png") 

    plt.close('all')
    
    plot_stress_field(X, Y, S11, r'$S_{11}$', f"{specifications['fig_path']}/sample_{i}/polymesh_S11_{i}.png", specifications, pmesh, compact_formatter)
    plot_stress_field(X, Y, S22, r'$S_{22}$', f"{specifications['fig_path']}/sample_{i}/polymesh_S22_{i}.png", specifications, pmesh, compact_formatter)
    plot_stress_field(X, Y, S12, r'$S_{12}$', f"{specifications['fig_path']}/sample_{i}/polymesh_S12_{i}.png", specifications,pmesh, compact_formatter)
    plot_stress_field(X, Y, Mises, r'$Mises$', f"{specifications['fig_path']}/sample_{i}/polymesh_Mises_{i}.png", specifications, pmesh, compact_formatter)
    
    plt.close('all')

def read_seeds(text):
    seeds = []
    seed = {}
    for line in text.split("\n"):
        if "Geometry: circle" in line:
            if seed:  # If the seed dictionary is not empty, append it to the list
                seeds.append(seed)
            seed = {}  # Start a new seed
        elif "Radius:" in line:
            seed['radius'] = float(re.search(r"Radius: (.*)", line).group(1))
        elif "Center:" in line:
            seed['center'] = tuple(map(float, re.search(r"Center: \((.*), (.*)\)", line).groups()))
        elif "Phase:" in line:
            seed['phase'] = float(re.search(r"Phase: (.*)", line).group(1))
        elif "Breakdown:" in line:
            breakdown_values = tuple(map(float, re.search(r"Breakdown: \(\((.*), (.*), (.*)\)\)", line).groups()))
            seed['breakdown'] = breakdown_values
        elif "Position:" in line:
            seed['position'] = tuple(map(float, re.search(r"Position: \((.*), (.*)\)", line).groups()))
    if seed:  # Add the last seed if it exists
        seeds.append(seed)
    return seeds

def plot_mesh(i):
    with open(f"{specifications['data_path']}/sample_{i}/seeds_{i}.pkl", "rb") as file:
        seeds = pickle.load(file)        
    with open(f"{specifications['data_path']}/sample_{i}/polymesh_{i}.pkl", "rb") as file:
        pmesh = pickle.load(file)
    with open(f"{specifications['data_path']}/sample_{i}/trimesh_{i}.pkl", "rb") as file:
        tmesh = pickle.load(file)
    
    os.makedirs(f"{specifications['fig_path']}/sample_{i}", exist_ok=True)
    seed_colors = ["C" + str(s.phase) for s in seeds]
    seeds.plot(facecolors=seed_colors, edgecolor="k")
    plt.axis("image")
    plt.savefig(f"{specifications['fig_path']}/sample_{i}/seeds_{i}.png")
    plt.clf()

    poly_colors = [seed_colors[n] for n in pmesh.seed_numbers]
    pmesh.plot(facecolors=poly_colors, edgecolor="k")
    plt.axis("image")
    plt.savefig(f"{specifications['fig_path']}/sample_{i}/polymesh_{i}.png")
    plt.clf()

    tri_colors = [seed_colors[n] for n in tmesh.element_attributes]
    tmesh.plot(facecolors=tri_colors, edgecolor="k")
    plt.axis("image")
    plt.savefig(f"{specifications['fig_path']}/sample_{i}/trimesh_{i}.png")
    plt.clf()

def run_plot(i, specifications):
    plot_mesh(i)
    process_sample(i, specifications)

with open('specifications_2D.pkl', 'rb') as f:
    specifications = pickle.load(f)    

if not os.path.exists(specifications["fig_path"]):
    os.makedirs(specifications["fig_path"])

# Parallel plotting version
Parallel(n_jobs=-1)(delayed(run_plot)(i, specifications) for i in range(specifications["num_samples"]))
batch_size = 2
for start_index in range(specifications["num_samples"], batch_size):
    end_index = min(start_index + batch_size, specifications["num_samples"])
    Parallel(n_jobs=-1)(
        delayed(run_plot)(i, specifications) for i in range(start_index, end_index)
    )
    
# # Sequential plotting version
# for i in range(specifications["num_samples"]):
#     run_plot(i, specifications)

images = []
for i in range(specifications["num_samples"]):
    image_path = f"{specifications['fig_path']}/sample_{i}/polymesh_{i}.png"
    images.append(imageio.imread(image_path))
        
output_gif_path = f"{specifications['fig_path']}/polymesh_animation.gif"
imageio.mimsave(output_gif_path, images, duration=0.3)  # Adjust the duration for desired frame rate

images2 = []
for i in range(specifications["num_samples"]):
    image_path = f"{specifications['fig_path']}/sample_{i}/polymesh_S11_{i}.png"
    images2.append(imageio.imread(image_path))
        
output_gif_path = f"{specifications['fig_path']}/S11_animation.gif"
imageio.mimsave(output_gif_path, images2, duration=0.3)  # Adjust the duration for desired frame rate