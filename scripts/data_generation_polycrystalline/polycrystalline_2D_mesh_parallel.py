from __future__ import division
import os
import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import distance
import microstructpy as msp
import scipy.stats
import pickle
from multiprocessing import Pool, cpu_count
import imageio.v2 as imageio

def plot_mesh(i,specifications):
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
    
def generate_sample(i,specifications):
    material_1 = {
    "name": "Matrix",
    "material_type": "crystalline",
    "fraction": 1,
    "shape": "circle",
    "size": scipy.stats.uniform(loc=specifications["grain_size"], scale=specifications["grain_variance"]),
    }
    materials = [material_1]
    os.makedirs(f"{specifications['data_path']}/sample_{i}", exist_ok=True)
    os.makedirs(f"{specifications['fig_path']}/sample_{i}", exist_ok=True)  # Add this line to create the figures directory
    domain = msp.geometry.Square(side_length=specifications["side_length"], corner=(0, 0))
    seed_area = domain.area
    # rng_seeds = {"size": np.random.uniform(-1,1)}
    rng_seeds = {"size": np.random.randint(0, 1e6)}
    # seeds = msp.seeding.SeedList.from_info(materials, seed_area, rng_seeds)

    # Create list of seed points
    factory = msp.seeding.Seed.factory
    n = specifications["num_seeds"]
    seeds = msp.seeding.SeedList([factory('circle', r=specifications["seed_radius"]) for i in range(n)])
    # Position seeds according to Mitchell's Best Candidate Algorithm
    np.random.seed(i)
    lims = np.array(domain.limits) * (1 - 1e-5)
    centers = np.zeros((n, 2))
    for ii in range(n):
        f = np.random.rand(ii + 1, 2)
        pts = f * lims[:, 0] + (1 - f) * lims[:, 1]
        try:
            min_dists = distance.cdist(pts, centers[:ii]).min(axis=1)
            ii_max = np.argmax(min_dists)
        except ValueError:  # this is the case when i=0
            ii_max = 0
        centers[ii] = pts[ii_max]
        seeds[ii].position = centers[ii]
    # seeds.position(domain)
    # Create Polygonal Mesh
    pmesh = msp.meshing.PolyMesh.from_seeds(seeds, domain)
    # Create Triangular Mesh
    min_angle = 25
    mesher = "Triangle/Tetgen"
    tmesh = msp.meshing.TriMesh.from_polymesh(pmesh, materials, mesher, min_angle, max_volume=specifications["max_volume"])

    # Save pkl files for later (plotting. etc.)
    with open(f"{specifications['data_path']}/sample_{i}/seeds_{i}.pkl", "wb") as file:
        pickle.dump(seeds, file)

    with open(f"{specifications['data_path']}/sample_{i}/polymesh_{i}.pkl", "wb") as file:
        pickle.dump(pmesh, file)

    with open(f"{specifications['data_path']}/sample_{i}/trimesh_{i}.pkl", "wb") as file:
        pickle.dump(tmesh, file)

    tmesh.write(f"{specifications['data_path']}/sample_{i}/trimesh_{i}.inp", format="abaqus", polymesh=pmesh)
    
    # plot_mesh(i,specifications)
    
with open('specifications_2D.pkl', 'rb') as f:
    specifications = pickle.load(f) 

os.makedirs(specifications['data_path'], exist_ok=True)
with open(f"{specifications['data_path']}/specifications_2D.pkl", "wb") as file:
        pickle.dump(specifications, file)
if __name__ == '__main__':
    with Pool(cpu_count()) as p:
        p.starmap(generate_sample, [(i, specifications) for i in range(specifications["num_samples"])])

# for i in range(specifications["num_samples"]):
#     # generate_sample(i, specifications)
#     plot_mesh(i,specifications)
    
# images = []
# for i in range(specifications["num_samples"]):
#     plot_mesh(i,specifications)
#     image_path = f"{specifications['fig_path']}/sample_{i}/polymesh_{i}.png"
#     images.append(imageio.imread(image_path))
        
# output_gif_path = f"{specifications['fig_path']}/polymesh_animation.gif"
# imageio.mimsave(output_gif_path, images, format='GIF', duration=1) 

