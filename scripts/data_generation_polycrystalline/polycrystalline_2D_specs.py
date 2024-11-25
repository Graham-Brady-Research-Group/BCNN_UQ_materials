import pickle
from scipy.io import savemat

specifications = {
    "num_samples": 5000,
    "side_length": 10,
    "grain_size": 1,
    "grain_variance": 1,
    "fig_size": 256,
    "fig_path": "../../figures_polycrystalline_2Df",
    "data_path": "../../data_polycrystalline_2Df",
    "dpi": 100,
    "displacement": 0.1,
    "dpi": 100,
    "max_volume": 0.008, 
    "num_seeds":20,
    "seed_radius":0.5,
}
# Print and save specifications vertically 
for key, value in specifications.items():
    print(f'{key}: {value}')
with open('specifications_2D.pkl', 'wb') as f:
    pickle.dump(specifications, f)
savemat('specifications_2D.mat',specifications) 