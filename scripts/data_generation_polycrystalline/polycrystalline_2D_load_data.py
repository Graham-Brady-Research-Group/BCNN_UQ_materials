
# coding: utf-8

#%% 
# # -*- coding: utf-8 -*-
"""
Created  Wed Feb 22 2023
Modified Wed Feb 22 2023
@author: George
code was initially based on  the keras_unet_segmentation https://github.com/nchlis/keras_UNET_segmentation
"""
# import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import os
from scipy.io import savemat
import imageio
from sklearn.model_selection import train_test_split
import cv2  # cv2.imread() for grayscale images
from mpl_toolkits import axes_grid1
import scipy.io
from joblib import Parallel, delayed
import multiprocessing as mp
from scipy.interpolate import griddata
import pickle
from matplotlib import gridspec
from multiprocessing import Pool, cpu_count
from functools import partial
#%%
def add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
    """
    Add a vertical color bar to an image plot.
    https://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
    """
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1.0 / aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)

def normalize_images(X, Y, channels, Nimages):
    # print(f"Processing image {i}...")
    X_norm = np.zeros_like(X)
    Y_norm = np.zeros_like(Y)
    for i in range(Nimages):
        X_min = np.amin(X[:, :, 0, i])
        X_max = np.amax(X[:, :, 0, i])
        X_norm[:, :, 0, i] = (X[:, :, 0, i] - X_min) / (X_max - X_min)
        for c in channels:
            Y_min = np.amin(Y[:, :, c - 1, i])
            Y_max = np.amax(Y[:, :, c - 1, i])
            Y_norm[:, :, c - 1, i] = (Y[:, :, c - 1, i] -
                                  Y_min) / (Y_max - Y_min)
    return  X_norm, Y_norm

def compact_formatter(x, _):
    return f"{x:.2e}"

def plot_stress_field(X, Y, data, label, specifications, pmesh, compact_formatter):
    dpi = specifications["dpi"]
    size = specifications["fig_size"]
    width, height = size / dpi, size / dpi
    fig = plt.figure(figsize=(width, height), dpi=dpi)
    gs = gridspec.GridSpec(1, 2, width_ratios=[9, 1])
    ax = plt.subplot(gs[0])
    pcm = ax.pcolormesh(X, Y, data, cmap=plt.cm.plasma)
    plt.axis("image")
    pmesh.plot(edgecolor="k", facecolors="white", alpha=0.1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    cbar_ax = plt.subplot(gs[1])
    cbar = plt.colorbar(pcm, cax=cbar_ax, format=plt.FuncFormatter(compact_formatter))
    cbar.set_label(label)
    plt.close(fig)

def plot_nan_values(X, Y, data, title, filename, pmesh, formatter):
    # Create a mask where the data is NaN
    nan_mask = np.isnan(data)

    # Create a new array with NaN values set to 1 and non-NaN values set to 0
    nan_data = np.zeros_like(data)
    nan_data[nan_mask] = 1

    # Call the plot_stress_field function with the new nan_data array
    plot_stress_field(X, Y, nan_data, title, filename, pmesh, formatter)

def process_out(i, specifications):
    df_stress = pd.read_csv(f"{specifications['data_path']}/sample_{i}/stress_int_bound_{i}.csv")
    points = np.column_stack((df_stress['X'], df_stress['Y']))
    size = specifications['fig_size']  # Or wherever you define size
    XX, YY = np.meshgrid(np.linspace(0, specifications['side_length'], size), np.linspace(0, specifications['side_length'], size))
    out_data = np.empty((size, size, 4))
    out_data[:, :, 0] = griddata(points, df_stress['S11'], (XX, YY), method='linear')/1000
    # out_data[:, :, 1] = griddata(points, df_stress['S22'], (XX, YY), method='linear')/1000
    # out_data[:, :, 2] = griddata(points, df_stress['S12'], (XX, YY), method='linear')/1000
    # out_data[:, :, 3] = griddata(points, df_stress['Mises'], (XX, YY), method='linear')/1000
    return out_data
#%% --------------------------------------------------------------------
# Convert the polycrystalline input figure to data file - save them also for book keeping
with open('specifications_2D.pkl', 'rb') as f:
    specifications = pickle.load(f) 
#%%
num_samples = specifications["num_samples"]
size = specifications["fig_size"]
inp_data_all = np.zeros((num_samples, size, size, 1))  # Create a 4-dimensional array
out_data_all = np.zeros((num_samples, size, size, 4))  # Create a 4-dimensional array
dpi = specifications["dpi"]
#%%
for i in range(num_samples):
    with open(f"{specifications['data_path']}/sample_{i}/angle_grid_{i}.pkl", 'rb') as f:
        angle_grid = pickle.load(f)
    inp_data = angle_grid
    inp_data = inp_data/np.pi # normalize the pixel values to the range [0, 1]
    inp_data = np.expand_dims(inp_data, axis=-1)
    inp_data_all[i] = inp_data
#%%
xx = np.linspace(0, specifications["side_length"], size)
yy = np.linspace(0, specifications["side_length"], size)
XX, YY = np.meshgrid(xx, yy)
plt.figure()
plt.imshow(inp_data_all[0,:,:,0], cmap='bwr')
# plt.colorbar(contour)
plt.xticks([]) 
plt.yticks([]) 
plt.margins(x=0)
#%% --------------------------------------------------------------------
# Create the output stresses on a 256x256 or 512x512 grid - save them also for book keeping
delta = 0.2 # this reduces the domain of the stresses field in orde to avoid the boundary effects
out_data_all= np.empty((num_samples, size, size, 1))
S11_all = np.empty((num_samples, size, size))
S22_all = np.empty((num_samples, size, size))
S12_all = np.empty((num_samples, size, size))
Mises_all = np.empty((num_samples, size, size))
# if __name__ == '__main__':
#     with Pool(cpu_count()) as pool:
#         func = partial(process_out, specifications=specifications)
#         out_data_all = pool.map(func, range(num_samples))
for i in range(num_samples):
    print(f"Processing image {i}...") 
    df_stress = pd.read_csv(f"{specifications['data_path']}/sample_{i}/stress_int_bound_{i}.csv")
    nan_values = df_stress.isna()
    # Count the number of NaN values in the DataFrame
    num_nan = nan_values.sum().sum()
    points = np.column_stack((df_stress['X'], df_stress['Y']))
    # Divide each stress component by 1000 to convert from MPa to GPa
    out_data_all[i, :, :, 0] = griddata(points, df_stress['S11'], (XX, YY), method='linear')/1000
    # out_data_all[i, :, :, 1] = griddata(points, df_stress['S22'], (XX, YY), method='linear')/1000
    # out_data_all[i, :, :, 2] = griddata(points, df_stress['S12'], (XX, YY), method='linear')/1000
    # out_data_all[i, :, :, 3] = griddata(points, df_stress['Mises'], (XX, YY), method='linear')/1000
#%% !!!!!! Save flipped because mesh does not have the origin at the bottom left corner !!!!!!!!!!!!
# ---------------------------------------------------------------------------------------------
out_data_all = np.flip(out_data_all, axis=1)
# ---------------------------------------------------------------------------------------------
nan_values = np.isnan(out_data_all)
num_nan = np.sum(nan_values)
print(num_nan)
plt.figure()
plt.imshow(out_data_all[0,:,:,0], cmap='bwr')
#%%
channels = [1, 2, 3, 4]
sam_names = [f"sample{i+1}" for i in range(num_samples)]
X = inp_data_all
Y = out_data_all
#%%
# Split into training and validation sets
Nsamples = num_samples
Ntrain = int(0.85 * Nsamples)
Ntest = int(0.05 * Nsamples)
# The rest of the data will be used for validation
# create the indices for the training images
ix_tr = np.arange(Ntrain)
# create the indices for the testing images
ix_ts = np.arange(Ntrain, Ntrain+Ntest)
# create the indices for the validation images
ix_val = np.arange(Ntrain+Ntest, Nsamples)
assert len(np.intersect1d(ix_tr, ix_val)) == 0
assert len(np.intersect1d(ix_tr, ix_ts)) == 0
assert len(np.intersect1d(ix_val, ix_ts)) == 0
X_tr = X[ix_tr,:,:]; Y_tr = Y[ix_tr,:,:,:]
X_val = X[ix_val,:,:]; Y_val = Y[ix_val,:,:,:]
X_ts = X[ix_ts,:,:]; Y_ts = Y[ix_ts,:,:,:]
#%%
# Create an array of names named wells for the the number of images
# wells = np.array([f"img_{i+1}" for i in range(Nimages)])
# Save the training and validation sample indices of sam_names
fnames_tr = np.array(sam_names)[ix_tr].tolist()
fnames_val = np.array(sam_names)[ix_val].tolist()
fnames_ts = np.array(sam_names)[ix_ts].tolist()
fname_split = (
    ["train"] * len(fnames_tr)
    + ["validation"] * len(fnames_val)
    + ["test"] * len(fnames_ts))
df = pd.DataFrame({"name": fnames_tr + fnames_val +
                  fnames_ts, "split": fname_split})
df.to_csv(f"{specifications['data_path']}/train_val_test_split.csv", index=False)
#%%
# Create an array of names named wells for the the number of images
# wells = np.array([f"img_{i+1}" for i in range(Nimages)])
# Save the training and validation sample indices
fnames_tr = np.array(sam_names)[ix_tr].tolist()
fnames_val = np.array(sam_names)[ix_val].tolist()
fnames_ts = np.array(sam_names)[ix_ts].tolist()
fname_split = (
   ["train"] * len(fnames_tr)
   + ["validation"] * len(fnames_val)
   + ["test"] * len(fnames_ts)
)
df = pd.DataFrame({"img": fnames_tr + fnames_val + fnames_ts, "split": fname_split})
df.to_csv(f"{specifications['data_path']}/training_validation_test_splits.csv", index=False)
#%%
# Save to disk
savemat(f"{specifications['data_path']}/X_tr.mat", {'X_tr':Y_tr[:10,:,:,0]})
savemat(f"{specifications['data_path']}/Y_tr.mat", {'Y_tr': Y_tr[:10,:,:,0]})
np.save(f"{specifications['data_path']}/X_tr.npy", X_tr)
np.save(f"{specifications['data_path']}/X_val.npy", X_val)
np.save(f"{specifications['data_path']}/X_ts.npy", X_ts)
np.save(f"{specifications['data_path']}/Y_tr.npy", Y_tr)
np.save(f"{specifications['data_path']}/Y_val.npy", Y_val)
np.save(f"{specifications['data_path']}/Y_ts.npy", Y_ts)
# %%
