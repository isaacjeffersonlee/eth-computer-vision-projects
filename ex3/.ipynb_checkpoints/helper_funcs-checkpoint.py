import os
import yaml
import h5py
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import plotly
import plotly.graph_objs as go


def create_dir(path):
    try:
        os.makedirs(path)
    except FileExistsError:
        # Directory exists
        pass


def load_data(data_path, scene, mode='gray'):
    """
    Returns:
        img_l         ... Left camera image
        img_r         ... Right camera image
        calib_dict    ... Dictionary with camera intrinsics
        calib_points  ... DataFrame with some pairs for calibration
    """
    data_path = Path(data_path) / scene

    fname_l = data_path / 'cam_l.png'
    fname_r = data_path / 'cam_r.png'

    img_l = cv2.imread(str(fname_l))
    img_r = cv2.imread(str(fname_r))

    # Convert images
    if mode == 'gray':
        img_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
        img_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
    elif mode == 'colour':
        img_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2RGB)
        img_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2RGB)
        pass
    else:
        raise ValueError(f"Unexpected mode {mode}. Choose 'gray' or 'colour'")

    img_l = img_l.astype(float) / 255.
    img_r = img_r.astype(float) / 255.

    # Read points for calibration
    calib_points = pd.read_csv(data_path / 'calib_points.csv')

    # Read calibration dictionary
    with open(data_path / 'calib_dict.yml', 'r') as stream:
        try:
            calib_dict = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    return img_l, img_r, calib_dict, calib_points


def test_triangulation(est_calib_dict, val_calib_points, triangulation_fn):
    """
    Validate estimation of focal_length (f), base_line (b) and triangulation function
    
    Args:
        est_calib_dict    ... Calibration dictionary with keys f and b estimated by students
        val_calib_points  ... A set of validation calibration points
        triangulation_fn  ... The triangulation function (takes arguments: xl, xr, y, calib_dict)

    Returns:
        NRMSE between predicted (X, Y, Z) and ground truth (X, Y, Z)
    """

    error = 0.0
    n_points = len(val_calib_points)
    for idx, row in val_calib_points.iterrows():

        # Predict (X, Y, Z)
        xl = row['xl [px]']
        yl = row['yl [px]']
        xr = row['xr [px]']
        yr = row['yr [px]']

        # Predictions should be in [mm]
        xyz_pred = triangulation_fn(xl, xr, yl, est_calib_dict)

        # Get ground truth from calib_points - converted to [mm]
        X = 1000. * row['X [m]']
        Y = 1000. * row['Y [m]']
        Z = 1000. * row['Z [m]']

        xyz_ref = np.stack([X, Y, Z], axis=-1)

        # Compute difference
        mse = np.sum((xyz_pred - xyz_ref)**2)
        error += np.sqrt(mse / np.sum(xyz_ref)**2)

    return error / n_points


def test_ncc(ncc_fn):
    patch = np.array([
        [0, 0, 0, 0],
        [0, 1, -1, 0],
        [0, 0, 0, 0]
    ])

    corr_sol = np.array([[[1.0, -0.5], [-0.5, 1.0]]])
    corr = ncc_fn(patch, patch, 1)

    if np.allclose(corr, corr_sol, rtol=1e-2):
        print("Test of compute_ncc() successful :)")
    else:
        print("ERROR!!! Test of compute_ncc() failed :(")

    print('Here is the computed NCC')
    print(corr)


def plot_correlation(img_left, img_right, corr, col_to_plot, mode):
    """
    Plot the normalized cross-correlation for a given column.
    The column for which NCC is being plotted is marked
    with a red line in the left image.

    Args:
        img_l   (np.array of shape (num_rows, num_cols)): left grayscale image
        img_r   (np.array of shape (num_rows, num_cols)): right grayscale image
        corr    (np.array of shape
                (
                    num_rows - 2*mask_halfwidth,
                    num_cols - 2*mask_halfwidth,
                    num_cols - 2*mask_halfwidth
                ):
                Computed normalized cross-correlation (NCC) between patches
                in the two images.
        col_to_plot: the column in the left image for which to plot the NCC
        mode: Either 'gray' or 'colour'
    """

    # Create copies not to write into originals
    img_l = img_left.copy()
    img_r = img_right.copy()

    # Pad the slice so that it's size is same as the images
    pad_rows = int((img_l.shape[0] - corr.shape[0]) / 2)
    pad_cols = int((img_l.shape[1] - corr.shape[1]) / 2)
    corr = np.pad(corr, (
        (pad_rows, pad_rows),
        (pad_cols, pad_cols),
        (pad_cols, pad_cols)
    ), 'constant', constant_values=0)

    corr_slice = corr[:, col_to_plot, :]

    # Draw line in the left image to denote the column being visualized
    if mode == 'gray':
        img_l = np.dstack([img_l, img_l, img_l])

    img_l[:, col_to_plot, 0] = 1.
    img_l[:, col_to_plot, 1] = 0.
    img_l[:, col_to_plot, 2] = 0.

    plt.ion()
    f, axes_array = plt.subplots(1, 3, figsize=(15, 5))
    axes_array[0].set_title('Left camera image', fontsize=12)
    axes_array[0].imshow(img_l, cmap=plt.cm.gray)

    axes_array[0].tick_params(
        bottom='off', labelbottom='off', left='off', labelleft='off'
    )
    axes_array[1].set_title('Right camera image', fontsize=12)
    axes_array[1].imshow(img_r, cmap=plt.cm.gray)
    axes_array[1].tick_params(
        bottom='off', labelbottom='off', left='off', labelleft='off'
    )

    axes_array[2].set_title('NCC for column marked by red line', fontsize=12)
    axes_array[2].imshow(corr_slice)
    axes_array[2].tick_params(
        bottom='off', labelbottom='off', left='off', labelleft='off'
    )

    plt.show(block=True)


def plot_point_cloud(gray_left, gray_right, points3d):
    """ Visualize the re-constructed point-cloud

        Args:
            gray_left (np.array of shape (num_rows, num_cols)): left grayscale image
            gray_right (np.array of shape (num_rows, num_cols)): right grayscale image
            points3d ((np.array of shape (num_rows - 2*mask_halfwidth, num_cols - 2*mask_halfwidth, 3)):
                3D World co-ordinates for each pixel in the left image (excluding the boundary pixels
                which are ignored during NCC calculation).
        """

    plt.close('all')
    plt.ion()
    f, axes_array = plt.subplots(1, 2, figsize=(15, 6))
    axes_array[0].set_title('Left camera image', fontsize=12)
    axes_array[0].imshow(gray_left, cmap=plt.cm.gray)
    axes_array[0].tick_params(bottom='off', labelbottom='off', left='off', labelleft='off')
    axes_array[1].set_title('Right camera image', fontsize=12)
    axes_array[1].imshow(gray_right, cmap=plt.cm.gray)
    axes_array[1].tick_params(bottom='off', labelbottom='off', left='off', labelleft='off')
    plt.show()

    margin_y = gray_left.shape[0] - points3d.shape[0]
    margin_x = gray_left.shape[1] - points3d.shape[1]

    points3d = points3d[5:-5,5:-5,:]
    colors = []

    if gray_left.ndim == 2:
        # Pick colours for grayscale image
        for r in range(points3d.shape[0]):
            for c in range(points3d.shape[1]):
                col = gray_left[r+margin_y,c+margin_x]
                colors.append('rgb('+str(col)+','+str(col)+','+str(col)+')')
    elif gray_left.ndim == 3:
        # Pick colours for RGB image
        for r in range(points3d.shape[0]):
            for c in range(points3d.shape[1]):
                col = gray_left[r+margin_y,c+margin_x]
                colors.append('rgb('+str(col[0])+','+str(col[1])+','+str(col[2])+')')

    data = [go.Scatter3d(
        x=-1*points3d[:,:,0].flatten(),
        y=-1*points3d[:,:,2].flatten(),
        z=-1*points3d[:,:,1].flatten(),
        mode='markers',
        marker=dict(
            size=1,
            color=colors,
            line=dict(width=0)
        )
    )]
    layout = go.Layout(
        scene=dict(camera=dict(
                    up=dict(x=0, y=0, z=1),
                    center=dict(x=0, y=0, z=0),
                    eye=dict(x=0.1, y=1, z=0.1)
                )
        ),
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=0
        )
    )
    plotly.offline.init_notebook_mode(connected=True)
    fig = go.Figure(data=data, layout=layout)
    return fig




