# -*- coding: utf-8 -*-
"""
fMRI Analysis Project - Haxby Dataset
Basic preprocessing, GLM analysis, and functional connectivity
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from nilearn import datasets, image, masking, plotting
from nilearn.glm.first_level import FirstLevelModel
from nilearn.plotting import plot_stat_map, plot_design_matrix
from nilearn.image import clean_img, smooth_img
from nilearn.maskers import NiftiLabelsMasker

# Setup: create folder for output figures
if not os.path.exists("figures"):
    os.makedirs("figures")

# Load Haxby dataset for subject 1
haxby = datasets.fetch_haxby(subjects=[1])
func_img = haxby.func[0]
labels = pd.read_csv(haxby.session_target[0], sep=" ")

print(f"total timepoints: {len(labels)}")
print(f"stimulus categories: {list(labels['labels'].unique())}")

# Preprocessing
# Brain mask
mask_img = masking.compute_epi_mask(func_img)

# Smoothing with 6mm kernel
smoothed = smooth_img(func_img, fwhm=6)

# Detrending, filtering, standardizing (z-scoring)
cleaned = clean_img(smoothed, t_r=2.5, high_pass=0.01, standardize=True)

# GLM analysis
# Build events dataframe from labels
labels_series = labels["labels"]
events = []
current_label = labels_series.iloc[0]
start_vol = 0

for i, lab in enumerate(labels_series):
    if lab != current_label:
        events.append(dict(
            onset=start_vol * 2.5,
            duration=(i - start_vol) * 2.5,
            trial_type=current_label))
        current_label = lab
        start_vol = i

# Add last block
events.append(dict(
    onset=start_vol * 2.5,
    duration=(len(labels_series) - start_vol) * 2.5,
    trial_type=current_label))

events_df = pd.DataFrame(events)

# Keep only faces and houses for GLM
events_df = events_df[events_df["trial_type"].isin(["face", "house"])].reset_index(drop=True)

print(f"analyzing {len(events_df)} blocks")
print(f"  face blocks: {len(events_df[events_df['trial_type']=='face'])}")
print(f"  house blocks: {len(events_df[events_df['trial_type']=='house'])}")

# Fit GLM
print("\nfitting GLM model...")
flm = FirstLevelModel(
    t_r=2.5,
    hrf_model="spm",
    mask_img=mask_img,
    smoothing_fwhm=None,
    standardize=False)

flm = flm.fit(run_imgs=cleaned, events=events_df)

# Design matrix figure
design_matrix = flm.design_matrices_[0]
plot_design_matrix(design_matrix)
plt.savefig("figures/design_matrix.png", dpi=120)
plt.show()

# Computing face > house contrast
contrast_vec = np.array([
    1 if "face" in col else (-1 if "house" in col else 0)
    for col in design_matrix.columns])

z_map = flm.compute_contrast(contrast_vec, output_type="z_score")

# Plot z map
plot_stat_map(
    z_map,
    bg_img=image.mean_img(func_img),
    threshold=3.0,
    title="face > house (z>3.0)",
    cut_coords=5,
    display_mode="z")
plt.savefig("figures/face_house_zmap.png", dpi=120)
plt.show()

# Functional connectivity
# Load AAL atlas
aal = datasets.fetch_atlas_aal()
atlas_labels = aal.labels
atlas_img = aal.maps

print(f"atlas has {len(atlas_labels)} regions")

# Extract timeseries per region
masker = NiftiLabelsMasker(labels_img=atlas_img, standardize=True, t_r=2.5)
time_series = masker.fit_transform(func_img)

print(f"time series shape: {time_series.shape}")

# Correlation matrix
corr_mat = np.corrcoef(time_series.T)
corr_no_diag = corr_mat.copy()
np.fill_diagonal(corr_no_diag, 0)

# Plot connectivity matrix
plt.figure(figsize=(8, 7))
sns.heatmap(corr_no_diag, vmin=-0.5, vmax=0.5, cmap="RdBu_r",
            xticklabels=False, yticklabels=False,
            cbar_kws={"label": "correlation"})
plt.title("functional connectivity matrix")
plt.tight_layout()
plt.savefig("figures/correlation_matrix.png", dpi=120)
plt.show()

# Top hub regions
region_connectivity = np.sum(np.abs(corr_no_diag), axis=0)
top_indices = np.argsort(region_connectivity)[-10:][::-1]

print("\ntop 10 hub regions:")
for idx, region_idx in enumerate(top_indices, 1):
    print(f"  {idx}. {atlas_labels[region_idx]} ({region_connectivity[region_idx]:.1f})")
