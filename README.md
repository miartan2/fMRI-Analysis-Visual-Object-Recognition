# fMRI-Analysis-Visual-Object-Recognition

## Overview
This project explores how the human brain differentiates between faces and houses, using a block-design fMRI dataset.

## Data  
This project uses the built-in Haxby dataset from Nilearn:
Haxby, J. V. et al. (2001).
Distributed and overlapping representations of faces and objects in ventral temporal cortex. Science, 293(5539), 2425–2430.
Nilearn documentation: https://nilearn.github.io
- 1452 fMRI timepoints (TR = 2.5s)
- Visual stimuli including faces, houses, and objects
Load data:
haxby = datasets.fetch_haxby(subjects=[1])
func_img = haxby.func[0]
labels = pd.read_csv(haxby.session_target[0], sep=" ")

## Methods
The script demonstrates three core neuroimaging methods:
1. Preprocessing
  - Brain masking
  - Spatial smoothing
  - Temporal filtering and standardization
2. GLM Analysis
  - Constructing an events table from stimulus labels
  - Fitting a first-level GLM
  - Computing and visualizing the face > house contrast
3. Functional Connectivity
  - Extracting regional time series using the AAL atlas
  - Computing a brain-wide correlation matrix
  - Identifying the most highly correlated regions

## To Run
Required: pip install nilearn numpy scipy pandas matplotlib seaborn
Run script: python "fMRI Face vs House Recognition.py"
