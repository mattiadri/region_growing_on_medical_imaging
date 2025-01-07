# Prostate MRI Segmentation Using Region Growing Algorithm

This repository implements a **Region Growing** algorithm to segment medical images from **Magnetic Resonance Imaging (MRI)** scans, specifically targeting the prostate. The segmentation process leverages the **Region Growing** technique to identify and isolate regions of interest (ROI) within the prostate tissue based on pixel intensity values.

## Overview

The algorithm applies a thresholding method, which was initially estimated on a **training set** of MRI images, and further validated on a **test set**. The training set and corresponding dataset used for validation are sourced from the **PROMISE12** challenge:

**[PROMISE12: Data from the MICCAI Grand Challenge: Prostate MR Image Segmentation 2012](https://promise12.grand-challenge.org/)**

This challenge aimed to compare interactive and (semi)-automatic segmentation algorithms applied to MRI scans of the prostate. The training set used to estimate the threshold for the Region Growing algorithm can be accessed through the above link.

### Threshold Estimation

The threshold value used in the Region Growing algorithm was derived from the **training dataset** and subsequently validated on a **test set**. The estimation process ensures that the algorithm performs robustly across different prostate MRI scans.

### Example Output

Here is an example image illustrating how the Region Growing algorithm segments the prostate from an MRI scan:

![Algorithm Example](inspection_results.png)

## Installation

To run the algorithm locally, clone this repository and install the required dependencies.

```bash
git clone <repository-url>
cd <repository-directory>
pip install -r requirements.txt
```

## Usage

```python
python3 main.py
```
This will exstimate the threshold on training data

## License

See the [Link Text](LICENSE.TXT) file for details.