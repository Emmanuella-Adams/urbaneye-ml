# UrbanEye: Building Footprint Detection for Urban Planning Using Satellite Imagery

UrbanEye is a lightweight, fully reproducible computer vision pipeline for detecting and segmenting **building footprints** from satellite imagery. It is designed to support **urban planning, population estimation, geospatial research, and disaster response**, especially for fast-growing African cities.

This project uses sample tiles derived from **Microsoft Building Footprints – Kenya (GeoJSON)** paired with rasterized masks for segmentation. A U-Net with a ResNet encoder is used to produce high-quality pixel-level predictions.

[View Pitch](https://gamma.app/docs/UrbanEye-Building-Footprint-Detection-for-Urban-Planning-n3d72fowb0iu4wo)
---

## Table of Contents
1.  [Project Structure](#1-project-structure)
2.  [Abstract](#2-abstract)
3.  [Dataset](#3-dataset)
4.  [Methodology](#4-methodology)
5.  [Evaluation](#5-evaluation)
6.  [Results](#6-results)
7.  [Limitations](#7-limitations)
8.  [Future Work](#8-future-work)
9.  [How to Run](#9-how-to-run)
10. [References](#10-references)
11. [License](#11-license)

---

## 1. Project Structure

```
/<repo-root>
├── README.md
├── requirements.txt
├── project_plan.md
├── sample_data/
│   ├── images/
│   ├── masks/
│   └── kenya_buildings.geojson
├── notebooks/
│   ├── 01_data_prep.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_evaluation_visuals.ipynb
├── src/
│   ├── data_utils.py
│   ├── model_utils.py
│   └── eval_utils.py
├── figures/
└── reports/
```

---

## 2. Abstract

UrbanEye is a reproducible segmentation pipeline that detects building footprints from satellite images using samples extracted from the **Microsoft Kenya Building Footprints** GeoJSON dataset. A U-Net with a ResNet encoder trains on 256×256 tiles to produce segmentation masks suitable for lightweight geospatial analysis.

**Example Performance Metrics:**
*   Mean IoU: `<MEAN_IOU>`
*   F1 Score: `<F1_SCORE>`

This project aligns with **Google Research Africa’s geospatial research priorities** through practical, data-driven urban analysis.

---

## 3. Dataset

### Source
*   **Dataset:** Microsoft Building Footprints (Kenya)
*   **License:** Open Data Commons ODbL.

### Repository Dataset Contents
This repository contains a small, curated sample (10–30 tiles) for demonstration, including:
*   Satellite image tiles.
*   Corresponding binary masks rasterized from the GeoJSON polygons.
*   **Note:** The full dataset is not included.

### Pipeline Overview
The data preparation process is detailed in `notebooks/01_data_prep.ipynb`:
1.  Clip the Kenya footprints GeoJSON to an area of interest.
2.  Rasterize polygons into binary mask images.
3.  Export paired 256×256 image and mask tiles.

---

## 4. Methodology

### 4.1 Preprocessing
*   Resize images and masks to 256×256 pixels.
*   Normalize image pixel values to the range [0, 1].
*   Masks are binary (0: background, 1: building).
*   Data split: 70% Train, 15% Validation, 15% Test.

### 4.2 Model Architecture
A U-Net model is implemented using `segmentation-models-pytorch`:
*   **Encoder:** ResNet-18 (pretrained on ImageNet).
*   **Decoder:** Standard U-Net decoder.
*   **Activation:** Sigmoid for binary output.
*   **Loss Function:** Binary Cross-Entropy + Dice Loss.
*   **Optimizer:** Adam (learning rate = 1e-4).

### 4.3 Training
Training is conducted in `notebooks/02_model_training.ipynb`.

**Settings:**
*   Epochs: 5–20.
*   Augmentations: Rotations and flips.
*   Callbacks: Early stopping to prevent overfitting.

---

## 5. Evaluation

**Metrics:**
*   Intersection over Union (IoU)
*   F1 Score
*   Precision
*   Recall

**Visualizations:**
Ground truth vs. predicted mask comparisons and prediction overlays are generated and saved in the `/figures/` directory.

---

## 6. Results

| Metric | Score |
|--------|--------|
| Mean IoU | `<MEAN_IOU>` |
| F1 Score | `<F1_SCORE>` |
| Precision | `<PRECISION>` |
| Recall | `<RECALL>` |

### Key Findings
*   Strong detection performance for medium and large rooftops.
*   Weak performance on informal settlements and very small structures.
*   Common errors are caused by shadows, vegetation occlusion, and image-mask misalignment.

---

## 7. Limitations
*   **Scale:** Trained on a small sample, limiting model generalization.
*   **Imagery:** Uses RGB-only (3-channel) satellite imagery.
*   **Output:** Produces raster masks; no direct polygon vectorization.
*   **Data:** Does not utilize multispectral bands available in many satellite sources.

---

## 8. Future Work
*   Train on the full Kenya dataset or other benchmarks like SpaceNet.
*   Incorporate multispectral imagery (e.g., Near-Infrared bands).
*   Implement polygonization to convert masks into vector shapefiles.
*   Deploy a lightweight model for near-real-time field mapping applications.
*   Add more robust augmentations to handle noise and atmospheric conditions.

---

## 9. How to Run

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Execute the Pipeline
Run the notebooks in sequence:
1.  `01_data_prep.ipynb`
2.  `02_model_training.ipynb`
3.  `03_evaluation_visuals.ipynb`

### Inference Example
```python
import torch
import segmentation_models_pytorch as smp

# Load model architecture
model = smp.Unet(
    encoder_name="resnet18",
    encoder_weights=None,
    in_channels=3,
    classes=1,
    activation="sigmoid"
)
# Load trained weights
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()

# ... (proceed with inference on new images)
```

---

## 10. References
*   Microsoft Building Footprints
*   SpaceNet Challenges
*   DeepGlobe Building Detection Challenge

---

## 11. License
*   The **project code** is licensed under the **MIT License**.
*   The underlying **Microsoft Building Footprints dataset** is licensed under the **Open Data Commons Open Database License (ODbL)**.
