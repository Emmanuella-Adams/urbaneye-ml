# **UrbanEye Project Plan: Satellite Imagery Building Footprint Detection**

## **1. Research Goal & Scope**
Accurate detection and segmentation of building footprints from high-resolution satellite imagery is critical for urban planning, disaster response, informal settlement mapping, and population density estimation in fast-growing African cities. UrbanEye implements an end-to-end, reproducible deep learning segmentation pipeline using a PyTorch U-Net architecture trained with a hybrid Binary Cross-Entropy (BCE) + Dice Loss function.

---

## **2. Architecture & Pipeline**

```
+------------------------------+
| Satellite RGB Image Tiles    | (sample_data/images/ 256x256)
| Ground Truth Binary Masks    | (sample_data/masks/ 256x256)
+------------------------------+
               |
               v
+------------------------------+
| Data Preprocessing & Augment | (src/data_utils.py)
| - Min-Max Normalization [0,1]| - Random Flips & Rotations
| - Train/Val/Test Random Split| (70% Train, 15% Val, 15% Test)
+------------------------------+
               |
               v
+------------------------------+
| PyTorch U-Net Deep Network   | (src/model_utils.py)
| - Contracting Encoder        | - Double Conv Blocks
| - Expanding Decoder          | - Skip Connections
| - Loss: BCE + Dice Loss      | - Sigmoid Output
+------------------------------+
               |
               v
+------------------------------+
| Geospatial Benchmarking      | (src/eval_utils.py)
| - Quantitative Metrics       | (Mean IoU, F1 Score, Precision, Recall)
| - Visual Prediction Overlay  | (figures/segmentation_results.png)
+------------------------------+
```

---

## **3. Implementation Deliverables**
1. **PyTorch Modular Package:** `src/data_utils.py`, `src/model_utils.py`, `src/eval_utils.py`.
2. **Reproducible Notebook Suite:** `01_data_prep.ipynb`, `02_model_training.ipynb`, `03_evaluation_visuals.ipynb`.
3. **Model Weights Artifact:** `model_weights.pth`.
4. **Visual Figures:** `figures/training_loss_curve.png`, `figures/segmentation_results.png`.

---

## **4. Future Research Work**
* Expand model training to full multispectral imagery (e.g. Sentinel-2 / Landsat 8 NIR bands).
* Vectorization post-processing to convert binary raster masks into ESRI Shapefiles / GeoJSON polygons.