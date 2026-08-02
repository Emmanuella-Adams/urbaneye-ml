import os
import numpy as np
import matplotlib.pyplot as plt
import torch

# ------------------------------
# Semantic Segmentation Metrics
# ------------------------------
def calculate_metrics(y_true, y_pred, threshold=0.5, smooth=1e-6):
    """
    Calculate Mean IoU, F1 Score (Dice), Precision, Recall, and Accuracy for binary segmentation.
    Inputs can be numpy arrays or PyTorch Tensors.
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    y_pred_bin = (y_pred >= threshold).astype(np.float32)
    y_true_bin = (y_true >= threshold).astype(np.float32)

    y_pred_flat = y_pred_bin.ravel()
    y_true_flat = y_true_bin.ravel()

    tp = np.sum((y_true_flat == 1) & (y_pred_flat == 1))
    fp = np.sum((y_true_flat == 0) & (y_pred_flat == 1))
    fn = np.sum((y_true_flat == 1) & (y_pred_flat == 0))
    tn = np.sum((y_true_flat == 0) & (y_pred_flat == 0))

    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    f1_score = (2 * precision * recall) / (precision + recall + smooth)
    mean_iou = (tp + smooth) / (tp + fp + fn + smooth)
    pixel_accuracy = (tp + tn) / (tp + fp + fn + tn + smooth)

    return {
        'mean_iou': float(mean_iou),
        'f1_score': float(f1_score),
        'precision': float(precision),
        'recall': float(recall),
        'pixel_accuracy': float(pixel_accuracy)
    }

# ------------------------------
# Visualization Functions
# ------------------------------
def plot_segmentation_predictions(images, masks, preds, num_samples=3, save_path=None):
    """
    Plot visual comparison panels: Satellite RGB Image | Ground Truth Mask | Predicted Mask | Overlay
    """
    num_samples = min(num_samples, len(images))
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))

    if num_samples == 1:
        axes = np.expand_dims(axes, axis=0)

    for i in range(num_samples):
        img_np = images[i].cpu().numpy().transpose(1, 2, 0)
        mask_np = masks[i].cpu().numpy().squeeze()
        pred_np = (preds[i].cpu().numpy().squeeze() >= 0.5).astype(np.float32)

        # Panel 1: Original Image
        axes[i, 0].imshow(img_np)
        axes[i, 0].set_title(f'Sample {i+1}: Satellite RGB', fontsize=12)
        axes[i, 0].axis('off')

        # Panel 2: Ground Truth
        axes[i, 1].imshow(mask_np, cmap='gray')
        axes[i, 1].set_title('Ground Truth Footprints', fontsize=12)
        axes[i, 1].axis('off')

        # Panel 3: Predicted Mask
        axes[i, 2].imshow(pred_np, cmap='gray')
        axes[i, 2].set_title('U-Net Predicted Mask', fontsize=12)
        axes[i, 2].axis('off')

        # Panel 4: Overlay
        overlay = img_np.copy()
        overlay[pred_np == 1] = [1.0, 0.2, 0.2]  # Highlight predicted footprints in red
        axes[i, 3].imshow(overlay)
        axes[i, 3].set_title('Prediction Overlay', fontsize=12)
        axes[i, 3].axis('off')

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Segmentation prediction comparison saved to {save_path}")
    plt.close()

def plot_training_curves(history, save_path=None):
    """
    Plot and save training/validation loss curves over epochs.
    """
    plt.figure(figsize=(8, 5))
    epochs = range(1, len(history['train_loss']) + 1)
    plt.plot(epochs, history['train_loss'], 'b-o', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'r-s', label='Validation Loss')
    plt.title('U-Net Model Loss (BCE + Dice Loss)', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Training loss curves saved to {save_path}")
    plt.close()