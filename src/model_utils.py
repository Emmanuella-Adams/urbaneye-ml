import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------
# U-Net Architecture Implementation
# ------------------------------
class DoubleConv(nn.Module):
    """(Convolution => BatchNorm => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    """
    Standard U-Net Architecture for Semantic Segmentation of Satellite Imagery.
    """
    def __init__(self, in_channels=3, out_channels=1, features=[32, 64, 128, 256]):
        super(UNet, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder (Contracting Path)
        curr_in = in_channels
        for feature in features:
            self.downs.append(DoubleConv(curr_in, feature))
            curr_in = feature

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Decoder (Expanding Path)
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feature * 2, feature))

        # Final Classifier Head
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:], mode="bilinear", align_corners=True)

            concat_x = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_x)

        return self.sigmoid(self.final_conv(x))

# ------------------------------
# Loss Function: BCE + Dice Loss
# ------------------------------
class BCEWithDiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(BCEWithDiceLoss, self).__init__()
        self.bce = nn.BCELoss()
        self.smooth = smooth

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice_score = (2. * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)
        dice_loss = 1.0 - dice_score
        return bce_loss + dice_loss

# ------------------------------
# Model Training & Management
# ------------------------------
def create_model(model_type='unet', in_channels=3, out_channels=1):
    """
    Factory function for segmentation model creation.
    """
    if model_type.lower() == 'unet':
        return UNet(in_channels=in_channels, out_channels=out_channels)
    else:
        raise ValueError(f"Model type '{model_type}' not supported.")

def train_unet_model(model, train_loader, val_loader, epochs=10, lr=1e-3, device='cpu'):
    """
    Train U-Net PyTorch model over epochs, logging BCE+Dice loss and Mean IoU.
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = BCEWithDiceLoss()

    history = {'train_loss': [], 'val_loss': []}

    print(f"Starting training U-Net model for {epochs} epochs on device '{device}'...")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item() * images.size(0)

        val_loss /= len(val_loader.dataset)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    return model, history

def save_model(model, filepath='model_weights.pth'):
    """
    Save PyTorch state dict checkpoint.
    """
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    torch.save(model.state_dict(), filepath)
    print(f"Model state dict saved successfully to {filepath}")

def load_model(filepath='model_weights.pth', device='cpu'):
    """
    Load PyTorch state dict checkpoint into UNet architecture.
    """
    model = UNet()
    model.load_state_dict(torch.load(filepath, map_location=device))
    model.to(device)
    model.eval()
    print(f"Model loaded successfully from {filepath}")
    return model