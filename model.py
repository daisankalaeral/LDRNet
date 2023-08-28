import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torchvision
from torchvision.utils import save_image
import lightning as pl
import cv2 as cv
from loss import calculate_total_loss
import configs

class LDRNet(pl.LightningModule):
    def __init__(self, n_points = 100, alpha = 0.3, **kwargs):
        super().__init__()
        self.n_points = n_points

        self.backbone_model = torchvision.models.quantization.mobilenet_v2(weights=torchvision.models.MobileNet_V2_Weights.DEFAULT, quantize = False, **kwargs)
        self.backbone_model.classifier[1] = nn.Linear(self.backbone_model.last_channel, 8)
        
        self.backbone_model.border = nn.Sequential(
            nn.Dropout(kwargs['dropout']),
            nn.Linear(self.backbone_model.last_channel, (n_points - 4) * 2)
        )

    def custom_forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        x = self.backbone_model.features(x)
        # Cannot use "squeeze" as batch-size can be 1
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        corners = self.backbone_model.classifier(x)
        points = self.backbone_model.border(x)
        return corners, points

    def forward(self, inputs):
        x = self.custom_forward_impl(inputs)
        return x
    
    def _common_step(self, batch, which_loss):
        image, corner_coords_true = batch

        corner_coords_pred, border_coords_pred = self(image)

        loss = calculate_total_loss(corner_coords_true, corner_coords_pred, border_coords_pred)

        self.log_dict(
            {
                which_loss: loss
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        return loss

    def training_step(self, batch, batch_idx):
        loss = self._common_step(batch, "train_loss")

        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self._common_step(batch, "valid_loss")

        return loss
    
    def test_step(self, batch, batch_idx):
        loss = self._common_step(batch, "test_loss")

        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.RMSprop(self.parameters(), eps=1e-7, lr = configs.lr)
        # optimizer = torch.optim.Adam(self.parameters(), lr = 1e-4)
        return [optimizer]

if __name__ == '__main__':


    model = LDRNet(100, 1.0, dropout = 0.2)
    x = torch.rand((1,3,128,128))
    print(model(x))