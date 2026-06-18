import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import wandb
from point_cloud_classifier.helper import iou_score
from typing import Union
from point_cloud_classifier.loss_function import BCEDiceLoss, BCEDiceWeightedLoss
  
class CarNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.enc1 = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.out = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        b = self.bottleneck(p3)

        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.out(d1)

class Trainer(object):

    def __init__(self, model, lr, epochs, batch_size, optim='AdamW', weight_decay=1e-4, device=torch.device('cuda'), loss:  Union[nn.Module, torch.nn.modules.loss._Loss] = BCEDiceLoss(), use_board: bool = False, lr_adapt: bool = False):
        self.lr = lr
        self.epochs = epochs
        self.model = model.to(device)
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.device = device
        self.use_board = use_board
        self.lr_adapt = lr_adapt

        self.criterion = loss
        self.criterion = self.criterion.to(device)

        if optim.lower() == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optim.lower() == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optim.lower() == 'adamw':
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        if self.lr_adapt:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=2, factor=0.5)

        if self.use_board:
            wandb.init(project="car-segmentation", config={"lr": lr, "epochs": epochs, "batch_size": batch_size, "model": "CarNet", "loss": self.criterion.__class__.__name__})

            self.global_step = 0

    def train_all(self, train_loader, val_loader: torch.utils.data.DataLoader | None = None, save_callback = None):
        best_val_iou: float = -1.0
        for ep in range(self.epochs):

            self.train_one_epoch(train_loader, ep)

            current_lr = self.optimizer.param_groups[0]['lr']

            if val_loader:
                val_loss, val_iou = self.validate(val_loader)

                if self.lr_adapt:
                    self.scheduler.step(val_loss)

                if self.use_board:
                    wandb.log({"val_loss": val_loss,"val_iou": val_iou,"epoch": ep, "learning_rate": current_lr})

                print(f" | Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}")

                if val_iou > best_val_iou:
                    best_val_iou = val_iou
                    if save_callback is not None:
                        save_callback()

        if self.use_board: 
            wandb.finish()
            
    def train_one_epoch(self, dataloader, ep):

        self.model.train()

        total_loss, total_iou, num_batches = 0, 0, 0

        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device).float().unsqueeze(1)
            logits = self.model(x_batch)
            loss = self.criterion(logits, y_batch)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            iou = iou_score(preds, y_batch)

            total_loss += loss.item()
            total_iou += iou.item()
            num_batches += 1

            if self.use_board:
                if self.global_step % 20 == 0:
                    wandb.log({"batch_loss": loss.item(),"batch_iou": iou.item(),"step": self.global_step})

            self.global_step += 1
        
        mean_loss = total_loss / num_batches
        mean_iou = total_iou / num_batches

        print(f"\rEpoch {ep+1}/{self.epochs}, Loss: {mean_loss:.4f}, IoU: {mean_iou:.4f}", end="")

        if self.use_board:
            wandb.log({"epoch_loss": mean_loss,"epoch_iou": mean_iou,"epoch": ep})

    def predict_torch(self, dataloader):

        self.model.eval()
        all_preds = []

        with torch.no_grad():
            for x_batch in dataloader:
                x_batch = x_batch[0].to(self.device)
                logits = self.model(x_batch)
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float() 
                all_preds.append(preds)

        pred_labels = torch.cat(all_preds)
        return pred_labels.cpu().numpy().astype(np.uint8)

    def predict(self, test_data):

        test_tensor = torch.from_numpy(test_data).float().to(self.device)
        test_dataset = TensorDataset(test_tensor)
        
        test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        pred_labels = self.predict_torch(test_dataloader).squeeze(1)

        return pred_labels

    def validate(self, dataloader):

        self.model.eval()

        total_loss = 0
        total_iou = 0
        num_batches = 0

        with torch.no_grad():
            for x_batch, y_batch in dataloader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device).float().unsqueeze(1)

                logits = self.model(x_batch)
                loss = self.criterion(logits, y_batch)

                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()

                iou = iou_score(preds, y_batch)

                total_loss += loss.item()
                total_iou += iou.item()
                num_batches += 1

        return total_loss / num_batches, total_iou / num_batches