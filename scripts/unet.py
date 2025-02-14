import os
import torch
import torch.optim as optim
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from segmentation_models_pytorch.utils.metrics import IoU, Fscore
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp


class Unet:
    def __init__(self, model_name=None, dataset_dir="./Data", model_path=None):
        if model_name is None:
            model_name = input("Enter the model encoder (resnet50 or vgg16): ")
            if model_name not in ["resnet50", "vgg16"]:
                raise ValueError("Invalid model name. Choose either 'resnet50' or 'vgg16'.")
        if model_path is None:
            model_path = input("Enter the path to the pre-downloaded model: ")
            if not os.path.exists(model_path):
                raise ValueError("Invalid model path. The file does not exist.")
        if model_name is None:
            model_name = input("Enter the model encoder (resnet50 or vgg16): ")
            if model_name not in ["resnet50", "vgg16"]:
                raise ValueError("Invalid model name. Choose either 'resnet50' or 'vgg16'.")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.dataset_dir = dataset_dir
        self.train_images_dir = os.path.join(dataset_dir, "train/images")
        self.train_masks_dir = os.path.join(dataset_dir, "train/masks")
        self.val_images_dir = os.path.join(dataset_dir, "val/images")
        self.val_masks_dir = os.path.join(dataset_dir, "val/masks")

        self.train_transform = A.Compose([
            A.Resize(256, 256),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.Rotate(limit=20, p=0.5),
            A.Normalize(mean=(0,), std=(1,)),
            ToTensorV2()
        ])

        self.val_transform = A.Compose([
            A.Resize(256, 256),
            A.Normalize(mean=(0,), std=(1,)),
            ToTensorV2()
        ])

        self.train_dataset = SegmentationDataset(self.train_images_dir, self.train_masks_dir, transform=self.train_transform)
        self.val_dataset = SegmentationDataset(self.val_images_dir, self.val_masks_dir, transform=self.val_transform)

        self.train_loader = DataLoader(self.train_dataset, batch_size=8, shuffle=True, num_workers=2)
        self.val_loader = DataLoader(self.val_dataset, batch_size=8, shuffle=False, num_workers=2)

        self.model = smp.Unet(
            encoder_name=model_name,
            encoder_weights="imagenet",
            in_channels=1,
            classes=1
        ).to(self.device)

        self.loss_fn = smp.losses.DiceLoss(mode="binary")
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", factor=0.5, patience=3)

    def train(self, num_epochs=20):
        best_val_loss = float("inf")
        for epoch in range(num_epochs):
            self.model.train()
            train_loss = 0

            for images, masks in self.train_loader:
                images, masks = images.to(self.device), masks.float().to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.loss_fn(outputs, masks)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

            train_loss /= len(self.train_loader)
            val_loss = self.evaluate_loss()
            self.scheduler.step(val_loss)
            print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), f"./{self.model.encoder_name}_best.pth")
                print("Saved new best model")

    def evaluate_loss(self):
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, masks in self.val_loader:
                images, masks = images.to(self.device), masks.float().to(self.device)
                outputs = self.model(images)
                loss = self.loss_fn(outputs, masks)
                val_loss += loss.item()
        return val_loss / len(self.val_loader)

    def evaluate_model(self):
        self.model.eval()
        iou_metric = IoU(threshold=0.5)
        dice_metric = Fscore(threshold=0.5, beta=1.0)
        iou_scores = []
        dice_scores = []

        with torch.no_grad():
            for images, masks in self.val_loader:
                images, masks = images.to(self.device), masks.to(self.device)
                masks = masks.float()
                outputs = self.model(images)
                outputs = torch.sigmoid(outputs)
                outputs = (outputs > 0.5).float()
                iou_scores.append(iou_metric(outputs, masks).cpu().numpy().astype(np.float32))
                dice_scores.append(dice_metric(outputs, masks).cpu().numpy().astype(np.float32))

        mean_iou = np.mean(iou_scores)
        mean_dice = np.mean(dice_scores)
        print(f"Model Evaluation Completed!")
        print(f"Mean IoU: {mean_iou:.4f}")
        print(f"Mean Dice Score: {mean_dice:.4f}")
        return mean_iou, mean_dice
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, masks in self.val_loader:
                images, masks = images.to(self.device), masks.float().to(self.device)
                outputs = self.model(images)
                loss = self.loss_fn(outputs, masks)
                val_loss += loss.item()
        return val_loss / len(self.val_loader)

class SegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.images = sorted(os.listdir(images_dir))
        self.masks = sorted(os.listdir(masks_dir))
        self.images = [img for img in self.images if img.replace("_cropped_gray.jpg", "_mask.png") in self.masks]
        self.masks = [mask for mask in self.masks if mask.replace("_mask.png", "_cropped_gray.jpg") in self.images]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.images[idx])
        mask_path = os.path.join(self.masks_dir, self.masks[idx])
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 0).astype(np.uint8)
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
        return image, mask
