import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image
from glob import glob
from utils import get_bounding_box, visualize_img_mask_box


train_transform = A.Compose([
    A.Resize(512, 512),
    A.RandomCrop(width=256, height=256),
    A.HorizontalFlip(p=0.5),
    # A.ImageCompression(quality_lower=50, p=0.4),
    # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])


class RoadDataset(Dataset):

    def __init__(self, data_root, mode="train", transform=None):
        self.data_root = data_root
        self.mode = mode
        self.transform = transform
        self.initial_transform = A.Compose([
            A.Resize(512, 512),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
        self.img_names = sorted(glob(self.data_root + '/*_sat.jpg'))
        # Keep track of transformed images
        self.transformed_images = 0

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        img_file_path = self.img_names[index]
        mask_file_name = img_file_path.replace('.jpg', '.png').replace('sat', 'mask')

        image = Image.open(img_file_path).convert('RGB')
        mask = Image.open(mask_file_name).convert('L')
        image, mask = np.asarray(image), np.asarray(mask)

        if self.transform:
                transformed = self.transform(image=image, mask=mask)
                image = transformed['image']
                mask = transformed['mask']

                # Append the transformed image to the list
                self.transformed_images += 1

        if self.mode in ['valid', 'test']:
            transformed = self.initial_transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
            # Append the transformed image to the list
            self.transformed_images += 1

        boxes = get_bounding_box(mask)

        return image, mask, torch.tensor(boxes).float()

    def get_num_images(self):
        return self.transformed_images


if __name__ == '__main__':
    train_root = "../graduating_project/Dataset/DeepGlobeRoadExtraction/road/train/"

    train_dataset = RoadDataset(train_root, mode="train", transform=train_transform)
    train_dataLoader = DataLoader(train_dataset, batch_size=32, shuffle = True)

    visualize_img_mask_box(train_dataset, 3)

    for batch in train_dataLoader:
        image, mask, box = batch
        print("Image shape:", image.shape)
        print("Mask shape:", mask.shape)
        print("Box shape:", box.shape)
        break