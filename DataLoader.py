import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image
from glob import glob
from utils import get_bounding_box, visualize_img_mask_box, train_transform
from cfg import parse_args


class RoadDataset(Dataset):

    def __init__(self, data_root, image_size = 256, train=True, box=True, transform=None):
        """
        Args:
            data_root: The directory path where the dataset is stored or located.
            image_size: Desired size for input images.
            train: Flag indicating if the dataset is for training.
            box: Indicates if bounding box information is included.
            transform: Data transformations to be applied on the training dataset.
        """
        self.data_root = data_root  
        self.train  = train  
        self.transform = transform  
        self.box = box  
        self.image_size = image_size 

        # Initial transformation pipeline for mostley validation and test datset.
        self.initial_transform = A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
        # List of image file paths.
        self.img_names = sorted(glob(self.data_root + '/*_sat.jpg'))
        # Counter for transformed images.
        self.transformed_images = 0

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        img_file_path = self.img_names[index]
        mask_file_name = img_file_path.replace('.jpg', '.png').replace('sat', 'mask')

        image = Image.open(img_file_path).convert('RGB')
        mask = Image.open(mask_file_name).convert('L')

        image, mask = np.asarray(image), np.asarray(mask)

        h, w, _ = image.shape

        # Condition to check if the dataset is used for training
        if self.train:
            if self.transform is True:
                # Obtain training transformation
                transformer = train_transform(self.image_size, h, w)
                augmented = transformer(image=image, mask=mask)
                image = augmented['image']
                mask = augmented['mask']

                # Append the transformed image to the list
                self.transformed_images += 1

        else:
            transformed = self.initial_transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
            # Append the transformed image to the list
            self.transformed_images += 1

        # Condition to check if bounding boxes should be included
        if self.box is True:   
            boxes = get_bounding_box(mask)
            return {
                'image': torch.tensor(image).float(), 
                'mask': torch.tensor(mask).float(), 
                'box': torch.tensor(boxes).float()}
            # This returns the image, mask, and bounding boxes as PyTorch tensors if the dataset includes bounding boxes.
        
        # This returns the image and mask as PyTorch tensors if the dataset does not include bounding boxes.
        return {
            'image': torch.tensor(image).float(), 
            'mask': torch.tensor(mask).float(), 
        }

    def get_num_images(self):
        return self.transformed_images


if __name__ == '__main__':
      
    args = parse_args()

    dataset = RoadDataset(args.train_root, 256, True, box = True, transform=True)
    train_dataloader = DataLoader(dataset, 2, True)

    for i, batch in enumerate(train_dataloader):
        image = batch['image']
        mask = batch['mask']
        if 'box' in batch:
            box = batch['box']

    print(len(dataset))
    print(f'shape of image: {image.shape}')  # [B, 3, 256, 256]
    print(f'shape of mask: {mask.shape}')  # [B, 256, 256]
    print(f'shape of box: {box.shape}')  # [B, 4]
    visualize_img_mask_box(dataset, 3)