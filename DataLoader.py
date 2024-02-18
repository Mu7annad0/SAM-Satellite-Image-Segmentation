import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image
from glob import glob
from utils import get_bounding_box, visualize, init_point_sampling
from cfg import parse_args
import matplotlib.pyplot as plt


def train_transform(img_size, orig_h, orig_w):

    transforms = []
    if orig_h < img_size and orig_w < img_size:
        transforms.append(A.PadIfNeeded(min_height=img_size, min_width=img_size, 
                                        border_mode=cv2.BORDER_CONSTANT, value=(0, 0, 0)))
    else:
        transforms.append(A.Resize(int(img_size), int(img_size), interpolation=cv2.INTER_NEAREST))

    # transforms.append(A.HorizontalFlip(p=0.5))
    # transforms.append(A.Rotate()),
    transforms.append(A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    transforms.append(ToTensorV2(p=1.0))

    return A.Compose(transforms, p=1.)


class RoadDataset(Dataset):

    def __init__(self, data_root, image_size = 512, train=True, box=True, points=True, transform=None):
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
        self.points = points
        self.image_size = image_size 

        # Initial transformation pipeline for mostley validation and test datset.
        self.initial_transform = A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
        # List of image file paths.
        self.img_names = sorted(glob(self.data_root + '/*_sat.jpg'))

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        img_file_path = self.img_names[index]
        mask_file_name = img_file_path.replace('.jpg', '.png').replace('sat', 'mask')

        image = Image.open(img_file_path).convert('RGB')
        mask = Image.open(mask_file_name).convert('L')        

        image, mask = np.asarray(image), np.asarray(mask)
        # image = ((image - np.mean(image)) / np.std(image))
        
        if mask.max() == 255:
            mask = mask / 255

        h, w, _ = image.shape

        # Condition to check if the dataset is used for training
        if self.train:
            if self.transform is True:
                # Obtain training transformation
                transformer = train_transform(self.image_size, h, w)
                augmented = transformer(image=image, mask=mask)
                image = augmented['image']
                mask = augmented['mask']

        else:
            transformed = self.initial_transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
            # Append the transformed image to the list
            self.transformed_images += 1

        # Condition to check if bounding boxes should be included
        if self.box is True:   
            boxes = get_bounding_box(mask)
            boxes = torch.tensor(boxes).float()
        else:
            boxes = None
        
        if self.points is True:
            point_coords, point_labels = init_point_sampling(mask, get_point=4)
        else:
            point_coords = torch.zeros((0, 2))  # Empty tensor for coordinates
            point_labels = torch.zeros(0, dtype=torch.int)  # Empty tensor for labels

        return {
                'image': torch.tensor(image).float(),
                'mask': torch.tensor(mask).float(),
                'box': boxes,
                'point_coords' : point_coords,
                'point_labels' : point_labels
            }


if __name__ == '__main__':
      
    args = parse_args()

    dataset = RoadDataset(args.train_root, 512, True, box = True, points=True, transform=True)
    train_dataloader = DataLoader(dataset, 3, True)

    for i, batch in enumerate(train_dataloader):
        image = batch['image']
        mask = batch['mask']
        box = batch['box']
        points_coords = batch['point_coords']
        points_labels = batch['point_labels']
        break

    print(len(dataset))
    print(f'shape of image: {image.shape}')  # [B, C, H, W]
    print(f'shape of mask: {mask.shape}')  # [B, H, W]
    print(f'shape of box: {box.shape}')  # [B, 4]
    print(f'shape of point coors: {points_coords.shape}')  # [B, No_points, 2]
    print(f'shape of point labels: {points_labels.shape}')  # [B, No_points]
    visualize(train_dataloader, 3)