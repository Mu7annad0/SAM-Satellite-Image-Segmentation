import os
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from glob import glob
from utils import get_bounding_box, get_boxes_from_mask, visualize, init_point_sampling, transformation
from cfg import parse_args


class BaseDataset(Dataset):
    def __init__(self, data_root, image_size=512, is_train=True, is_box=True, points=None,
                 transformation=transformation):
        """
        Args:
            data_root: The directory path where the dataset is stored or located.
            image_size: Desired size for input images.
            is_train: Flag indicating if the dataset is for training.
            is_box: Indicates if bounding box information is included.
            points: Number of points 
        """
        self.data_root = data_root
        self.image_size = image_size
        self.is_box = is_box
        self.points = points
        self.is_train = is_train
        self.transformation = transformation

        # List of image file paths.
        self.image_paths = self.list_image_files()
        self.mask_paths = self.list_mask_files()

    def list_image_files(self):
        pass

    def list_mask_files(self):
        pass

    def normalize(self, tensor):
        minFrom= tensor.min()
        maxFrom= tensor.max()
        minTo = 0
        maxTo=1
        return minTo + (maxTo - minTo) * ((tensor - minFrom) / (maxFrom - minFrom))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):

        batch = {}
        img_file_path = self.image_paths[index]
        mask_file_path = self.mask_paths[index]

        image = Image.open(img_file_path).convert('RGB')
        mask = Image.open(mask_file_path).convert('L')

        image, mask = np.asarray(image), np.asarray(mask)

        if mask.max() == 255:
            mask = mask / 255

        h, w, _ = image.shape

        transformer = self.transformation(self.image_size, h, w, self.is_train)
        augmented = transformer(image=image, mask=mask)
        image = augmented['image']
        mask = augmented['mask'].to(torch.int64)

        image = self.normalize(image)
        # mask = self.normalize(mask)

        # Condition to check if bounding boxes should be included
        if self.is_box is True:
            boxes = get_bounding_box(mask)
            boxes = torch.tensor(boxes).float()
            batch['boxes'] = boxes

        if self.points is not None:
            point_coords, point_labels = init_point_sampling(mask, get_point=self.points)
        else:
            point_coords = torch.zeros((0, 2))  # Empty tensor for coordinates
            point_labels = torch.zeros(0, dtype=torch.int)  # Empty tensor for labels

        batch['image'] = image.float()
        batch['mask'] = mask.unsqueeze(0)
        batch['point_coords'] = point_coords
        batch['point_labels'] = point_labels

        return batch


class DubaiDataset(Dataset):
    def __init__(self, data_root, image_size=512, is_train=True, is_box=True, points=None,
                 transformation=transformation):
        """
        Args:
            data_root: The directory path where the dataset is stored or located.
            image_size: Desired size for input images.
            is_train: Flag indicating if the dataset is for training.
            is_box: Indicates if bounding box information is included.
            points: Number of points 
        """
        self.data_root = data_root
        self.image_size = image_size
        self.is_box = is_box
        self.points = points
        self.is_train = is_train
        self.transformation = transformation

        self.BGR_classes = {'Building': [152, 16, 60],
                            'Land': [246, 41, 132],
                            'Road': [228, 193, 110],
                            'Vegetation': [58, 221, 254],
                            'Water': [41, 169, 226],
                            'Unlabeled': [155, 155, 155]}

        # List of image file paths.
        self.image_paths = sorted(glob(self.data_root + '/images/*.jpg'))
        self.mask_paths = sorted(glob(self.data_root + '/masks/*.png'))

    def __len__(self):
        return len(self.image_paths)
    
    def normalize(self, tensor):
        minFrom= tensor.min()
        maxFrom= tensor.max()
        minTo = 0
        maxTo=1
        return minTo + (maxTo - minTo) * ((tensor - minFrom) / (maxFrom - minFrom))
    
    def __getitem__(self, index):

        batch = {}
        img_file_path = self.image_paths[index]
        mask_file_path = self.mask_paths[index]

        image = cv2.imread(img_file_path)
        mask = cv2.imread(mask_file_path)

        new_mask = np.zeros(mask.shape)
        new_mask[mask == self.BGR_classes['Building']] = 1
        new_mask[mask == self.BGR_classes['Land']] = 0
        new_mask[mask == self.BGR_classes['Road']] = 3
        new_mask[mask == self.BGR_classes['Vegetation']] = 4
        new_mask[mask == self.BGR_classes['Water']] = 5
        new_mask[mask == self.BGR_classes['Unlabeled']] = 2

        new_mask = new_mask[:, :, 0]

        if new_mask.max() == 255:
            new_mask = new_mask / 255

        image = np.asarray(image)
        h, w, _ = image.shape

        transformer = self.transformation(self.image_size, h, w, self.is_train)
        augmented = transformer(image=image, mask=new_mask)
        image = augmented['image']
        mask = augmented['mask']

        mask = augmented['mask'].to(torch.int64)

        image = self.normalize(image)
        # mask = self.normalize(mask)

        # Condition to check if bounding boxes should be included
        if self.is_box is True:
            boxes = get_boxes_from_mask(mask, 1)
            batch['boxes'] = boxes

        if self.points is not None:
            # print(mask)
            point_coords, point_labels = init_point_sampling(mask, get_point=self.points)
        else:
            point_coords = torch.zeros((0, 2))  # Empty tensor for coordinates
            point_labels = torch.zeros(0, dtype=torch.int)  # Empty tensor for labels

        batch['image'] = image.float()
        batch['mask'] = mask.unsqueeze(0)
        batch['point_coords'] = point_coords
        batch['point_labels'] = point_labels

        return batch


class RoadDataset(BaseDataset):
    def list_image_files(self):
        return sorted(glob(self.data_root + '/*_sat.jpg'))

    def list_mask_files(self):
        return sorted(glob(self.data_root + '/*_mask.png'))


class xbdDataset(BaseDataset):
    def list_image_files(self):
        return sorted(glob(self.data_root + '/*disaster.png'))

    def list_mask_files(self):
        return sorted(glob(self.data_root + '/*disaster_target.png'))


class AerialDataset(BaseDataset):
    def list_image_files(self):
        return sorted(glob(self.data_root + '/*_image.jpg'))

    def list_mask_files(self):
        return sorted(glob(self.data_root + '/*_mask.png'))


class Nails(BaseDataset):
    def list_image_files(self):
        return sorted(glob(self.data_root + '/images/*.jpg'))

    def list_mask_files(self):
        return sorted(glob(self.data_root + '/labels/*.jpg'))
    
    
if __name__ == '__main__':

    args = parse_args()
    xd_train_root = "../Dataset/nails_segmentation/train/"
    # dataset = Nails(xd_train_root, 512, True, True, points=10)
    # dataset = RoadDataset(args.train_root, 512, True, False, points=10)
    dataset = DubaiDataset(args.dubai_valid_root, 512, True, False, points=10)
    
    train_dataloader = DataLoader(dataset, 1, True)

    for i, batch in enumerate(train_dataloader):
        image = batch['image']
        mask = batch['mask']
        # box = batch['boxes']
        points_coords = batch['point_coords']
        points_labels = batch['point_labels']
        break

    print(len(dataset))
    print(f'shape of image: {image.shape}')  # [B, C, H, W]
    print(f'shape of mask: {mask.shape}')  # [B, H, W]
    # print(f'shape of box: {box.shape}')  # [B, 4]
    print(f'shape of point coors: {points_coords.shape}')  # [B, No_points, 2]
    print(f'shape of point labels: {points_labels.shape}')  # [B, No_points]
    visualize(train_dataloader, 3, True)
