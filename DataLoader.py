import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from glob import glob
from utils import get_bounding_box, visualize, init_point_sampling, transformation
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

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
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
        mask = augmented['mask']

        # Condition to check if bounding boxes should be included
        if self.is_box is True:
            boxes = get_bounding_box(mask)
            boxes = torch.tensor(boxes).float()
        else:
            boxes = None

        if self.points is not None:
            point_coords, point_labels = init_point_sampling(mask, get_point=self.points)
        else:
            point_coords = torch.zeros((0, 2))  # Empty tensor for coordinates
            point_labels = torch.zeros(0, dtype=torch.int)  # Empty tensor for labels

        return {
            'image': image.float(),
            'mask': mask.float(),
            'box': boxes,
            'point_coords': point_coords,
            'point_labels': point_labels
        }


class RoadDataset(BaseDataset):
    def list_image_files(self):
        return sorted(glob(self.data_root + '/*_sat.jpg'))

    def list_mask_files(self):
        return sorted(glob(self.data_root + '/*_mask.png'))


class DubaiDataset(BaseDataset):
    def list_image_files(self):
        return sorted(glob(self.data_root + '/*/images/*'))

    def list_mask_files(self):
        return sorted(glob(self.data_root + '/*/masks/*'))


if __name__ == '__main__':

    args = parse_args()

    dataset = RoadDataset(args.train_root, 512, True, True, points=20)
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
