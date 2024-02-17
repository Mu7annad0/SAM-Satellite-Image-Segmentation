import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
import numpy as np
from PIL import Image
from glob import glob

from preprocessing import Preprocessing
from utils import get_bounding_box, visualize, init_point_sampling
from cfg import parse_args


# Generic class to be inherited by the dataset classes
class BaseDataset(Dataset):
    def __init__(self, data_root: str, image_size: int = 256, is_train: bool = True, is_box: bool = True,
                 is_points: bool = False, augment: bool = False, augmentations: list[A.Compose] = None):
        """
        Args:
            data_root: The directory path where the dataset is stored or located.
            image_size: Desired size for input images.
            is_train: Flag indicating if the dataset is for training.
            is_box: Indicates if bounding box information is included.
            is_points: Indicates if point information is included.
            augment: Indicates if augmentation is needed.
        """
        self.data_root = data_root
        self.is_box = is_box
        self.is_points = is_points

        # List of image file paths.
        self.image_paths = self.list_image_files()
        self.mask_paths = self.list_mask_files()

        # Initialize the preprocessor of the dataset
        self.preprocessor = Preprocessing(image_size, is_train, augment, augmentations)

    def list_image_files(self) -> list[str]:
        """
        This method should be implemented by subclasses of BaseDataset class to list the image files in the dataset.
        """
        raise NotImplementedError("Subclasses of BaseDataset should implement list_image_files.")

    def list_mask_files(self) -> list[str]:
        """
        This method should be implemented by subclasses of BaseDataset class to list the mask files in the dataset.
        """
        raise NotImplementedError("Subclasses of BaseDataset should implement list_mask_files.")

    def get_num_processed_images(self) -> object:
        """
        This method returns the number of processed images in the dataset.
        """
        return self.preprocessor.processed_images

    def __getitem__(self, index, image_read_mode: str = 'RGB', mask_read_mode: str = 'L') -> dict:
        """
        Args:
            index: The index of the image to be opened.
            image_read_mode: The mode to read the image.
            mask_read_mode: The mode to read the mask.

        return:
            A tuple containing the image and mask as a numpy arrays.

        This method returns the image and mask at the specified index.
        """
        img_file_path = self.image_paths[index]
        mask_file_path = self.mask_paths[index]

        image_file = Image.open(img_file_path).convert(image_read_mode)
        mask_file = Image.open(mask_file_path).convert(mask_read_mode)

        np_image, np_mask = np.asarray(image_file), np.asarray(mask_file)

        # Apply the dataset preprocessor
        processed_image, processed_mask = self.preprocessor.process_image(np_image, np_mask)

        # Condition to check if bounding boxes should be included
        if self.is_box:
            boxes = get_bounding_box(processed_mask)
            boxes = torch.tensor(boxes).float()
        else:
            boxes = None

        if self.is_points:
            point_coords, point_labels = init_point_sampling(processed_mask, get_point=8)
        else:
            point_coords = torch.zeros((0, 2))  # Empty tensor for coordinates
            point_labels = torch.zeros(0, dtype=torch.int)  # Empty tensor for labels

        return {
            'image': torch.tensor(processed_image).float(),
            'mask': torch.tensor(processed_mask).float(),
            'box': boxes,
            'point_coords': point_coords,
            'point_labels': point_labels
        }

    def __len__(self):
        """
        This method returns the length of the dataset.
        """
        return len(self.image_paths)


class RoadDataset(BaseDataset):
    def list_image_files(self):
        return sorted(glob(self.data_root + '/*_sat.jpg'))

    def list_mask_files(self):
        return sorted(glob(self.data_root + '/*_mask.png'))


if __name__ == '__main__':
    args = parse_args()

    dataset = BaseDataset(args.train_root, 512, True, is_box=True, is_points=True, augment=True)
    train_dataloader = DataLoader(dataset, 5, True)

    for i, batch in enumerate(train_dataloader):
        image = batch['image']
        mask = batch['mask']
        box = batch['box']
        points_coords = batch['point_coords']
        points_labels = batch['point_labels']

    print(len(dataset))
    print(f'shape of image: {image.shape}')  # [B, C, H, W]
    print(f'shape of mask: {mask.shape}')  # [B, H, W]
    print(f'shape of box: {box.shape}')  # [B, 4]
    print(f'shape of point coors: {points_coords.shape}')  # [B, No_points, 2]
    print(f'shape of point labels: {points_labels.shape}')  # [B, No_points]

    visualize(dataset, 3)
