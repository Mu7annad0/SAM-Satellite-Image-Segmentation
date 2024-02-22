import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from glob import glob

from utils import get_bounding_box, visualize, init_point_sampling, get_transformations
from cfg import parse_args


# Generic class to be inherited by the dataset classes
class BaseDataset(Dataset):
    def __init__(self, data_root: str, is_train: bool = True, is_box: bool = True,
                 is_points: bool = False, patch: bool = False):
        """
        Args:
            data_root: The directory path where the dataset is stored or located.
            is_train: Flag indicating if the dataset is for training.
            is_box: Indicates if bounding box information is included.
            is_points: Indicates if point information is included.
            patch: Indicates to patch images in dataset or not.
        """
        self.data_root = data_root
        self.is_box = is_box
        self.is_points = is_points
        self.is_train = is_train
        self.patch = patch

        # List of image file paths.
        self.image_paths = self.list_image_files()
        self.mask_paths = self.list_mask_files()

        # Define image size
        self.image_size = 512

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

        # Normalize mask
        np_mask = np_mask / 255

        h, w, _ = np_image.shape

        # Get the transformations to be applied to the image and mask
        transformer = get_transformations(self.image_size, h, w, self.is_train, self.patch)
        transformed = transformer(image=np_image, mask=np_mask)
        processed_image, processed_mask = transformed['image'], transformed['mask']

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
            'image': processed_image,
            'mask': processed_mask,
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

    dataset = RoadDataset(args.train_root, is_train=True, is_box=True)
    train_dataloader = DataLoader(dataset, 5, True)

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