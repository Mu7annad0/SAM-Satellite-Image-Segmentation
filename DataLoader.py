import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
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


# Generic class to be inherited by the dataset classes
class BaseDataset(Dataset):
    def __init__(self, data_root: str, image_size: int = 256, is_train: bool = True, is_box: bool = True,
                 is_points: bool = False, is_transform: bool = False):
        """
        Args:
            data_root: The directory path where the dataset is stored or located.
            image_size: Desired size for input images.
            is_train: Flag indicating if the dataset is for training.
            is_box: Indicates if bounding box information is included.
            is_points: Indicates if point information is included.
            is_transform: Data transformations to be applied on the training dataset.
        """
        self.data_root = data_root
        self.image_size = image_size
        self.is_train = is_train
        self.is_transform = is_transform
        self.is_box = is_box
        self.is_points = is_points

        self.initial_transform = A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

        self.transformed_images = 0
        # List of image file paths.
        self.image_paths = self.list_image_files()
        self.mask_paths = self.list_mask_files()

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

    def get_num_transformed_images(self) -> object:
        """
        This method returns the number of transformed images in the dataset.
        """
        return self.transformed_images

    def __getitem__(self, index, image_read_mode: str = 'RGB', mask_read_mode: str = 'L') -> tuple[
        np.ndarray, np.ndarray]:
        """
        This method returns the image and mask at the specified index.
        """
        img_file_path = self.image_paths[index]
        mask_file_path = self.mask_paths[index]

        image_file = Image.open(img_file_path).convert(image_read_mode)
        mask_file = Image.open(mask_file_path).convert(mask_read_mode)

        np_image, np_mask = np.asarray(image_file), np.asarray(mask_file)
        return np_image, np_mask

    def process_image(self, index: int) -> dict:
        """
        This method should be implemented by subclasses of BaseDataset class to processes the image and mask.
        """
        raise NotImplementedError("Subclasses of BaseDataset should implement process_image.")

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

    def process_image(self, index) -> dict:
        image, mask = self.__getitem__(index)
        mask = mask / 255
        h, w, _ = image.shape

        if self.is_train:
            if self.is_transform:
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
        if self.is_box:
            boxes = get_bounding_box(mask)
            boxes = torch.tensor(boxes).float()
        else:
            boxes = None

        if self.is_points:
            point_coords, point_labels = init_point_sampling(mask, get_point=8)
        else:
            point_coords = torch.zeros((0, 2))  # Empty tensor for coordinates
            point_labels = torch.zeros(0, dtype=torch.int)  # Empty tensor for labels

        return {
            'image': torch.tensor(image).float(),
            'mask': torch.tensor(mask).float(),
            'box': boxes,
            'point_coords': point_coords,
            'point_labels': point_labels
        }


if __name__ == '__main__':
    args = parse_args()

    dataset = BaseDataset(args.train_root, 512, True, is_box=True, is_points=True, is_transform=True)
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
