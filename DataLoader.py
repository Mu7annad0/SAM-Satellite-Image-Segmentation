import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
import numpy as np
from PIL import Image
from glob import glob
from utils import get_bounding_box, visualize, init_point_sampling, transformation
from cfg import parse_args


class RoadDataset(Dataset):

    def __init__(self, data_root, image_size=512, train=True, box=True, points=True, transformation=transformation):
        """
        Args:
            data_root: The directory path where the dataset is stored or located.
            image_size: Desired size for input images.
            train: Flag indicating if the dataset is for training.
            box: Indicates if bounding box information is included.
            points: Indicates if points is included.
            transformation: modifying the characteristics or representation of data
        """
        self.data_root = data_root
        self.train = train
        self.box = box
        self.points = points
        self.image_size = image_size
        self.transformation = transformation

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

        if mask.max() == 255:
            mask = mask / 255

        h, w, _ = image.shape

        transformer = self.transformation(self.image_size, h, w, self.train)
        augmented = transformer(image=image, mask=mask)
        image = augmented['image']
        mask = augmented['mask']

        # Condition to check if bounding boxes should be included
        if self.box is True:
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


if __name__ == '__main__':

    args = parse_args()

    dataset = RoadDataset(args.train_root, 512, True, box=True, points=20)
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
