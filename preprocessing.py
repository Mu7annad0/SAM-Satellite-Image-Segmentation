import numpy as np
import albumentations as A
import cv2
import torch
from albumentations.pytorch import ToTensorV2


class Preprocessing:
    def __init__(self, patch_size: int, is_train: bool = True, augment: bool = False,
                 augmentations: list[A.Compose] = None):
        """
        Args:
            patch_size: The size of the patch.
            is_train: Flag indicating if the dataset is for training.
            augment: Indicates if augmentation is needed.
            augmentations: A list of augmentations to be applied to the images.

        This class is responsible for preprocessing the images and masks.
        """
        self.patch_size = patch_size
        self.is_train = is_train
        self.augment = augment
        self.augmentations = augmentations
        self.initial_transformation = self.get_initial_transformation()
        self.processed_images = 0

    def get_initial_transformation(self, percentage: float = 0.5) -> A.Compose:
        """
        params:
            percentage: The percentage of applying the transformation.

        return:
            A.Compose: The initial transformation.

        This method returns the initial transformation.
        """
        if self.is_train and self.augment:
            # Check if augmentations are provided
            if self.augmentations is not None:
                return A.Compose([
                    *self.augmentations,
                    ToTensorV2()
                ])
            else:
                # Is order of the transforms will impact the result?
                return A.Compose([
                    A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                    A.HorizontalFlip(p=percentage),
                    A.Rotate(p=percentage),
                    ToTensorV2()
                ])
        else:
            return A.Compose([
                A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ToTensorV2()
            ])

    def check_image_size(self, original_width: int, original_height: int) -> bool:
        """
        params:
            original_width: The original width of the image.
            original_height: The original height of the image.

        return:
            A boolean indicating if the image is smaller or greater than the desired size.

        This method checks the size of the image at the specified index.
        """
        if original_width < self.patch_size or original_height < self.patch_size:
            return False
        else:
            return True

    def handle_small_image(self, small_image: np.ndarray, small_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        params:
            small_image: The small image to be padded to be the same size as the image_size of the class.
            small_mask: The small mask to be padded to be the same size as the image_size of the class.

        return:
            A tuple containing the processed image and mask.

        This method handles the size of the small image.
        """
        padding_transformer = A.Compose([
            A.PadIfNeeded(min_height=self.patch_size, min_width=self.patch_size,
                          border_mode=cv2.BORDER_CONSTANT, value=(0, 0, 0))
        ])

        transform_result = padding_transformer(image=small_image, mask=small_mask)

        return transform_result['image'], transform_result['mask']

    def handle_large_image(self, large_image: np.ndarray, large_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        params:
            large_image: The large image to be resized to the image_size of the class.
            large_mask: The large mask to be resized to the image_size of the class.

        return:
            A tuple containing the processed image and mask.

        This method handles the size of the large image.
        """
        resize_transformer = A.Compose([
            A.Resize(self.patch_size, self.patch_size, interpolation=cv2.INTER_NEAREST)
        ])

        transform_result = resize_transformer(image=large_image, mask=large_mask)

        return resize_transformer(image=large_image)['image']

    def process_image(self, image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        params:
            image: The image to be processed.
            mask: The mask to be processed.

        return:
            A tuple containing the processed image and mask.

        This method processes the image and mask based on the preprocessing arguments.
        """
        h, w, _ = image.shape

        # Check if the image is smaller or greater than the desired size
        is_bigger = self.check_image_size(w, h)
        # Handle the smaller image size
        if not is_bigger:
            image, mask = self.handle_small_image(image, mask)
        else:
            # Handle the larger image size
            image, mask = self.handle_large_image(image, mask)

        # Apply the initial transformation
        transformed = self.initial_transformation(image=image, mask=mask)
        image = transformed['image']
        mask = transformed['mask']

        self.processed_images += 1

        return image, mask
