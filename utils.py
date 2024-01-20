import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image


def get_bounding_box(ground_truth_map):
    # The ground truth map is converted to a NumPy array, to perform array operations
    ground_truth_array = np.array(ground_truth_map)

    # Check if there are non-zero values in the mask
    if np.count_nonzero(ground_truth_array) == 0:
          # If there are no non-zero values, return a default bounding box or handle it as needed
          return [0, 0, 1, 1]
    
    # get bounding box from mask
    y_indices, x_indices = np.where(ground_truth_array > 0)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    
    # add perturbation to bounding box coordinates
    H, W = ground_truth_array.shape
    x_min = max(0, x_min - np.random.randint(0, 20))
    x_max = min(W, x_max + np.random.randint(0, 20))
    y_min = max(0, y_min - np.random.randint(0, 20))
    y_max = min(H, y_max + np.random.randint(0, 20))
    bbox = [x_min, y_min, x_max, y_max]

    return bbox


def visualize_img_mask_box(dataset, num_samples_to_visualize):

    random_indices = torch.randperm(len(dataset))[:num_samples_to_visualize]

    for index in random_indices:
        if len(dataset[index]) == 3:
            image, mask, boxes = dataset[index]
            box = boxes.numpy()

        elif len(dataset[index]) == 2:
            image, mask = dataset[index]
        # Convert tensor to numpy array
        # image = image.numpy().astype('uint8')
        mask = mask.numpy().squeeze().astype('uint8')
        

        plt.clf()
        # Plot the image
        plt.imshow(image.permute(1, 2, 0).numpy())
        plt.title("Image with Mask and Bounding Box")

        # Plot the mask
        plt.imshow(mask, alpha=0.3, cmap='viridis')  # Adjust cmap based on your mask values

        
        if len(dataset[index]) == 3:
            x_min, y_min, x_max, y_max = map(int, box)
            width = x_max - x_min
            height = y_max - y_min

            rect = plt.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor='r', facecolor='none')
            plt.gca().add_patch(rect)
            
        plt.show()