import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image
from albumentations.pytorch import ToTensorV2
import cv2
import torch.nn.functional as F
import albumentations as A

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


def train_transform(img_size, orig_h, orig_w):

    transforms = []
    if orig_h < img_size and orig_w < img_size:
        transforms.append(A.PadIfNeeded(min_height=img_size, min_width=img_size, 
                                        border_mode=cv2.BORDER_CONSTANT, value=(0, 0, 0)))
    else:
        transforms.append(A.Resize(int(img_size), int(img_size), interpolation=cv2.INTER_NEAREST))

    # transforms.append(A.HorizontalFlip(p=0.5))
    transforms.append(A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)))
    transforms.append(ToTensorV2(p=1.0))

    return A.Compose(transforms, p=1.)
    

# -----------------------------loss functions--------------------------------------
# Focal loss

class Focal_loss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes=1, size_average=True):
        """
        Args:
            alpha: Controls the weighting of hard vs. easy examples.
            gamma: Controls the modulating factor that down-weights easy examples.
            num_classes: Number of classes in the classification problem.
            size_average: Whether to average the loss over the batch or sum it.
        """
        super(Focal_loss, self).__init__()
        self.size_average = size_average
        # Handling alpha values
        if isinstance(alpha, list):
            assert len(alpha) == num_classes
            print(f'Focal loss alpha={alpha}, will assign alpha values for each class')
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha < 1
            print(f'Focal loss alpha={alpha}, will shrink the impact in background')
            # Handling alpha value for background class
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] = alpha
            self.alpha[1:] = 1 - alpha
        self.gamma = gamma
        self.num_classes = num_classes

    def forward(self, preds, labels):
        """
        Calc focal loss
        :param preds: size: [B, C, H, W]:
        :param labels: size:[B, H, W]: 
        :return:
        """
        self.alpha = self.alpha.to(preds.device)

        # Reshape predictions for calculations
        preds = preds.permute(0, 2, 3, 1).contiguous()
        preds = preds.view(-1, preds.size(-1))

        B, H, W = labels.shape
         # Assert shapes to ensure correctness
        assert B * H * W == preds.shape[0]
        assert preds.shape[-1] == self.num_classes
        preds_logsoft = F.log_softmax(preds, dim=1)  # log softmax
        preds_softmax = torch.exp(preds_logsoft)  # softmax

        # Gather predicted probabilities based on ground truth labels
        preds_softmax = preds_softmax.gather(1, labels.view(-1, 1).to(torch.int64))
        preds_logsoft = preds_logsoft.gather(1, labels.view(-1, 1).to(torch.int64))
        # Gather alpha values based on ground truth labels
        alpha = self.alpha.gather(0, labels.view(-1).to(torch.int64))

        # Compute focal loss
        loss = -torch.mul(torch.pow((1 - preds_softmax), self.gamma),
                          preds_logsoft)  # torch.low(1 - preds_softmax) == (1 - pt) ** r
        loss = torch.mul(alpha, loss.t())

        # Calculate mean or sum based on size_average parameter
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss
    

# Dice Loss
    
class DiceLoss(nn.Module):
    def __init__(self, n_classes=1):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(),
                                                                                                  target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


# Intersection over Union (IoU) Loss

class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #intersection is equivalent to True Positive count
        #union is the mutually inclusive area of all labels & predictions 
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection 
        
        IoU = (intersection + smooth)/(union + smooth)
                
        return 1 - IoU