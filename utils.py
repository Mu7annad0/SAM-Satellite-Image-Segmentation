import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import scipy.ndimage
import albumentations as A
import torch.nn as nn
import multiprocessing as mp
from torchvision.transforms.functional import to_pil_image
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from time import time
from skimage.measure import label, regionprops
from PIL import Image


def init_point_sampling2(mask, get_point=1):
    """
    Initialization samples points from the mask and assigns labels to them.
    Args:
        mask (torch.Tensor): Input mask tensor.
        num_points (int): Number of points to sample. Default is 1.
    Returns:
        coords (torch.Tensor): Tensor containing the sampled points' coordinates (x, y).
        labels (torch.Tensor): Tensor containing the corresponding labels (0 for background, 1 for foreground).
    """
    if isinstance(mask, torch.Tensor):
        mask = mask.numpy()

    # Get coordinates of black/white pixels
    fg_coords = np.argwhere(mask == 1)[:, ::-1]
    bg_coords = np.argwhere(mask == 0)[:, ::-1]

    fg_size = len(fg_coords)
    bg_size = len(bg_coords)

    if get_point == 1:
        if fg_size > 0:
            index = np.random.randint(fg_size)
            fg_coord = fg_coords[index]
            label = 1
        else:
            index = np.random.randint(bg_size)
            fg_coord = bg_coords[index]
            label = 0
        return torch.as_tensor([fg_coord.tolist()], dtype=torch.float), torch.as_tensor([label], dtype=torch.int)

    else:
        num_fg = get_point // 2
        num_bg = get_point - num_fg
        fg_indices = np.random.choice(fg_size, size=num_fg, replace=True)
        bg_indices = np.random.choice(bg_size, size=num_bg, replace=True)
        fg_coords = fg_coords[fg_indices]
        bg_coords = bg_coords[bg_indices]
        coords = np.concatenate([fg_coords, bg_coords], axis=0)
        labels = np.concatenate([np.ones(num_fg), np.zeros(num_bg)]).astype(int)
        indices = np.random.permutation(get_point)
        coords, labels = torch.as_tensor(coords[indices], dtype=torch.float), torch.as_tensor(labels[indices],
                                                                                              dtype=torch.int)
        return coords, labels
    

def init_point_sampling(mask, get_point=1):
    """
    Initialization samples points from the mask and assigns labels to them.
    Args:
        mask (torch.Tensor): Input mask tensor.
        get_point (int): Number of points to sample. Default is 1.
    Returns:
        coords (torch.Tensor): Tensor containing the sampled points' coordinates (x, y).
        labels (torch.Tensor): Tensor containing the corresponding labels (0 for background, 1 for foreground).
    """
    if isinstance(mask, torch.Tensor):
        mask = mask.numpy()

    # Find connected components (regions) in the foreground
    labels, num_regions = scipy.ndimage.label(mask)

    num_fg = min(get_point, num_regions)
    num_bg = get_point - num_fg

    # Sample one point per region (if possible)
    if num_fg > 0:
        fg_coords = []
        for label in range(1, num_regions + 1):
            # Get random coordinates from the current region
            region_mask = labels == label
            region_coords = np.argwhere(region_mask)[:, ::-1]  # Get x, y
            if len(region_coords) > 0:
                index = np.random.randint(len(region_coords))
                fg_coords.append(region_coords[index].tolist())
    else:
        fg_coords = []  # Empty list if no foreground regions

    # Sample remaining points from background
    bg_coords = np.argwhere(mask == 0)[:, ::-1]
    bg_indices = np.random.choice(len(bg_coords), size=num_bg, replace=True)
    bg_coords = bg_coords[bg_indices]

    # Handle empty fg_coords gracefully to avoid dimension mismatch
    if not fg_coords:  # Check if fg_coords is empty
        coords = bg_coords  # If empty, use only background coordinates
    else:
        # Convert fg_coords to a writable NumPy array with at least 2 dimensions
        fg_coords = np.asarray(fg_coords)
        coords = np.concatenate([fg_coords, bg_coords], axis=0)

    labels = np.concatenate([np.ones(num_fg), np.zeros(num_bg)]).astype(int)

    indices = np.random.permutation(get_point)
    coords, labels = torch.as_tensor(coords[indices], dtype=torch.float), torch.as_tensor(labels[indices],
                                                                                          dtype=torch.int)

    return coords, labels


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


def transformation(img_size, orig_h, orig_w, train=True):
    transforms = []
    if orig_h < img_size and orig_w < img_size:
        transforms.append(A.PadIfNeeded(min_height=img_size, min_width=img_size,
                                        border_mode=cv2.BORDER_CONSTANT, value=(0, 0, 0)))
    else:
        transforms.append(A.Resize(int(img_size), int(img_size), interpolation=cv2.INTER_NEAREST))

    if train:
        transforms.append(A.HorizontalFlip(p=0.5))
        transforms.append(A.Rotate()),

    # transforms.append(A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    transforms.append(ToTensorV2(p=1.0))

    return A.Compose(transforms, p=1.)


# ------------------------------------------visualization-------------------------------------------------


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.5])], axis=0)
    else:
        color = np.array([255 / 255, 0 / 255, 0 / 255, .4])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='#57d459', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='#FF8000', facecolor=(0, 0, 0, 0), lw=2))


def visualize_from_path(image_path, mask_path, box=True, points=True):
    plt.figure(figsize=(10, 10))
    image = Image.open(image_path).convert('RGB')
    image = np.array(image)
    mask = Image.open(mask_path).convert('L')
    mask = np.array(mask)
    if mask.max() == 255:
        mask = mask / 255

    plt.imshow(image)
    show_mask(mask, plt.gca())
    if box:
        box = np.array(get_bounding_box(mask))
        show_box(box, plt.gca())

    if points:
        points_coords, point_labels = init_point_sampling(mask, 8)
        point_coords = np.array(points_coords)
        point_labels = np.array(point_labels)
        show_points(point_coords, point_labels, plt.gca())

    plt.axis('on')
    plt.show()


def visualize(dataloader, num_images, boxes=False):
    for batch in dataloader:
        images = batch['image']
        masks = batch['mask']
        # box = batch['boxes']
        points_coords = batch['point_coords']
        point_labels = batch['point_labels']

        # Iterate over images and masks in the batch
        for image, mask, points_coords, point_labels in zip(images, masks, points_coords, point_labels):
            image_pil = to_pil_image(image)
            # box = np.array((box))
            # plt.figure(figsize=(10, 10))
            plt.imshow(image_pil)
            show_mask(mask, plt.gca())
            # if boxes is True:
            #     for box in box:
            #         show_box(box.cpu().numpy(), plt.gca())
            # else:
            #     show_box(box, plt.gca())
            point_coords = np.array(points_coords)
            point_labels = np.array(point_labels)
            show_points(point_coords, point_labels, plt.gca())
            plt.axis('on')
            plt.show()
            num_images -= 1  # Decrement the count of images
            if num_images == 0:
                return


# ------------------------------------------loss functions------------------------------------------------


# Focal loss

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred, mask):
        """
        pred: [B, 1, H, W]
        mask: [B, 1, H, W]
        """
        assert pred.shape == mask.shape, "pred and mask should have the same shape."
        p = torch.sigmoid(pred)
        num_pos = torch.sum(mask)
        num_neg = mask.numel() - num_pos
        w_pos = (1 - p) ** self.gamma
        w_neg = p ** self.gamma

        loss_pos = -self.alpha * mask * w_pos * torch.log(p + 1e-12)
        loss_neg = -(1 - self.alpha) * (1 - mask) * w_neg * torch.log(1 - p + 1e-12)

        loss = (torch.sum(loss_pos) + torch.sum(loss_neg)) / (num_pos + num_neg + 1e-12)

        return loss


# Dice Loss

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, mask):
        """
        pred: [B, 1, H, W]
        mask: [B, 1, H, W]
        """
        assert pred.shape == mask.shape, "pred and mask should have the same shape."
        p = torch.sigmoid(pred)
        intersection = torch.sum(p * mask)
        union = torch.sum(p) + torch.sum(mask)
        dice_loss = (2.0 * intersection + self.smooth) / (union + self.smooth)

        return 1 - dice_loss


# Intersection over Union (IoU) Loss

class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # intersection is equivalent to True Positive count
        # union is the mutually inclusive area of all labels & predictions
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection

        IoU = (intersection + smooth) / (union + smooth)

        return 1 - IoU


class MaskIoULoss(nn.Module):

    def __init__(self, ):
        super(MaskIoULoss, self).__init__()

    def forward(self, pred_mask, ground_truth_mask, pred_iou):
        """
        pred_mask: [B, 1, H, W]
        ground_truth_mask: [B, 1, H, W]
        pred_iou: [B, 1]
        """
        assert pred_mask.shape == ground_truth_mask.shape, "pred_mask and ground_truth_mask should have the same shape."

        p = torch.sigmoid(pred_mask)
        intersection = torch.sum(p * ground_truth_mask)
        union = torch.sum(p) + torch.sum(ground_truth_mask) - intersection
        iou = (intersection + 1e-7) / (union + 1e-7)
        iou_loss = torch.mean((iou - pred_iou) ** 2)
        return iou_loss


# FocalDiceloss_IoULoss
class FocalDiceloss_IoULoss(nn.Module):

    def __init__(self, weight=20.0, iou_scale=1.0):
        super(FocalDiceloss_IoULoss, self).__init__()
        self.weight = weight
        self.iou_scale = iou_scale
        self.focal_loss = FocalLoss()
        self.dice_loss = DiceLoss()
        self.maskiou_loss = MaskIoULoss()

    def forward(self, pred, mask, pred_iou):
        """
        pred: [B, 1, H, W]
        mask: [B, 1, H, W]
        """
        assert pred.shape == mask.shape, "pred and mask should have the same shape."

        focal_loss = self.focal_loss(pred, mask)
        dice_loss = self.dice_loss(pred, mask)
        loss1 = self.weight * focal_loss + dice_loss
        loss2 = self.maskiou_loss(pred, mask, pred_iou)
        loss = loss1 + loss2 * self.iou_scale
        return loss


# -------------------------------------------Metrics------------------------------------------------

def _threshold(x, threshold=None):
    if threshold is not None:
        return (x > threshold).type(x.dtype)
    else:
        return x


def _list_tensor(x, y):
    m = torch.nn.Sigmoid()
    if type(x) is list:
        x = torch.tensor(np.array(x))
        y = torch.tensor(np.array(y))
        if x.min() < 0:
            x = m(x)
    else:
        x, y = x, y
        if x.min() < 0:
            x = m(x)
    return x, y


def iou(pr, gt, eps=1e-7, threshold=0.5):
    pr_, gt_ = _list_tensor(pr, gt)
    pr_ = _threshold(pr_, threshold=threshold)
    gt_ = _threshold(gt_, threshold=threshold)
    intersection = torch.sum(gt_ * pr_, dim=[1, 2, 3])
    union = torch.sum(gt_, dim=[1, 2, 3]) + torch.sum(pr_, dim=[1, 2, 3]) - intersection
    return ((intersection + eps) / (union + eps)).cpu().numpy()


def dice(pr, gt, eps=1e-7, threshold=0.5):
    pr_, gt_ = _list_tensor(pr, gt)
    pr_ = _threshold(pr_, threshold=threshold)
    gt_ = _threshold(gt_, threshold=threshold)
    intersection = torch.sum(gt_ * pr_, dim=[1, 2, 3])
    union = torch.sum(gt_, dim=[1, 2, 3]) + torch.sum(pr_, dim=[1, 2, 3])
    return ((2. * intersection + eps) / (union + eps)).cpu().numpy()


def SegMetrics(pred, label, metrics):
    metric_list = []
    if isinstance(metrics, str):
        metrics = [metrics, ]
    for i, metric in enumerate(metrics):
        if not isinstance(metric, str):
            continue
        elif metric == 'iou':
            metric_list.append(np.mean(iou(pred, label)))
        elif metric == 'dice':
            metric_list.append(np.mean(dice(pred, label)))
        else:
            raise ValueError('metric %s not recognized' % metric)
    if pred is not None:
        metric = np.array(metric_list)
    else:
        raise ValueError('metric mistakes in calculations')
    return metric


class EarlyStopping:

    def __init__(self, patience=7, delta=0):
        """
        Early stops the training if validation loss doesn't improve after a given patience.
        Args:
            patience (int): How long to wait after last time validation loss improved.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, val_loss):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score

        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')

            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


def save_checkpoint(states, is_best, output_dir, filename):
    if is_best:
        torch.save(states, os.path.join(output_dir, 'checkpoint_best.pth'))
    torch.save(states, os.path.join(output_dir, filename))


def num_workers(dataset):
    for num_workers in range(0, mp.cpu_count(), 2):
        train_loader = DataLoader(dataset, shuffle=True, num_workers=num_workers, batch_size=16, pin_memory=True)
        start = time()
        for epoch in range(1, 3):
            for i, data in enumerate(train_loader, 0):
                pass
        end = time()
        print("Finish with:{} second, num_workers={}".format(end - start, num_workers))


def to_device(batch_input, device):
    device_input = {
        key: (
            value.float().to(device)
            if key in ('image', 'label') and value is not None
            else value.to(device) if value is not None and not isinstance(value, (list, torch.Size))
            else value
        )
        for key, value in batch_input.items()
    }
    return device_input


def get_boxes_from_mask(mask, box_num=1, std=0.1, max_pixel=5):
    """
    Args:
        mask: Mask, can be a torch.Tensor or a numpy array of binary mask.
        box_num: Number of bounding boxes, default is 1.
        std: Standard deviation of the noise, default is 0.1.
        max_pixel: Maximum noise pixel value, default is 5.
    Returns:
        noise_boxes: Bounding boxes after noise perturbation, returned as a torch.Tensor.
    """
    if isinstance(mask, torch.Tensor):
        mask = mask.numpy()

    label_img = label(mask)
    regions = regionprops(label_img)

    # Iterate through all regions and get the bounding box coordinates
    boxes = [tuple(region.bbox) for region in regions]

    # If the generated number of boxes is greater than the number of categories,
    # sort them by region area and select the top n regions
    if len(boxes) >= box_num:
        sorted_regions = sorted(regions, key=lambda x: x.area, reverse=True)[:box_num]
        boxes = [tuple(region.bbox) for region in sorted_regions]

    # If the generated number of boxes is less than the number of categories,
    # duplicate the existing boxes
    elif len(boxes) < box_num:
        num_duplicates = box_num - len(boxes)
        boxes += [boxes[i % len(boxes)] for i in range(num_duplicates)]

    # Perturb each bounding box with noise
    noise_boxes = []
    for box in boxes:
        y0, x0, y1, x1 = box
        width, height = abs(x1 - x0), abs(y1 - y0)
        # Calculate the standard deviation and maximum noise value
        noise_std = min(width, height) * std
        max_noise = min(max_pixel, int(noise_std * 5))

        # Ensure positive range for noise (corrected line)
        noise_range = (0, 2 * max_noise + 1)  # 0 is included, max_noise + 1 is excluded

        # Add random noise to each coordinate
        noise_x = np.random.randint(*noise_range)  # Use unpacking for range
        noise_y = np.random.randint(*noise_range)
        x0, y0 = x0 + noise_x - max_noise, y0 + noise_y - max_noise  # Subtract to center around box
        x1, y1 = x1 + noise_x - max_noise, y1 + noise_y - max_noise
        noise_boxes.append((x0, y0, x1, y1))
    return torch.as_tensor(noise_boxes, dtype=torch.float)


def prepare_mask_and_point_data(masks, labels, low_res_masks, batch, point_num):
    masks_sigmoid = torch.sigmoid(masks.clone())
    masks_binary = (masks_sigmoid > 0.5).float()

    low_res_masks_sigmoid = torch.sigmoid(low_res_masks.clone())

    points, point_labels = extract_error_points(masks_binary, labels, point_num)

    batch["mask_inputs"] = low_res_masks_sigmoid
    batch["point_coords"] = torch.as_tensor(points)
    batch["point_labels"] = torch.as_tensor(point_labels)
    batch["boxes"] = None
    return batch


def extract_error_points(pr, gt, point_num=9):
    """
    Selects random points from the predicted and ground truth masks and assigns labels to them.
    Args:
        pred (torch.Tensor): Predicted mask tensor.
        gt (torch.Tensor): Ground truth mask tensor.
        point_num (int): Number of random points to select. Default is 9.
    Returns:
        batch_points (np.array): Array of selected points coordinates (x, y) for each batch.
        batch_labels (np.array): Array of corresponding labels (0 for background, 1 for foreground) for each batch.
    """
    pred, gt = pr.data.cpu().numpy(), gt.data.cpu().numpy()
    error = np.zeros_like(pred)
    error[pred != gt] = 1

    # error = np.logical_xor(pred, gt)
    batch_points = []
    batch_labels = []
    for j in range(error.shape[0]):
        one_pred = pred[j].squeeze(0)
        one_gt = gt[j].squeeze(0)
        one_erroer = error[j].squeeze(0)

        indices = np.argwhere(one_erroer == 1)
        if indices.shape[0] > 0:
            selected_indices = indices[np.random.choice(indices.shape[0], point_num, replace=True)]
        else:
            indices = np.random.randint(0, 256, size=(point_num, 2))
            selected_indices = indices[np.random.choice(indices.shape[0], point_num, replace=True)]
        selected_indices = selected_indices.reshape(-1, 2)

        points, labels = [], []
        for i in selected_indices:
            x, y = i[0], i[1]
            if one_pred[x, y] == 0 and one_gt[x, y] == 1:
                label = 1
            elif one_pred[x, y] == 1 and one_gt[x, y] == 0:
                label = 0
            else:
                label = -1
            points.append((y, x))  # Negate the coordinates
            labels.append(label)

        batch_points.append(points)
        batch_labels.append(labels)
    return np.array(batch_points), np.array(batch_labels)
