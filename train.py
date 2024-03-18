import os
import random
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import functional as F
from tqdm import tqdm
from datasets import RoadDataset
from cfg import parse_args
from segment_anything import sam_model_registry
from utils import FocalDiceloss_IoULoss, SegMetrics, EarlyStopping, prepare_mask_and_point_data, save_checkpoint, \
    to_device
from monai.losses import DiceFocalLoss

def prompt_mask_blocks(args, model, image_embedding, batch, propmt_grad=True, train_mode=True):
    if batch['point_coords'] is not None:
        points = (batch["point_coords"], batch["point_labels"])
    else:
        points = None

    if train_mode:

        if propmt_grad:
            sparse_embeddings, dense_embeddings = model.prompt_encoder(
                points=points,
                boxes=batch.get("boxes", None),
                masks=batch.get("mask_inputs", None)
            )
        else:
            with torch.no_grad():
                sparse_embeddings, dense_embeddings = model.prompt_encoder(
                    points=points,
                    boxes=batch.get("boxes", None),
                    masks=batch.get("mask_inputs", None)
                )

        # Generate the predicted masks and IoU predictions using the mask decoder
        low_res_masks, iou_pred = model.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=args.multimask
        )

    else:

        with torch.no_grad():
            sparse_embeddings, dense_embeddings = model.prompt_encoder(
                points=points,
                boxes=batch.get("boxes", None),
                masks=batch.get("mask_inputs", None)
            )

            low_res_masks, iou_pred = model.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=args.multimask
            )

    if args.multimask:
        max_values, max_indexs = torch.max(iou_pred, dim=1)
        max_values = max_values.unsqueeze(1)
        iou_pred = max_values
        low_res = []
        for i, idx in enumerate(max_indexs):
            low_res.append(low_res_masks[i:i + 1, idx])
        low_res_masks = torch.stack(low_res, 0)

    pred_masks = F.interpolate(low_res_masks, size=(args.img_size, args.img_size), mode="bilinear",
                               align_corners=False, )

    return pred_masks, iou_pred, low_res_masks


def train_sam(args, model, optimizer, criterion, train_loader, device, epoch):
    epoch_losses = []
    train_iter_metrics = [0] * len(args.metrics)
    model.train()

    with tqdm(total=len(train_loader), desc=f'Training {epoch + 1}/{args.num_epochs}', unit='img') as pbar:

        for batch in train_loader:
            # print("-----start training-----")
            batch = to_device(batch, device)

            if args.use_adapter:
                for name, param in model.image_encoder.named_parameters():
                    if "Adapter" in name:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False

            else:
                for name, param in model.named_parameters():
                    param.requires_grad = "image_encoder" not in name

            image_embedding = (model.image_encoder(batch['image'])).to(device)

            pred_masks, iou_pred, low_res_masks = prompt_mask_blocks(args, model, image_embedding, batch,
                                                                     propmt_grad=True)
            mask = batch['mask']

            loss = criterion(pred_masks, mask)
            loss.backward(retain_graph=False)

            optimizer.zero_grad()
            optimizer.step()

            point_num = random.choice(args.point_list)
            batch = prepare_mask_and_point_data(pred_masks, mask, low_res_masks, batch, point_num)
            batch = to_device(batch, args.device)

            image_embedding = image_embedding.detach().clone()

            for name, param in model.named_parameters():
                param.requires_grad = "image_encoder" not in name

            for iter in range(args.point_iterator):
                pred_masks, iou_pred, low_res_masks = prompt_mask_blocks(args, model, image_embedding, batch,
                                                                                propmt_grad=False)
                loss = criterion(pred_masks, mask)
                loss.backward(retain_graph=True)

                optimizer.step()
                optimizer.zero_grad()

                if iter != args.point_iterator - 1:
                    point_num = random.choice(args.point_list)
                    batch = prepare_mask_and_point_data(pred_masks, mask, low_res_masks, batch, point_num)
                    batch = to_device(batch, args.device)

            epoch_losses.append(loss.item())

            batch_metrics = SegMetrics(pred_masks, mask, args.metrics)
            epoch_metrics = [train_iter_metrics[i] + batch_metrics[i] for i in range(len(args.metrics))]

            pbar.update(1)

    return epoch_losses, epoch_metrics


def validate_sam(args, model, criterion, val_loader, device, epoch):
    epoch_losses = []
    valid_iter_metrics = [0] * len(args.metrics)
    model.eval()

    with tqdm(total=len(val_loader), desc=f'Validtion', unit='img') as pbar:
        for batch in val_loader:
            batch = to_device(batch, device)

            for name, param in model.named_parameters():
                param.requires_grad = "image_encoder" not in name

            image_embedding = (model.image_encoder(batch['image'])).to(device)

            mask = batch['mask']
            for iter in range(args.point_iterator):

                pred_masks, iou_pred, low_res_masks = prompt_mask_blocks(args, model, image_embedding, batch, False)
                if iter != args.point_iterator - 1:
                    point_num = random.choice(args.point_list)
                    batch = prepare_mask_and_point_data(pred_masks, mask, low_res_masks, batch, point_num)
                    batch = to_device(batch, args.device)

            loss = criterion(pred_masks, mask)
            epoch_losses.append(loss.item())

            batch_metrics = SegMetrics(pred_masks, mask, args.metrics)
            epoch_metrics = [valid_iter_metrics[i] + batch_metrics[i] for i in range(len(args.metrics))]

            pbar.update(1)

    return epoch_losses, epoch_metrics


def main(args):
    # Set the random seed for torch and numpy operations to ensure reproducibility and consistency
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)

    model = sam_model_registry[args.model_type](args.checkpoint, args.use_adapter).to(device)

    criterion = DiceFocalLoss(sigmoid=True, squared_pred=True, reduction='mean')
    # criterion = FocalDiceloss_IoULoss()
    optimizer = optim.Adam(model.mask_decoder.parameters(), lr=args.lr, weight_decay=0)
    scheduler1 = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.4, patience=2, verbose=True)
    scheduler2 = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3, 5, 7], gamma = 0.7)
    early_stop = EarlyStopping(patience=5)

    model_save_path = os.path.join(args.work_dir, args.run_name)
    os.makedirs(model_save_path, exist_ok=True)

    start_epoch = 0

    # args.resume_training = '../workdir/sam-satellite-models/checkpoint_best.pth'
    if args.resume_training is not None:
        print(f'=> resuming from {args.resume_training}')
        assert os.path.exists(args.resume_training)
        checkpoint_file = os.path.join(args.resume_training)
        loc = 'mps:{}'.format(args.device)
        checkpoint = torch.load(checkpoint_file, map_location=loc)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        print(f'=> loaded checkpoint {checkpoint_file} (epoch {start_epoch})')

    # load the datasets
    train_dataset = RoadDataset(args.train_root, is_train=True, is_box=False, points=args.num_points)
    train_dataloader = DataLoader(train_dataset, args.batch_size, True, num_workers=4, pin_memory=True)

    valid_dataset = RoadDataset(args.valid_root, is_train=True, is_box=False, points=args.num_points)
    valid_dataloader = DataLoader(valid_dataset, args.batch_size, True, num_workers=4, pin_memory=True)

    best_loss = 9e10

    for i in range(start_epoch, args.num_epochs):

        train_loss, train_metrics = train_sam(args, model, optimizer, criterion, train_dataloader, device, i)

        valid_loss, valid_metrics = validate_sam(args, model, criterion, valid_dataloader, device, i)

        train_metrics = [metric / len(train_dataloader) for metric in train_metrics]
        train_metrics = {args.metrics[i]: '{:.4}'.format(train_metrics[i])
                         for i in range(len(train_metrics))}

        valid_metrics = [metric / len(valid_dataloader) for metric in valid_metrics]
        valid_metrics = {args.metrics[i]: '{:.4}'.format(valid_metrics[i])
                         for i in range(len(valid_metrics))}

        average_train_loss = np.mean(train_loss)
        average_valid_loss = np.mean(valid_loss)

        if args.use_scheduler:
            scheduler1.step(average_valid_loss)
            # scheduler2.step()

        # early stopping
        if args.early_stop:
            early_stop(average_valid_loss)
            if early_stop.early_stop:
                break

        # saving the ck
        if average_valid_loss < best_loss:
            best_loss = average_valid_loss
            is_Best = True
            save_checkpoint({
                'epoch': i + 1,
                'model': args.net,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_tol': best_loss,
            }, is_Best, model_save_path, filename=f"Road{i + 1}_checkpoint.pth")
        else:
            is_Best = False

        print(
            f"<=======Train=======> loss {average_train_loss:.4f} | IOU_Score {train_metrics['iou']} | Dice_Score {train_metrics['dice']} \n"
            f"<=======Valid=======> loss {average_valid_loss:.4f} | IOU_Score {valid_metrics['iou']} | Dice_Score {valid_metrics['dice']} \n")


if __name__ == '__main__':
    args = parse_args()
    main(args)
