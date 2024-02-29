import os
import numpy as np
import einops
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
import torch.optim as optim
from tqdm import tqdm
from DataLoader import RoadDataset
from cfg import parse_args
from segment_anything import sam_model_registry
from utils import FocalDiceloss_IoULoss, SegMetrics, EarlyStopping


def prompt_mask_blocks(args, model, image_embedding, points, box, train_mode=True):
    if train_mode:

        if args.propmt_grad is True:
            sparse_embeddings, dense_embeddings = model.prompt_encoder(
                points=points,
                boxes=box,
                masks=None
            )
        else:
            with torch.no_grad():
                sparse_embeddings, dense_embeddings = model.prompt_encoder(
                    points=points,
                    boxes=box,
                    masks=None
                )

        # Generate the predicted masks and IoU predictions using the mask decoder
        low_res_masks, iou_pred = model.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False
        )

    else:

        with torch.no_grad():
            sparse_embeddings, dense_embeddings = model.prompt_encoder(
                points=points,
                boxes=box,
                masks=None
            )

            low_res_masks, iou_pred = model.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False
            )

    # The pred_masks in shape of [B, C, 256, 256] we need to upsample or downsample to the img input size
    # So the below line will # Resize to the ordered output size
    pred_masks = F.interpolate(low_res_masks, size=(args.img_size, args.img_size))

    return pred_masks, iou_pred


def train_sam(args, model, optimizer, criterion, train_loader, device, epoch):
    epoch_losses = []
    train_iter_metrics = [0] * len(args.metrics)
    model.train()

    with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1} Training', unit='img') as pbar:
        for batch in train_loader:
            # print("-----start training-----")
            image = batch['image'].to(device)
            mask = batch['mask'].to(device)
            box = batch['box'].to(device)
            point_coords = batch["point_coords"]
            point_labels = batch["point_labels"]

            # Convert point coordinates and labels to torch tensors and move to device
            coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=device)
            labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=device)

            points = (coords_torch, labels_torch)

            if args.use_adapter:
                for name, param in model.image_encoder.named_parameters():
                    if "Adapter" in name:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False

            else:
                for name, param in model.named_parameters():
                    param.requires_grad = "image_encoder" not in name

            image_embedding = (model.image_encoder(image)).to(device)

            pred_masks, iou_pred = prompt_mask_blocks(args, model, image_embedding, points, box, True)

            # to add the mask to the loss function it need to be shape of [B, 1, H, W] (1)
            # but it in shape of [B, H, W] (2), so the blow line convert from shape 2 --> 1
            mask = einops.repeat(mask, "b h w -> b 1 h w")

            loss = criterion(pred_masks, mask, iou_pred)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

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
            image = batch['image'].to(device)
            mask = batch['mask'].to(device)
            box = batch['box'].to(device)
            point_coords = batch["point_coords"]
            point_labels = batch["point_labels"]

            coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=device)
            labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=device)

            points = (coords_torch, labels_torch)

            for name, param in model.named_parameters():
                param.requires_grad = "image_encoder" not in name

            image_embedding = (model.image_encoder(image)).to(device)

            pred_masks, iou_pred = prompt_mask_blocks(args, model, image_embedding, points, box, False)

            mask = einops.repeat(mask, "b h w -> b 1 h w")

            loss = criterion(pred_masks, mask, iou_pred)
            epoch_losses.append(loss.item())

            batch_metrics = SegMetrics(pred_masks, mask, args.metrics)
            epoch_metrics = [valid_iter_metrics[i] + batch_metrics[i] for i in range(len(args.metrics))]

            pbar.update(1)

    return epoch_losses, epoch_metrics


def main(args):
    device = torch.device(args.device)

    model = sam_model_registry[args.model_type](args).to(device)
    criterion = FocalDiceloss_IoULoss()
    optimizer = optim.Adam(model.mask_decoder.parameters(), lr=args.lr, weight_decay=0)
    early_stop = EarlyStopping(patience=4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.3, patience=3, verbose=True)

    model_save_path = os.path.join(args.work_dir, args.run_name)
    os.makedirs(model_save_path, exist_ok=True)

    # load the datasets
    train_dataset = RoadDataset(args.train_root, points=15)
    train_dataloader = DataLoader(train_dataset, args.batch_size, True)

    valid_dataset = RoadDataset(args.valid_root, train=False, points=15)
    valid_dataloader = DataLoader(valid_dataset, args.batch_size, True)

    best_loss = 9e10

    for i in range(args.num_epochs):

        train_loss, _ = train_sam(args, model, optimizer, criterion, train_dataloader, device, i)

        valid_loss, valid_metrics = validate_sam(args, model, criterion, valid_dataloader, device, i)

        valid_metrics = [metric / len(valid_dataloader) for metric in valid_metrics]
        valid_metrics = {args.metrics[i]: '{:.4}'.format(valid_metrics[i])
                         for i in range(len(valid_metrics))}

        average_train_loss = np.mean(train_loss)
        average_valid_loss = np.mean(valid_loss)

        if args.use_scheduler:
            scheduler.step(average_valid_loss)

        # early stopping
        if args.early_stop:
            early_stop(average_valid_loss)
            if early_stop.early_stop:
                break
        
        # saving the ck
        if average_valid_loss < best_loss:
            best_loss = average_valid_loss
            model_saving = os.path.join(model_save_path, f"epoch_{i + 1}_SamSatellite.pth")
            state = {'model': model.float().state_dict(), 'optimizer': optimizer}
            torch.save(state, model_saving)

        print(f"<------------> Train_loss {average_train_loss:.4f} | Valid_loss {average_valid_loss:.4f} | "
              f"IOU_Score {valid_metrics['iou']} | Dice_Score {valid_metrics['dice']} \n ")


if __name__ == '__main__':
    args = parse_args()
    main(args)
