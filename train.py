import os
import einops
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
import torch.optim as optim
from tqdm import tqdm
from DataLoader import RoadDataset, train_transform
from cfg import parse_args
from segment_anything import SamPredictor, sam_model_registry
from utils import FocalDiceloss_IoULoss


args = parse_args()
ckeckpoint_dir = args.checkpoint
train_root = args.train_root
device = torch.device(args.device)

model_save_path = os.path.join(args.work_dir, args.run_name)
os.makedirs(model_save_path, exist_ok=True)
model = sam_model_registry[args.model_type](checkpoint=ckeckpoint_dir).to(device)

dataset = RoadDataset(train_root, 1024, "train", box = True, transform=train_transform)
train_dataloader = DataLoader(dataset, 4, True)


def train_sam(args, model, optimizer, criterion, train_loader, device, epoch):

    epoch_loss = 0

    model.train()
    optimizer.zero_grad()

    with tqdm(total=len(train_loader), desc=f'Epoch {epoch}', unit='img') as pbar:
        for pack in train_loader:

            image = pack['image'].to(device)
            mask = pack['mask'].to(device)

            if 'box' in pack:
                box = pack['box']

            for name, param in model.named_parameters():
                param.requires_grad = "image_encoder" not in name

            image_embedding = model.image_encoder(image)

            if args.propmt_grad is True:
                sparse_embeddings, dense_embeddings = model.prompt_encoder(
                points = None,
                boxes = box.to(device),
                masks = None
            )
            else:
                with torch.no_grad():
                    sparse_embeddings, dense_embeddings = model.prompt_encoder(
                    points = None,
                    boxes = box.to(device),
                    masks = None
            )
            pred_masks, iou_pred = model.mask_decoder(
                    image_embeddings = image_embedding.to(device),
                    image_pe = model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings = sparse_embeddings,
                    dense_prompt_embeddings = dense_embeddings,
                    multimask_output = False
            )       
            
            # The pred_masks in shape of [B, C, 256, 256] we need to upsample or downsample to the img input size 
            # So the below line will ## Resize to the ordered output size
            pred = F.interpolate(pred_masks,size=(args.img_size, args.img_size))

            # to add the mask to the loss function it need to be shape of [B, 1, H, W] (1)
            # but it in shape of [B, H, W] (2), so the blow line convert from shape 2 --> 1
            mask = einops.repeat(mask, "b h w -> b 1 h w")

            loss = criterion(pred, mask, iou_pred)

            pbar.set_postfix(**{'loss (batch)': loss.item()})
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()

            pbar.update()

    return epoch_loss


def validate_sam(args, model, epoch, criterion, val_loader, device):

    model.eval()

    with tqdm(total=len(val_loader), desc='Validation round', unit='batch', leave=False) as pbar:
        for pack in enumerate(val_loader):

            image = pack['image'].to(device)
            mask = pack['mask'].to(device)

            if 'box' in pack:
                box = pack['box']

            with torch.no_grad():
                image_embedding= model.image_encoder(image)

                sparse_embeddings, dense_embeddings = model.prompt_encoder(
                points = None,
                boxes = box.to(device),
                masks = None)

                pred_masks, iou_pred = model.mask_decoder(
                    image_embeddings = image_embedding.to(device),
                    image_pe = model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings = sparse_embeddings,
                    dense_prompt_embeddings = dense_embeddings,
                    multimask_output = False)
                
                pred = F.interpolate(pred_masks,size=(args.out_size,args.out_size))
                tot += criterion(pred, mask, iou_pred)

            pbar.update()

    return tot/len(val_loader)