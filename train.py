import time
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn import functional as F
import torch.optim as optim
from tqdm import tqdm
from DataLoader import RoadDataset
from cfg import parse_args
from segment_anything import SamPredictor, sam_model_registry


args = parse_args()
device = args.device
model_save_path = os.path.join(args.work_dir, args.run_name)
os.makedirs(model_save_path, exist_ok=True)
model = sam_model_registry[args.model_type](checkpoint=args.checkpoint).to(device)

