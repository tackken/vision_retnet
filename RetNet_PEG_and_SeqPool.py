import glob
import itertools
import json
import logging
import math
import os
import random
from typing import OrderedDict
import time
import itertools
import torch.nn.init as init
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import scipy.stats
import slackweb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from matplotlib.gridspec import GridSpec
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import KFold
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR
from torch.utils.data import Dataset
from torchvision import datasets, models, transforms
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision.models as models


######################################################################################################
######################################################################################################
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
dir_name = "Validation_vit"
mode = "."

tsfm = transforms.Compose([
    transforms.RandomInvert(p=0.7),
    transforms.RandomPosterize(bits=2, p=0.7),
    transforms.RandomSolarize(threshold=100, p=0.7),
    transforms.RandomEqualize(p=0.7)
])

def image_transformer(img, seed):
    random.seed(seed)
    image = Image.fromarray(img).convert("RGB")
    image = tsfm(image)
    return np.array(image)

class MyDataset(Dataset):
    def __init__(
        self,
        df,
        pt_list,
        msk_list,
        label_name,
        dir_name_IMG,
        dir_name_MSK,
        extension,
        transform
    ):
        self.df = df
        self.pt_list = pt_list
        self.msk_list = msk_list
        self.label_name = label_name
        self.dir_name_IMG = dir_name_IMG
        self.dir_name_MSK = dir_name_MSK
        self.extension = extension
        self.transform = transform

        self.data = []
        self.mask = []
        self.label = []
        self.path = []

        for pt_id in tqdm(pt_list):
            a_patient_df = df[df["Patient_id"] == pt_id]
            slice_ids = a_patient_df["Image_id"].values
            slice_ids = list(slice_ids)
            for img_id in slice_ids:
                # Make data for image
                image_sector = a_patient_df[a_patient_df["Image_id"] == img_id]

                # Make data for "label"
                label_sector = image_sector[lable_name].values[0].astype(np.int64)

                img_path = self.dir_name_IMG + f"HCC_{pt_id}/" + f"{img_id}.{extension}" #
                msk_path = self.dir_name_MSK + f"HCC_{pt_id}/" + f"{img_id}.{extension}" #

                ######
                image_open = Image.open(img_path).convert('RGB')
                image_open = image_open.resize((256, 256))
                if self.transform:
                    img_sector1 = self.transform(image_open)
                else:
                    transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                    ])
                    img_sector1 = transform(image_open)
                ######

                mask_open = Image.open(msk_path)
                mask_open = mask_open.resize((256, 256))
                msk_sector1 = np.array(mask_open)

                if self.transform:
                    seed = random.randint(0, 10000)
                    img_sector1 = image_transformer(img_sector1, seed)

                ##################### Original Image #####################
                img_sector1 = img_sector1.float()

                #####################  Mask image #####################
                msk_sector1 = msk_sector1

                #####################  Label #####################
                if label_sector == 0:
                    label_sector = label_sector
                if label_sector == 1:
                    label_sector = label_sector

                self.data.append(torch.tensor(img_sector1))
                self.mask.append(torch.tensor(msk_sector1))
                self.label.append(torch.tensor(label_sector))
                self.path.append(img_path)

    def __getitem__(self, idx):
        return self.data[idx], self.mask[idx], self.label[idx], self.path[idx]

    def __len__(self):
        return len(self.data)






######################################################################################################
######################################################################################################
""" Dataset load """
csv_name = "output_BW.csv"

df_all = pd.read_csv(f'{mode}/HCC/For_RetNet/HCC_マスク移動用/{csv_name}', index_col=0)
df_all = df_all.reset_index(drop=False)
lable_name = 'Vascular invasion'
dir_name_IMG = f'{mode}/HCC/For_RetNet/HCC_マスク移動用/PNG_256_CROP_BW/'
dir_name_MSK = f'{mode}/HCC/For_RetNet/HCC_マスク移動用/PNG_256_CROP_BW_Seg/'
extension = 'png'

# Seed
seed_value = 8
random.seed(seed_value)
random.shuffle(patient_without_list)
random.shuffle(patient_with_list)

# Read a patient list
df_pt_list = pd.read_csv(f'{mode}/HCC/csv_data/pt_list.csv')
patient_without_list = df_pt_list[df_pt_list['Vascular invasion'] == 0]['TCIA_ID'].tolist()
patient_with_list = df_pt_list[df_pt_list['Vascular invasion'] == 1]['TCIA_ID'].tolist()

# class -
patient_without_list_copy = patient_without_list.copy()
random.seed(seed_value)
without_pt_train = random.sample(patient_without_list_copy, k=49) ################ Train
for pt in without_pt_train:
    patient_without_list_copy.remove(pt)
random.seed(seed_value)
without_pt_validation = random.sample(patient_without_list_copy, k=16) ################ Validation
for pt in without_pt_validation:
    patient_without_list_copy.remove(pt)
without_pt_test = patient_without_list_copy ################ Test

# class +
patient_with_list_copy = patient_with_list.copy()
random.seed(seed_value)
with_pt_train = random.sample(patient_with_list_copy, k=14) ################ Train
for pt in with_pt_train:
    patient_with_list_copy.remove(pt)
random.seed(seed_value)
with_pt_validation = random.sample(patient_with_list_copy, k=4) ################ Validation
for pt in with_pt_validation:
    patient_with_list_copy.remove(pt)
with_pt_test = patient_with_list_copy ################ Test


"""For Machine Learning information"""
print(f"Train (-) pt name ({len(without_pt_train)}): {without_pt_train}")
print(f"Train (+) pt name ({len(with_pt_train)}): {with_pt_train}")
print(f"Validation (-) pt name ({len(without_pt_validation)}): {without_pt_validation}")
print(f"Validation (+) pt name ({len(with_pt_validation)}): {with_pt_validation}")
print(f"Test (-) pt name ({len(without_pt_test)}): {without_pt_test}")
print(f"Test (+) pt name ({len(with_pt_test)}): {with_pt_test}")
"""For Machine Learning information"""

pt_list_train = sorted(without_pt_train + with_pt_train)
pt_list_validation = sorted(without_pt_validation + with_pt_validation)
pt_list_test = sorted(without_pt_test + with_pt_test)

print("kkkkkkkkkkkkkkkkkkkkk")
print(pt_list_train)
print(pt_list_validation)
print(pt_list_test)

img_train_without = len(df_all[df_all['Patient_id'].isin(without_pt_train)])
img_train_with = len(df_all[df_all['Patient_id'].isin(with_pt_train)])

img_validation_without = len(df_all[df_all['Patient_id'].isin(without_pt_validation)])
img_validation_with = len(df_all[df_all['Patient_id'].isin(with_pt_validation)])

img_test_without = len(df_all[df_all['Patient_id'].isin(without_pt_test)])
img_test_with = len(df_all[df_all['Patient_id'].isin(with_pt_test)])

""" process for patient-base evaluation (Test dataset)"""
print("snbajkdfbhvshkdfbvkd,fhbzvjheasfdvfh,bvcnsdfjbv")
ptnt_id = []
img_len_for_patient = []
new_validation_label = []
all_patient_lst = without_pt_validation + with_pt_validation
for i in all_patient_lst:
    slice_num = len(df_all[df_all['Patient_id'].isin([i])])
    ptnt_id.append(i)
    img_len_for_patient.append(slice_num)

    patient_without_list = df_pt_list[df_pt_list['TCIA_ID'] == i]['Vascular invasion'].tolist()
    new_validation_label.append(patient_without_list[0])
print("snbajkdfbhvshkdfbvkd,fhbzvjheasfdvfh,bvcnsdfjbv")


# Train
train_set = MyDataset(
    df_all,
    pt_list_train,
    pt_list_train,
    lable_name,
    dir_name_IMG,
    dir_name_MSK,
    extension,
    transform=False
)

# Validation
validation_set = MyDataset(
    df_all,
    pt_list_validation,
    pt_list_validation,
    lable_name,
    dir_name_IMG,
    dir_name_MSK,
    extension,
    transform=False
)

# Test
test_set = MyDataset(
    df_all,
    pt_list_test,
    pt_list_test,
    lable_name,
    dir_name_IMG,
    dir_name_MSK,
    extension,
    transform=False
)


######################################################################################################
######################################################################################################
print("--------------")
total_images_train_set = len(train_set)
print(f"Total number of images(Train): {total_images_train_set} ([-]{img_train_without}({len(without_pt_train)}) [+]{img_train_with}({len(with_pt_train)}))")
total_images_validation_set = len(validation_set)
print(f"Total number of images(Validation): {total_images_validation_set} ([-]{img_validation_without}({len(without_pt_validation)}) [+]{img_validation_with}({len(with_pt_validation)}))")
total_images_test_set = len(test_set)
print(f"Total number of images(Test): {total_images_test_set} ([-]{img_test_without}({len(without_pt_test)}) [+]{img_test_with}({len(with_pt_test)}))")
print("--------------")

batch_size = 32
print(f"total_images_train_setとtotal_images_validation_setとtotal_images_test_setの共通の約数(Batch size)は: {batch_size}")

trainloader = torch.utils.data.DataLoader(
    train_set, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True, num_workers=0)
validationloader = torch.utils.data.DataLoader(
    validation_set, batch_size=batch_size, shuffle=False, drop_last=True, pin_memory=True, num_workers=0)
testloader = torch.utils.data.DataLoader(
    test_set, batch_size=batch_size, shuffle=False, drop_last=True, pin_memory=True, num_workers=0)











######################################################################################################
######################################################################################################
# 1. Patching
class Patching(nn.Module):
    def __init__(self, patch_size):
        """ [input]
            - patch_size (int) : パッチの縦の長さ（=横の長さ）
        """
        super().__init__()
        self.net = Rearrange("b c (h ph) (w pw) -> b (h w) (ph pw c)", ph = patch_size, pw = patch_size)

    def forward(self, x):
        """ [input]
            - x (torch.Tensor) : 画像データ
                - x.shape = torch.Size([batch_size, channels, image_height, image_width])
        """
        # x = x.unsqueeze(3) # For Gray image
        # x = x.permute(0, 3, 1, 2) #
        x = self.net(x) #  # [32, 1024, 192]
        return x


# 2. LinearProjection
class LinearProjection(nn.Module):
    def __init__(self, patch_dim, dim, dropout):
        """ [input]
            - patch_dim (int) : 一枚あたりのパッチのベクトルの長さ（= channels * (patch_size ** 2)）
            - dim (int) : パッチのベクトルが変換されたベクトルの長さ
        """
        super().__init__()
        self.linear_layer = nn.Linear(patch_dim, dim, bias=True)
        init.xavier_uniform_(self.linear_layer.weight) # WEIGHT
        init.zeros_(self.linear_layer.bias) # BIAS

        self.net = nn.Sequential(
            self.linear_layer,
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """ [input]
            - x (torch.Tensor)
                - x.shape = torch.Size([batch_size, n_patches, patch_dim])
        """
        x = self.net(x) # [32, 1024, 64]
        return x


# 3. Embedding
class Embedding(nn.Module):
    def __init__(self, dim, n_patches):
        """ [input]
            - dim (int) : パッチのベクトルが変換されたベクトルの長さ
            - n_patches (int) : パッチの枚数
        """
        super().__init__()
        # [class] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        # position embedding
        # self.pos_embedding = nn.Parameter(torch.randn(1, n_patches + 1, dim))
        self.pos_embedding = self.posemb(n_patches + 1, dim)

    def posemb(self, len_li, patch_dim, temperature=10000):
        assert (patch_dim % 2) == 0, 'feature dimension must be multiple of 2 for sincos emb'
        pe = torch.zeros(1, len_li, patch_dim, dtype=torch.float)
        div_term = torch.exp(torch.arange(0, patch_dim, 2).float() * (-math.log(temperature) / patch_dim))
        for i in range(len_li):
            pe[0, i, 0::2] = torch.sin(i * div_term)
            pe[0, i, 1::2] = torch.cos(i * div_term)
        return pe.to(device)

    def forward(self, x):
        """[input]
            - x (torch.Tensor)
                - x.shape = torch.Size([batch_size, n_patches, dim])
        """
        batch_size, _, __ = x.shape

        # [class] token
        # x.shape : [batch_size, n_patches, patch_dim] -> [batch_size, n_patches + 1, patch_dim]
        cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b = batch_size)
        x = torch.cat([cls_tokens, x], dim = 1)

        # Positional embedding
        x += self.pos_embedding # [64, 257, 16],,,,, x: [64, 257, 16], self.pos_embedding: [1, 257, 16]

        return x



# MLP
class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, dropout):
        """ [input]
            - dim (int) : パッチのベクトルが変換されたベクトルの長さ
            - hidden_dim (int) : 隠れ層のノード数
        """
        super().__init__()
        self.linear_layer = nn.Linear(dim, hidden_dim, bias=True)
        init.xavier_uniform_(self.linear_layer.weight) # WEIGHT
        init.zeros_(self.linear_layer.bias) # BIAS
        self.linear_layer_ = nn.Linear(hidden_dim, dim, bias=True)
        init.xavier_uniform_(self.linear_layer_.weight) # WEIGHT
        init.zeros_(self.linear_layer_.bias) # BIAS

        self.net = nn.Sequential(
            self.linear_layer,
            nn.GELU(),
            nn.Dropout(dropout),
            self.linear_layer_,
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """[input]
            - x (torch.Tensor)
                - x.shape = torch.Size([batch_size, n_patches + 1, dim])
        """
        x = self.net(x) # [64, 257, 16]
        return x


########################## RetNet ##########################
# Multi-Head Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, dim, n_heads, n_patches, dropout):
        """ [input]
            - dim (int) : パッチのベクトルが変換されたベクトルの長さ
            - n_heads (int) : heads の数
        """
        super().__init__()
        self.n_heads = n_heads
        self.n_patches = n_patches
        self.dim_heads = dim // n_heads
        self.past_kv = None
        self.decay_mask = self.dim_heads ** -0.5
        self.chunk_decay = self.dim_heads ** -0.5
        self.inner_decay = self.dim_heads ** -0.5

        project_out = not (n_heads == 1 and self.dim_heads == dim)
        inner_dim = self.dim_heads *  self.n_heads

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        self.softmax = nn.Softmax(dim = -1)

        self.dropout = nn.Dropout(dropout)

        self.concat = Rearrange("b h n d -> b n (h d)", h = self.n_heads)

    def ChunkwiseRetention(
        self,
        q, k, v,
        past_kv,
        decay_mask,
        chunk_decay,
        inner_decay,
        msk
        ):

        past_kv = torch.tensor(past_kv, dtype=torch.float32, device=device)
        decay_mask = torch.tensor(decay_mask, dtype=torch.float32, device=device)
        chunk_decay = torch.tensor(chunk_decay, dtype=torch.float32, device=device)
        inner_decay = torch.tensor(inner_decay, dtype=torch.float32, device=device)

        retention = torch.matmul(q, k.transpose(-1, -2)).to(device)
        retention = retention * decay_mask # [32, 4, 1025, 1025]


        # Before Casual Maskiing
        weights_before = retention / math.sqrt(self.dim_heads)
        weights_before = self.softmax(weights_before)


        # Casual Maskiing
        transform = transforms.Resize((16, 16))
        msk = torch.stack([transform(img.unsqueeze(0)).squeeze(0) for img in msk])

        batch_size, height, width = msk.shape

        # Initialize the resulting msk with zeros
        msk_result = torch.zeros(batch_size, 256 + 1, 256 + 1)

        for batch_idx in range(batch_size):

            #####################  Mask image #####################
            mask_open = msk[batch_idx]
            random_values = torch.randint(1, 128, mask_open.shape, dtype=torch.uint8, device=mask_open.device)
            mask_open[mask_open == 0] = random_values[mask_open == 0]

            pixel_values = mask_open.flatten().reshape(-1, 1).float()
            inner_product = torch.matmul(pixel_values,  pixel_values.transpose(1, 0))
            normalized_inner_product = inner_product / torch.max(inner_product)
            tiled_mask = normalized_inner_product

            exponential_factor = 5
            decreasing_tensor_0 = torch.exp(-exponential_factor * torch.linspace(0, 1, tiled_mask.shape[1])).unsqueeze(0).to(device)
            tiled_mask = torch.cat([decreasing_tensor_0, tiled_mask.to(device)], dim=0)
            decreasing_tensor_1 = torch.exp(-exponential_factor * torch.linspace(0, 1, tiled_mask.shape[0])).unsqueeze(1).to(device)
            tiled_mask = torch.cat([decreasing_tensor_1, tiled_mask], dim=1)

            tiled_mask.fill_diagonal_(1)
            # tiled_mask.fill_diagonal_(-torch.inf)

            tiled_mask = torch.tril(tiled_mask)
            msk_result[batch_idx, :, :] = tiled_mask

        msk_result = msk_result.unsqueeze(1).expand(-1, self.n_heads, -1, -1) # [32, 4, 1025, 1025]

        #########################################################################
        mask = msk_result.to(device)
        retention = retention * mask

        # After Casual Maskiing
        weights_after = retention / math.sqrt(self.dim_heads)
        weights_after = self.softmax(weights_after)


        inner_retention = torch.matmul(retention, v)
        inner_retention = self.dropout(inner_retention)
        cross_retention = torch.matmul(q, past_kv) * inner_decay # [32, 4, 1025, 16] * [32, 4, *16*, *16*] = [32, 4, 1025, 16]


        retention = inner_retention + cross_retention
        group_norm = nn.GroupNorm(num_groups=self.n_heads, num_channels=self.n_heads).to(device)
        output = group_norm(retention)
        output = self.concat(output)
        current_kv = chunk_decay * past_kv + torch.matmul(k.transpose(-1, -2), v)
        return output, current_kv, weights_before, weights_after


    def forward(self, x, msk):
        """[input]
            - x (torch.Tensor)
                - x.shape = torch.Size([batch_size, n_patches + 1, dim])
        """

        x = x.to(device)

        if self.past_kv == None:
            eye_matrix = np.eye(self.dim_heads)
            tiled_eye_matrix = np.tile(eye_matrix, (self.n_heads, 1, 1))
            self.past_kv = np.tile(tiled_eye_matrix, (batch_size, 1, 1, 1))

        past_kv = self.past_kv

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.n_heads), qkv)

        decay = self.dim_heads ** -0.5
        output, current_kv, weights_before, weights_after = self.ChunkwiseRetention(q,k,v, past_kv, decay, decay, decay, msk)

        self.output = output
        self.past_kv = current_kv

        attn_weights_before_mask = weights_before
        attn_weights_after_mask = weights_after

        return output, attn_weights_before_mask, attn_weights_after_mask


class PEG(nn.Module):
    def __init__(self, dim=256, k=3):
        super().__init__()
        self.pos = nn.Conv2d(dim, dim, k, 1, k//2, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        cls_token, feat_tokens = x[:, 0], x[:, 1:]
        feat_tokens = feat_tokens.transpose(1, 2).view(B, C, H, W)
        x = self.pos(feat_tokens) + feat_tokens
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x


# 4. Transformer Encoder
class TransformerEncoder(nn.Module):
    def __init__(self, dim, n_heads, mlp_dim, depth, n_patches, dropout):
        """ [input]
            - dim (int) : 各パッチのベクトルが変換されたベクトルの長さ（参考[1] (1)式 D）
            - depth (int) : Transformer Encoder の層の深さ（参考[1] (2)式 L）
            - n_heads (int) : Multi-Head Attention の head の数
            - mlp_dim (int) : MLP の隠れ層のノード数
        """
        super().__init__()

        # Layers
        self.norm = nn.LayerNorm(dim)
        self.multi_head_attention = MultiHeadAttention(dim = dim, n_heads = n_heads, n_patches = n_patches, dropout = dropout)
        self.mlp = MLP(dim = dim, hidden_dim = mlp_dim, dropout = dropout)
        self.depth = depth


    def forward(self, x, msk):
        """[input]
            - x (torch.Tensor)
                - x.shape = torch.Size([batch_size, n_patches + 1, dim])
        """
        for _ in range(self.depth):
            attn_x, attn_weights_before_mask, attn_weights_after_mask = self.multi_head_attention(self.norm(x), msk) # [32, 1025, 256]
            x = attn_x + x # Resudual Connection
            x = self.mlp(self.norm(x)) + x

        return x, attn_weights_before_mask, attn_weights_after_mask

# 5. MLPHead(SeqPool)
class MLPHead(nn.Module):
    def __init__(self, dim, out_dim, dropout):
        super().__init__()
        self.linear_layer = nn.Linear(dim, 1)
        init.xavier_uniform_(self.linear_layer.weight) # WEIGHT
        init.zeros_(self.linear_layer.bias) # BIAS
        self.linear_layer_ = nn.Linear(dim, out_dim)
        init.xavier_uniform_(self.linear_layer_.weight) # WEIGHT
        init.zeros_(self.linear_layer_.bias) # BIAS

        self.net_1 = nn.Sequential(
            nn.LayerNorm(dim)
        )
        self.net_1_2 = nn.Sequential(
           self.linear_layer
        )
        self.net_2 = nn.Sequential(
            self.linear_layer_,
            nn.Dropout(dropout)
        )
    def forward(self, x):
        x_input = x # [64, 257, 16]
        x = self.net_1(x_input) # [64, 257, 16]

        x = self.net_1_2(x) # [64, 257, 1]
        x = F.softmax(x.transpose(1, 2), dim=2)
        x = torch.matmul(x, x_input)  # [64, 1, 257] * [64, 257, 16] ----> [64, 1, 16]
        x = torch.flatten(x, start_dim=1) # [64, 16]
        x = self.net_2(x) # [64, 2]
        return x


class ViT(nn.Module):
    def __init__(self, image_size, patch_size, n_classes, dim, depth, n_heads, channels, mlp_dim, dropout):
        """ [input]
            - image_size (int) : 画像の縦の長さ（= 横の長さ）
            - patch_size (int) : パッチの縦の長さ（= 横の長さ）
            - n_classes (int) : 分類するクラスの数
            - dim (int) : 各パッチのベクトルが変換されたベクトルの長さ（参考[1] (1)式 D）
            - depth (int) : Transformer Encoder の層の深さ（参考[1] (2)式 L）
            - n_heads (int) : Multi-Head Attention の head の数
            - chahnnels (int) : 入力のチャネル数（RGBの画像なら3）
            - mlp_dim (int) : MLP の隠れ層のノード数
        """

        super().__init__()

        # Params
        n_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size * patch_size

        self.depth = depth
        self.dropout = dropout

        # Layers
        self.patching = Patching(patch_size = patch_size)
        self.linear_projection_of_flattened_patches = LinearProjection(patch_dim = patch_dim, dim = dim, dropout = dropout)
        self.embedding = PEG(dim = dim, n_patches = n_patches)
        self.transformer_encoder = TransformerEncoder(dim = dim, n_heads = n_heads, mlp_dim = mlp_dim, depth = depth, n_patches = n_patches, dropout = dropout)
        self.mlp_head = MLPHead(dim = dim, out_dim = n_classes, dropout = dropout)


    def forward(self, img, msk):
        """ [input]
            - img (torch.Tensor) : 画像データ
                - img.shape = torch.Size([batch_size, channels, image_height, image_width])
        """

        x = img # torch.Size([1, 128, 128, 3])

        # 1. Patching
        # x.shape : [batch_size, channels, image_height, image_width] -> [batch_size, n_patches, channels * (patch_size ** 2)]
        x = self.patching(x) # torch.Size([5, 256, 192])

        # 2. linear projection
        # x.shape : [batch_size, n_patches, channels * (patch_size ** 2)] -> [batch_size, n_patches, dim]
        x = self.linear_projection_of_flattened_patches(x)

        # 3. Embedding
        # x.shape : [batch_size, n_patches, dim] -> [batch_size, n_patches + 1, dim]
        x = self.embedding(x)

        # 4. Transformer Encoder
        x, attn_weights_before_mask, attn_weights_after_mask = self.transformer_encoder(x, msk) # x: torch.Size([5, 257, 16])


        # 5. MLP Head
        # x.shape : [batch_size, n_patches + 1, dim] -> [batch_size, dim] -> [batch_size, n_classes]
        # x = x[:, 0]
        x = self.mlp_head(x)

        return x, attn_weights_before_mask, attn_weights_after_mask





######################################################################################################
######################################################################################################
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, param_path='best-parameters.pt', model_path='best-model.pt'):
        """
        早期停止を行うクラス。

        Args:
            patience (int): モデルの改善が止まった後に待つエポック数。デフォルトは7。
            verbose (bool): Trueなら、早期停止をトリガーしたエポックを出力。デフォルトはFalse。
            delta (float): モデルの改善と見なされるための最小変化。デフォルトは0。
            param_path (str): ベストパラメータの保存ファイルパス。デフォルトは'best-parameters.pt'。
            model_path (str): ベストモデルの重みを保存するファイルパス。デフォルトは'best-model.pt'。
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.param_path = param_path
        self.model_path = model_path

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model and parameters...')
        torch.save(model.state_dict(), self.param_path)
        torch.save(model, self.model_path)
        self.val_loss_min = val_loss
######################################################################################################
######################################################################################################







######################################################################################################
######################################################################################################
for time in range(1, 2, 1):

    torch.backends.cudnn.benchmark = True

    dims = [16, 32, 64]
    depths = [4]
    n_heads = [4]
    mlp_dims = [32, 64, 128]
    dropouts = [0, 0.2, 0.4]
    lr_rates = [5e-4, 1e-4]
    optims = ["Adam", "SGD"]

    combinations = list(itertools.product(dims, depths, n_heads, mlp_dims, dropouts, lr_rates, optims))

    for i, (dim, depth, n_heads, mlp_dim, dropout, lr_rate, optim) in enumerate(combinations, start=1):
        print("-------------------------------------------------------------------------")
        print(f"Combination {i}: batch={batch_size}, dim={dim}, depth={depth}, n_heads={n_heads}, mlp_dim={mlp_dim}, dropout={dropout}, lr={lr_rate}, optim={optim}")
        print("-------------------------------------------------------------------------")

        best_score = 10000
        EPOCH = 300

        model = ViT(
            image_size=256,
            patch_size=16,
            n_classes=2,
            dim=dim,
            depth=depth,
            n_heads=n_heads,
            channels=3,
            mlp_dim=mlp_dim,
            dropout=dropout
        )

        weight_0_rate = round((len(with_pt_train) / total_images_train_set), 3)
        weight_1_rate = round((len(without_pt_train) / total_images_train_set), 3)
        weights = torch.tensor([0.5, 0.9]).cuda(1)
        loss_fn_1 = nn.CrossEntropyLoss(reduction='sum', weight=None)
        if optim == "Adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=lr_rate, weight_decay=1e-3)
        elif optim == "SGD":
            optimizer = torch.optim.SGD(model.parameters(), lr=lr_rate, weight_decay=1e-3)
        else:
            print("There is no Optimizer...")

        scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-5)

        model.to(device)

        file_name_0 = f"best-parameters_RETNET_{time}_batch{batch_size}_dims{dim}_depths{depth}_nheads{n_heads}_mlpdim{mlp_dim}_dropout{dropout}_lr{lr_rate}_optim{optim}.pt"
        file_name_1 = f"best-model_RETNET_{time}_batch{batch_size}_dims{dim}_depths{depth}_nheads{n_heads}_mlpdim{mlp_dim}_dropout{dropout}_lr{lr_rate}_optim{optim}.pt"
        param_path = f"{mode}/HCC/result_RetNet/{dir_name}/save_param/{file_name_0}"
        model_path = f"{mode}/HCC/result_RetNet/{dir_name}/save_param/{file_name_1}"
        early_stopping = EarlyStopping(patience=4, verbose=True, param_path=param_path, model_path=model_path)

        print(f"TIME:{time}")
        history_i = {
        'train_loss': [],
        'train_acc': [],
        'validation_loss': [],
        'validation_acc': []
        }

        accumulated_cm = np.zeros((2, 2), dtype=np.int64)

        train_loss = 0.0
        train_correct = 0
        train_total = 0

        validation_loss = 0.0
        validation_correct = 0
        validation_total = 0

        pridicted_value = []
        label_value = []
        TP = 0
        FP = 0
        FN = 0
        TN = 0
        AUC_ROC = 0
        label_list = []
        output_list = []
        positive_probability = []
        validation_labels = []
        predicted_labels = []

        scaler = torch.cuda.amp.GradScaler(init_scale=4096)

        for e in range(EPOCH):

            """ Training Part """
            model.train(True)
            with tqdm(trainloader) as pbar:
                pbar.set_description(f'[Epoch {e + 1}/{EPOCH}]')
                print(f'[Epoch {e + 1}/{EPOCH}]')
                for inputs, masks, labels in pbar:

                    inputs = inputs.to(device)
                    masks = masks.to(device)
                    labels = labels.to(device)
                    output, _, _ = model(inputs, masks)

                    """ CrossEntropyLoss """
                    output = torch.softmax(output, dim=1)
                    _, predicted = torch.max(output.data, 1)
                    loss_1 = loss_fn_1(output, labels)
                    loss = loss_1

                    train_loss += loss.item()
                    train_total += labels.size(0)
                    train_correct += (predicted == labels).sum().item()

                    # optimizer.zero_grad()
                    # loss.backward()
                    # optimizer.step()

                    for param in model.parameters():
                        param.grad = None
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    pbar.set_postfix(OrderedDict(Loss=loss.item(), Accuracy=torch.sum(
                        predicted == labels).item()/len(labels),))


            history_i['train_loss'].append(train_loss/train_total)
            history_i['train_acc'].append(train_correct/train_total)


            """ Validation Part """
            model.eval()

            with torch.no_grad():
                for data in validationloader:

                    inputs, masks, labels = data
                    inputs = inputs.to(device)
                    masks = masks.to(device)
                    labels = labels.to(device)
                    output, _, _ = model(inputs, masks)

                    """ CrossEntropyLoss """
                    output = torch.softmax(output, dim=1)
                    _, predicted = torch.max(output.data, 1)
                    loss_1 = loss_fn_1(output, labels)
                    loss = loss_1

                    ####################################################################################
                    predicted_ = predicted.to('cpu').detach().numpy().copy()
                    labels_ = labels.to('cpu').detach().numpy().copy()
                    cm = confusion_matrix(labels_, predicted_, labels=np.arange(2))
                    accumulated_cm += cm
                    ####################################################################################

                    label_value += labels.tolist()
                    validation_loss += loss.item()
                    pridicted_value += predicted.tolist()
                    validation_total += labels.size(0)
                    validation_correct += (predicted == labels).sum().item()

                    output_np = output.to('cpu').detach().numpy().copy()
                    label_np = labels.to('cpu').detach().numpy().copy()
                    label_np_ = predicted.to('cpu').detach().numpy().copy()

                    output_np = output.to('cpu').detach().numpy().copy()
                    label_np = labels.to('cpu').detach().numpy().copy()
                    label_np_ = predicted.to('cpu').detach().numpy().copy()

                    output_list = output_np.tolist() # [1]
                    label_list = label_np.tolist()   # [2]
                    label_list_ = label_np_.tolist() # [3]

                    # [1]
                    for i in output_list:
                            positive_probability.append(i[1])

                    # [2]
                    for i in label_list:
                        validation_labels.append(i)

                    # [3]
                    for i in label_list_:
                        predicted_labels.append(i)


            ####################################################################################
            # Assuming accumulated_cm is the accumulated confusion matrix
            TP = np.diag(accumulated_cm)
            FP = np.sum(accumulated_cm, axis=0) - TP
            FN = np.sum(accumulated_cm, axis=1) - TP
            TN = np.sum(accumulated_cm) - (TP + FP + FN)

            accuracy = (TP + TN) / np.sum(accumulated_cm)
            sensitivity = TP / (TP + FN)
            specificity = TN / (TN + FP)

            """
            # classごとの精度、感度、特異度
            for i in range(len(class_list)):
                print(f"Class {class_list[i]}:")
                print(f"  Accuracy: {accuracy[i]}")
                print(f"  Sensitivity (Recall): {sensitivity[i]}")
                print(f"  Specificity: {specificity[i]}")
                print()
            """

            # class全体の精度、感度、特異度
            overall_accuracy = np.sum(TP) / np.sum(accumulated_cm)
            overall_sensitivity = np.mean(sensitivity)
            overall_specificity = np.mean(specificity)

            """Image Base"""
            AUC_ROC = roc_auc_score(validation_labels, positive_probability)

            print("(I) validation loss = {0}, validation_acc = {1}, AUC_ROC = {2}".format(
                round(validation_loss/validation_total, 3), round(validation_correct/validation_total, 3), round(AUC_ROC, 3)))

            history_i['validation_loss'].append(validation_loss/validation_total)
            history_i['validation_acc'].append(validation_correct/validation_total)

            # Early stopping
            early_stopping(validation_loss/validation_total, model)
            if early_stopping.early_stop:
                print("------------- Early stopping -------------")
                break

            """
            print(f"---> {validation_loss}")
            if validation_loss < best_score:
                print('Break best Validation Loss!! saving the best model...')
                best_score = validation_loss
                torch.save(model, f"{mode}/HCC/result_RetNet/{dir_name}/save_param/best-model_RETNET.pt")
                torch.save(model.state_dict(), f{mode}./HCC/result_RetNet/{dir_name}/save_param/best-parameters_RETNET.pt")
            """

            # scheduler.step(validation_loss/validation_total)

            file_name = f"result_imagebase_{time}_batch{batch_size}_dims{dim}_depths{depth}_nheads{n_heads}_mlpdim{mlp_dim}_dropout{dropout}_lr{lr_rate}_optim{optim}.json"
            with open(f"{mode}/HCC/result_RetNet/{dir_name}/validation_imagebase/{file_name}", 'w') as outfile:
                json.dump(history_i, outfile, indent=4)

        torch.cuda.empty_cache()
######################################################################################################
######################################################################################################




######################################################################################################
######################################################################################################
def compute_attention_mask(att_mat, im_size):

    aug_att_mat = att_mat[0].unsqueeze(0)

    v = aug_att_mat[-1]

    # 1
    v_mean = v.mean(dim=0)
    v_mean = v_mean[1:]

    # 2
    grid_size = int(np.sqrt(aug_att_mat.size(-1)))
    prace = v[grid_size * grid_size, 1:] # Under triabngle matrix

    mask = v_mean.reshape(grid_size, grid_size).detach().cpu().numpy()
    mask_b = cv2.resize(mask / mask.max(), im_size)[..., np.newaxis]

    return mask_b



df = pd.read_csv(f'{mode}/HCC/csv_data/pt_list.csv')
vascular_invasion_dict = dict(zip(df['TCIA_ID'], df['Vascular invasion']))

for data in testloader:
    inputs, masks, labels, _ = data

    model = torch.load(f'{mode}/HCC/result_RetNet/{dir_name}/save_param/best-model_RETNET.pt')
    model.eval()
    model.to(device)

    inputs = inputs.to(device)
    masks = masks.to(device)
    labels = labels.to(device)

    # Plotting side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    inputs_ = inputs[0].cpu().numpy()
    masks_ = masks[0].cpu().numpy()

    inputs_ = np.transpose(inputs_, (1, 2, 0))
    inputs_ = np.mean(inputs_, axis=-1, keepdims=True)
    inputs_ = np.squeeze(inputs_)
    print(inputs_.shape)
    print(masks_.shape)

    ax1.imshow(inputs_, cmap='gray')
    ax1.set_title('Original')
    ax1.axis('off')

    ax2.imshow(masks_)
    ax2.set_title('Mask')
    ax2.axis('off')

    plt.savefig(f"{mode}/HCC/result_RetNet/{dir_name}/img/test.png")
    plt.show()

    logits, att_before, att_after = model(inputs, masks)
    print("kkkkkk")
    print(logits.shape)
    logits = logits.mean(dim=0, keepdim=True)

    #################################################### Before mask attention ####################################################
    att_mat = att_before.squeeze(1)
    b1 = att_mat[:, 0, :, :]
    b2 = att_mat[:, 1, :, :]
    b3 = att_mat[:, 2, :, :]
    b4 = att_mat[:, 3, :, :]
    b5 = torch.mean(att_mat, dim=1)

    im_size = (256, 256)
    b1 = compute_attention_mask(b1, im_size)
    b2 = compute_attention_mask(b2, im_size)
    b3 = compute_attention_mask(b3, im_size)
    b4 = compute_attention_mask(b4, im_size)
    b5 = compute_attention_mask(b5, im_size)

    fig = plt.figure(figsize=(16, 16))
    gs = GridSpec(1, 6, width_ratios=[1, 1, 1, 1, 0.05, 1])

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[0, 3])
    ax5 = fig.add_subplot(gs[0, 5])

    ax1.set_title('(Before Mask, Head:1)')
    ax2.set_title('(Before Mask, Head:2)')
    ax3.set_title('(Before Mask, Head:3)')
    ax4.set_title('(Before Mask, Head:4)')
    ax5.set_title('Average')

    ax1.imshow(b1[:, :, 0], cmap='viridis')
    ax1.axis('off')
    ax2.imshow(b2[:, :, 0], cmap='viridis')
    ax2.axis('off')
    ax3.imshow(b3[:, :, 0], cmap='viridis')
    ax3.axis('off')
    ax4.imshow(b4[:, :, 0], cmap='viridis')
    ax4.axis('off')
    ax5.imshow(b5[:, :, 0], cmap='viridis')
    ax5.axis('off')

    plt.show()

    #################################################### After mask attention ####################################################
    att_mat = att_after.squeeze(1)

    a1 = att_mat[:, 0, :, :]
    a2 = att_mat[:, 1, :, :]
    a3 = att_mat[:, 2, :, :]
    a4 = att_mat[:, 3, :, :]
    a5 = torch.mean(att_mat, dim=1)

    a1 = compute_attention_mask(a1, im_size)
    a2 = compute_attention_mask(a2, im_size)
    a3 = compute_attention_mask(a3, im_size)
    a4 = compute_attention_mask(a4, im_size)
    a5 = compute_attention_mask(a5, im_size)

    fig = plt.figure(figsize=(16, 16))
    gs = GridSpec(1, 6, width_ratios=[1, 1, 1, 1, 0.05, 1])

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[0, 3])
    ax5 = fig.add_subplot(gs[0, 5])

    ax1.set_title('(After Mask, Head:1)')
    ax2.set_title('(After Mask, Head:2)')
    ax3.set_title('(After Mask, Head:3)')
    ax4.set_title('(After Mask, Head:4)')
    ax5.set_title('Average')

    ax1.imshow(a1[:, :, 0], cmap='viridis')
    ax1.axis('off')
    ax2.imshow(a2[:, :, 0], cmap='viridis')
    ax2.axis('off')
    ax3.imshow(a3[:, :, 0], cmap='viridis')
    ax3.axis('off')
    ax4.imshow(a4[:, :, 0], cmap='viridis')
    ax4.axis('off')
    ax5.imshow(a5[:, :, 0], cmap='viridis')
    ax5.axis('off')

    plt.show()
######################################################################################################
######################################################################################################
