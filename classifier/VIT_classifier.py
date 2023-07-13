import os
import random
from PIL import Image
import torch
from torch._C import dtype
import torch.nn as nn
from torch.nn.modules.conv import Conv2d
import copy
import argparse
import sys
import torch.nn.functional as F
from tqdm import tqdm
import sys, os
import torch.optim as optim
import datetime, logging
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import cv2
import math


##
seed=42
im_s=224
base_learning_rate=3e-4
weight_decay=0.05
mask_ratio=0.75
total_epoch=30
warmup_epoch=5
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model_path='vit_mae.pt'
# output_dir="/home4/khanhnd/hagrid/MAE/output_pretrain/checkpoint"
batch_size=128
steps_per_update = 4
output_model_path="/home4/khanhnd/hagrid/MAE/output_cls_mae/checkpoint/classifier.ckpt"


################################################

import json
import logging
import os
import random
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch.utils.data
from PIL import Image, ImageOps
class Compose:
    def __init__(self, transforms: List[nn.Module]):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target
def get_crop_from_bbox(image: Image.Image, bbox: List, box_scale: float = 1.0) -> Tuple[Image.Image, np.array]:
    """
    Crop bounding box from image

    Parameters
    ----------
    image : Image.Image
        Source image for crop
    bbox : List
        Bounding box [xyxy]
    box_scale: float
        Scale for bounding box crop
    """
    int_bbox = np.array(bbox).round().astype(np.int32)

    x1 = int_bbox[0]
    y1 = int_bbox[1]
    x2 = int_bbox[2]
    y2 = int_bbox[3]
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

    w = h = max(x2 - x1, y2 - y1)
    x1 = max(0, cx - box_scale * w // 2)
    y1 = max(0, cy - box_scale * h // 2)
    x2 = cx + box_scale * w // 2
    y2 = cy + box_scale * h // 2
    x1, y1, x2, y2 = list(map(int, (x1, y1, x2, y2)))

    crop_image = image.crop((x1, y1, x2, y2))
    bbox_orig = np.array([x1, y1, x2, y2]).reshape(2, 2)

    return crop_image, bbox_orig
IMAGES = (".jpeg", ".jpg", ".jp2", ".png", ".tiff", ".jfif", ".bmp", ".webp", ".heic")

class GestureDataset(torch.utils.data.Dataset):
    """
    Custom Dataset for gesture classification pipeline
    """

    def __init__(self, is_train: bool=False,transform: Compose = None, is_test: bool = False):
        super().__init__()
        self.dataset_path="/home4/khanhnd/hagrid/hagrid-sample-500k-384p/hagrid_500k"
        self.annotations="/home4/khanhnd/hagrid/hagrid-sample-500k-384p/ann_train_val"
        self.targets=["dislike","like","mute","ok","palm","peace","stop","two_up","no_gesture"]
        self.transform = transform
        self.is_train = is_train
        self.random_state=42
        self.image_size=[224,224]
        self.labels = {
            label: num for (label, num) in zip(self.targets, range(len(self.targets)))
        }

        subset = -1

        self.annotations = self.__read_annotations(subset)

        users = self.annotations["user_id"].unique()
        random.Random(self.random_state).shuffle(users)

        self.annotations = self.annotations.copy()
        if is_train:
                self.annotations = self.annotations[:int(len(self.annotations)*0.8)]
        else:
                self.annotations = self.annotations[int(len(self.annotations)*0.8):]
    def __len__(self):
        return self.annotations.shape[0]

    def __getitem__(self, index: int):
        """
        Get item from annotations

        Parameters
        ----------
        index : int
            Index of annotation item
        """
        row = self.annotations.iloc[[index]].to_dict("records")[0]

        image_resized, gesture = self.__prepare_image_target(
            row["target"], row["name"], row["bboxes"], row["labels"]
        )

        label = self.labels[gesture]
        
        if self.transform is not None:
            image_resized, label = self.transform(image_resized, label)
        image_resized.save("/home4/khanhnd/hagrid/bruh1.png")
        image_resized=transforms.ToTensor()(image_resized)
        return image_resized, label

    @staticmethod
    def __get_files_from_dir(pth: str, extns: Tuple, subset: int = None):
        """
        Get list of files from dir according to extensions(extns)

        Parameters
        ----------
        pth : str
            Path ot dir
        extns: Tuple
            Set of file extensions
        subset : int
            Length of subset for each target
        """
        if not os.path.exists(pth):
            logging.warning(f"Dataset directory doesn't exist {pth}")
            return []
        files = [f for f in os.listdir(pth) if f.endswith(extns)]
        if (subset is not None) and (subset > 0):
            files = files[:subset]
        return files

    def __read_annotations(self, subset: int = None):
        """
        Read annotations json

        Parameters
        ----------
        subset : int
            Length of subset for each target
        """
        exists_images = []
        annotations_all = pd.DataFrame()
        path_to_json = os.path.expanduser(self.annotations)
        all_real_target=self.targets
        all_target=[i for i in os.listdir("/home4/khanhnd/hagrid/hagrid-sample-500k-384p/hagrid_500k")]
        
        all_fake_target=[i for i in all_target if i not in all_real_target]
        for target in all_target:
            target_tsv = os.path.join(path_to_json, f"{target}.json")
            if os.path.exists(target_tsv):
                json_annotation = json.load(open(os.path.join(path_to_json, f"{target}.json")))

                json_annotation = [
                    dict(annotation, **{"name": f"{name}.jpg"})
                    for name, annotation in zip(json_annotation, json_annotation.values())
                ]
                annotation = pd.DataFrame(json_annotation)
                annotation["target"] = target
                annotations_all = pd.concat([annotations_all, annotation], ignore_index=True)
                exists_images.extend(
                    self.__get_files_from_dir(os.path.join(self.dataset_path, ""+target), IMAGES, subset)
                )
            else:
                logging.info(f"Database for {target} not found")

        annotations_all["exists"] = annotations_all["name"].isin(exists_images)
        final_all=annotations_all[annotations_all["exists"]]
        total=0
        final_fake_list=[]
        fake_data=final_all[~final_all["target"].isin(all_real_target)].copy()
        final_all=final_all[final_all["target"].isin(all_real_target)].copy()
        def convert(t):
            return ["no_gesture" for j in range(len(t))]
        for df_fake in all_fake_target:
            temp=fake_data[fake_data["target"]==df_fake].copy()
            
            temp["labels"]=temp["labels"].apply(convert)
            temp=temp.sample(5000,random_state=42)
            final_fake_list.append(temp)
        final_fake_all=pd.concat(final_fake_list)

        final_final=pd.concat([final_fake_all,final_all])
        final_final=final_final.sample(frac=1,random_state=42)


        return final_final

    def __prepare_image_target(
        self, target: str, name: str, bboxes: List, labels: List
    ):
        """
        Crop and padding image, prepare target

        Parameters
        ----------
        target : str
            Class name
        name : str
            Name of image
        bboxes : List
            List of bounding boxes [xywh]
        labels: List
            List of labels
        """
        image_pth = os.path.join(self.dataset_path, ""+target, name)

        image = Image.open(image_pth).convert("RGB")

        width, height = image.size

        choice = np.random.choice(["gesture", "no_gesture"], p=[0.7, 0.3])

        bboxes_by_class = {}

        for i, bbox in enumerate(bboxes):
            x1, y1, w, h = bbox
            bbox_abs = [x1 * width, y1 * height, (x1 + w) * width, (y1 + h) * height]
            if labels[i] == "no_gesture":
                bboxes_by_class["no_gesture"] = (bbox_abs, labels[i])
            else:
                bboxes_by_class["gesture"] = (bbox_abs, labels[i])

        if choice not in bboxes_by_class:
            choice = list(bboxes_by_class.keys())[0]

        if self.is_train:
            box_scale = np.random.uniform(low=1.0, high=2.0)
        else:
            box_scale = 1.0

        image_cropped, bbox_orig = get_crop_from_bbox(image, bboxes_by_class[choice][0], box_scale=box_scale)

        image_resized = ImageOps.pad(image_cropped, tuple(self.image_size), color=(0, 0, 0))

        gesture = bboxes_by_class[choice][1]


        return image_resized, gesture


#################################################
train_dataset=GestureDataset(is_train=True,transform= None)
test_dataset=GestureDataset( is_test=True,transform= None)

#################################################
train_loader=DataLoader(train_dataset, batch_size=batch_size,
                        
                        shuffle=True, num_workers=8)
test_loader=DataLoader(test_dataset, batch_size=batch_size,
                        shuffle=True, num_workers=8)

###############################################
import torch
import timm
import numpy as np

from einops import repeat, rearrange
from einops.layers.torch import Rearrange

from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import Block

def random_indexes(size : int):
    forward_indexes = np.arange(size)
    np.random.shuffle(forward_indexes)
    backward_indexes = np.argsort(forward_indexes)
    return forward_indexes, backward_indexes

def take_indexes(sequences, indexes):
    return torch.gather(sequences, 0, repeat(indexes, 't b -> t b c', c=sequences.shape[-1]))

class PatchShuffle(torch.nn.Module):
    def __init__(self, ratio) -> None:
        super().__init__()
        self.ratio = ratio

    def forward(self, patches : torch.Tensor):
        T, B, C = patches.shape
        remain_T = int(T * (1 - self.ratio))

        indexes = [random_indexes(T) for _ in range(B)]
        forward_indexes = torch.as_tensor(np.stack([i[0] for i in indexes], axis=-1), dtype=torch.long).to(patches.device)
        backward_indexes = torch.as_tensor(np.stack([i[1] for i in indexes], axis=-1), dtype=torch.long).to(patches.device)

        patches = take_indexes(patches, forward_indexes)
        patches = patches[:remain_T]

        return patches, forward_indexes, backward_indexes

class MAE_Encoder(torch.nn.Module):
    def __init__(self,
                 image_size=224,
                 patch_size=32,
                 emb_dim=768,
                 num_layer=4,
                 num_head=4,
                 mask_ratio=0.75,
                 ) -> None:
        super().__init__()

        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.zeros((image_size // patch_size) ** 2, 1, emb_dim))
        self.shuffle = PatchShuffle(mask_ratio)

        self.patchify = torch.nn.Conv2d(3, emb_dim, patch_size, patch_size)

        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])

        self.layer_norm = torch.nn.LayerNorm(emb_dim)

        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, img):
        patches = self.patchify(img)
        patches = rearrange(patches, 'b c h w -> (h w) b c')
        patches = patches + self.pos_embedding

        patches, forward_indexes, backward_indexes = self.shuffle(patches)

        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)
        patches = rearrange(patches, 't b c -> b t c')
        features = self.layer_norm(self.transformer(patches))
        features = rearrange(features, 'b t c -> t b c')

        return features, backward_indexes

class MAE_Decoder(torch.nn.Module):
    def __init__(self,
                 image_size=224,
                 patch_size=32,
                 emb_dim=128,
                 num_layer=4,
                 num_head=4,
                 ) -> None:
        super().__init__()

        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.zeros((image_size // patch_size) ** 2 + 1, 1, emb_dim))

        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head,attn_drop=0.2) for _ in range(num_layer)])

        self.head = torch.nn.Linear(emb_dim, 3 * patch_size ** 2)
        self.patch2img = Rearrange('(h w) b (c p1 p2) -> b c (h p1) (w p2)', p1=patch_size, p2=patch_size, h=image_size//patch_size)

        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.mask_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, features, backward_indexes):
        T = features.shape[0]
        backward_indexes = torch.cat([torch.zeros(1, backward_indexes.shape[1]).to(backward_indexes), backward_indexes + 1], dim=0)
        features = torch.cat([features, self.mask_token.expand(backward_indexes.shape[0] - features.shape[0], features.shape[1], -1)], dim=0)
        features = take_indexes(features, backward_indexes)
        features = features + self.pos_embedding

        features = rearrange(features, 't b c -> b t c')
        features = self.transformer(features)
        features = rearrange(features, 'b t c -> t b c')
        features = features[1:] # remove global feature

        patches = self.head(features)
        mask = torch.zeros_like(patches)
        mask[T:] = 1
        mask = take_indexes(mask, backward_indexes[1:] - 1)
        img = self.patch2img(patches)
        mask = self.patch2img(mask)

        return img, mask

class MAE_ViT(torch.nn.Module):
    def __init__(self,
                 image_size=224,
                 patch_size=32,
                 emb_dim=480,
                 encoder_layer=12,
                 encoder_head=12,
                 decoder_layer=4,
                 decoder_head=4,
                 mask_ratio=0.75,
                 ) -> None:
        super().__init__()
        self.encoder = MAE_Encoder(image_size, patch_size, emb_dim, encoder_layer, encoder_head, mask_ratio)
        self.decoder = MAE_Decoder(image_size, patch_size, emb_dim, decoder_layer, decoder_head)

    def forward(self, img):
        features, backward_indexes = self.encoder(img)
        predicted_img, mask = self.decoder(features,  backward_indexes)
        return predicted_img, mask

class ViT_Classifier(torch.nn.Module):
    def __init__(self, encoder : MAE_Encoder, num_classes=9) -> None:
        super().__init__()
        self.cls_token = encoder.cls_token
        self.pos_embedding = encoder.pos_embedding
        self.patchify = encoder.patchify
        self.transformer = encoder.transformer
        self.layer_norm = encoder.layer_norm
        self.head = torch.nn.Linear(self.pos_embedding.shape[-1], num_classes)

    def forward(self, img):
        patches = self.patchify(img)
        patches = rearrange(patches, 'b c h w -> (h w) b c')
        patches = patches + self.pos_embedding
        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)
        patches = rearrange(patches, 't b c -> b t c')
        features = self.layer_norm(self.transformer(patches))
        features = rearrange(features, 'b t c -> t b c')
        logits = self.head(features[0])
        return logits
    
#############################################



if __name__=="__main__":
    print(device)
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_mae = MAE_ViT(mask_ratio=mask_ratio).to(device)
    # model_mae.load_state_dict(torch.load("/home4/khanhnd/hagrid/MAE/output_pretrain/vit_mae.pt_16"))
    writer = SummaryWriter(os.path.join("/home4/khanhnd/hagrid/MAE/output_cls_mae",'logs', 'classifier'))
    model = ViT_Classifier(model_mae.encoder, num_classes=9).to(device)
    model.load_state_dict(torch.load("/home4/khanhnd/hagrid/MAE/output_cls_mae/checkpoint/classifier.ckpt"))
    print("NUMBER OF PARAMS: ", count_parameters(model))
    from sklearn.metrics import f1_score
    loss_fn = torch.nn.CrossEntropyLoss()
    f1_fn = lambda logit, label: torch.mean((logit.argmax(dim=-1) == label).float())

    optim = torch.optim.AdamW(model.parameters(), lr=base_learning_rate * batch_size / 256, betas=(0.9, 0.999), weight_decay=weight_decay)
    lr_func = lambda epoch: min((epoch + 1) / (warmup_epoch + 1e-8), 0.5 * (math.cos(epoch / total_epoch * math.pi) + 1))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_func, verbose=True)

    best_val_f1 = 0
    step_count = 0
    optim.zero_grad()

            
    for e in range(total_epoch):
        model.train()
        losses = []
        f1es = []
        count_dict=dict()
        for img, label in tqdm(iter(train_loader)):
            for lab in label:
                count_dict[int(lab)]=count_dict.get(int(lab),0)+1
            step_count += 1
            img = img.to(device)
            label = label.to(device)
            logits = model(img)
            loss = loss_fn(logits, label)
            f1 = f1_fn(logits.cpu(), label.cpu())
            loss.backward()
            if step_count % steps_per_update == 0:
                optim.step()
                optim.zero_grad()
            losses.append(loss.item())
            f1es.append(f1.item())
        print("DISTRIBUTION", count_dict)
        lr_scheduler.step()
        avg_train_loss = sum(losses) / len(losses)
        avg_train_f1 = sum(f1es) / len(f1es)
        print(f'In epoch {e}, average training loss is {avg_train_loss}, average training f1 is {avg_train_f1}.')
        
        model.eval()
        with torch.no_grad():
            losses = []
            f1es = []
            for img, label in tqdm(iter(test_loader)):
                img = img.to(device)
                label = label.to(device)
                logits = model(img)
                loss = loss_fn(logits, label)
                f1 = f1_fn(logits.cpu(), label.cpu())
                losses.append(loss.item())
                f1es.append(f1.item())
            avg_val_loss = sum(losses) / len(losses)
            avg_val_f1 = sum(f1es) / len(f1es)
            print(f'In epoch {e}, average validation loss is {avg_val_loss}, average validation f1 is {avg_val_f1}.')  

        if avg_val_f1 > best_val_f1:
            best_val_f1 = avg_val_f1
            print(f'saving best model with f1 {best_val_f1} at {e} epoch!')       
            torch.save(model.state_dict(), output_model_path)

        writer.add_scalars('cls/loss', {'train' : avg_train_loss, 'val' : avg_val_loss}, global_step=e)
        writer.add_scalars('cls/f1', {'train' : avg_train_f1, 'val' : avg_val_f1}, global_step=e)



