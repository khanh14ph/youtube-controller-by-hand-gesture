from fastapi import FastAPI, Request
import uvicorn, numpy as np
import random, cv2
from io import BytesIO
import base64, time
from PIL import Image
import argparse
import logging
import time
from typing import Optional, Tuple
import os
import cv2
import mediapipe as mp
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image, ImageOps
from torch import Tensor
from torchvision.transforms import functional as f

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

from detector.model import TorchVisionModel
from detector.ssd_mobilenetv3 import SSDLiteMobilenet_small
from classifier import utils
def preprocess_from_array(img: np.ndarray) -> Tuple[Tensor, Tuple[int, int], Tuple[int, int]]:
        """
        Preproc image for model input
        Parameters
        ----------
        img: np.ndarray
            input image
        """
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(img)

        width, height = image.size

        image = ImageOps.pad(image, (max(width, height), max(width, height)))

        padded_width, padded_height = image.size
        image = image.resize((320, 320))
        
        img_tensor = f.pil_to_tensor(image)
        img_tensor = f.convert_image_dtype(img_tensor)
                                    
        img_tensor = img_tensor[None, :, :, :]
        return img_tensor,(width, height), (padded_width, padded_height)
if __name__=="__main__":
    img_array=cv2.imread("/home/msi/Documents/youtube-controller-by-hand-gesture/temp.png")
    preprocess_from_array(img_array)
    cv2.imwrite("temp2.png",img_array)