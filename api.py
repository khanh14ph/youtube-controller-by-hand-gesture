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

from constants import targets
from detector.model import TorchVisionModel
from detector.ssd_mobilenetv3 import SSDMobilenet
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
        return img_tensor, (width, height), (padded_width, padded_height)

def preprocess(path):
        """
        Preproc image for model input
        Parameters
        ----------
        img: np.ndarray
            input image
        """
        img=cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(img)
        width, height = image.size

        image = ImageOps.pad(image, (max(width, height), max(width, height)))
        padded_width, padded_height = image.size
        image = image.resize((224, 224))

        img_tensor = f.pil_to_tensor(image)
        img_tensor = f.convert_image_dtype(img_tensor)
        img_tensor = img_tensor[None, :, :, :]
        img_tensor=img_tensor.to(conf_classifier.device)
        return img_tensor, (width, height), (padded_width, padded_height)


conf_classifier = OmegaConf.load("/home/huynv/cv/classifier/config/default.yaml")
classifier_model=utils.build_model(
        model_name=conf_classifier.model.name,
        num_classes=len(targets),
        checkpoint=conf_classifier.model.get("checkpoint", None),
        device=conf_classifier.device,
        pretrained=conf_classifier.model.pretrained,
        freezed=conf_classifier.model.freezed,
        ff=conf_classifier.model.full_frame,
    )



classifier_model.eval()


app = FastAPI()

start_str = 'data:image/png;base64,'
lens = len(start_str)

def stringToRGB(base64_string):
    if base64_string[:lens] == start_str:
        base64_string= base64_string[lens:]
    img_data = base64.b64decode(str(base64_string))
    image = Image.open(BytesIO(img_data))
    return cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)


gestures = [
    "dislike",
    "like",
    "mute",
    "ok",
    "palm",
    "peace",
    "stop",
    "two_up",
    "no_gesture"
]

detector = SSDMobilenet(num_classes=20)
detector.load_state_dict("/home/vuhl/cv/cv/Computer-Vision/SSDLite.pth", map_location=conf_classifier.device)
detector.eval()

img_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "img", "receive.png")
full_img_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "img", "receive_full.png")

@app.post("/detector")
async def get_bounding_box(request: Request):

    NUM_HANDS = 100
    json_file: bytes = await request.json()
    imageSrc = json_file['imageSrc']
    image_np = stringToRGB(imageSrc)
    # cv2.imwrite(full_img_path, image_np)

    processed_frame, size, padded_size = preprocess_from_array(image_np)

    with torch.no_grad():
        output = detector(processed_frame)[0]
    boxes = output["boxes"][:NUM_HANDS]
    scores = output["scores"][:NUM_HANDS]
            
    scores = scores[:min(NUM_HANDS, len(boxes))]
    i = np.argmax(scores)
    rx = ry = rw = rh = 0
    if scores[i] > 0.1:
        width, height = size
        padded_width, padded_height = padded_size
        scale = max(width, height) / 320

        padding_w = abs(padded_width - width) // (2 * scale)
        padding_h = abs(padded_height - height) // (2 * scale)

        x1 = int((boxes[i][0] - padding_w) * scale)
        y1 = int((boxes[i][1] - padding_h) * scale)
        x2 = int((boxes[i][2] - padding_w) * scale)
        y2 = int((boxes[i][3] - padding_h) * scale)

        rx = x1 / image_np.shape[1]
        ry = y1 / image_np.shape[0]
        rw = (x2 - x1) / image_np.shape[1]
        rh = (y2 - y1) / image_np.shape[0]
    res = {"rx":str(rx), "ry":str(ry), "rw":str(rw), "rh":str(rh)}
    return res

@app.post("/movie_controller")
async def detect(request: Request):

    json_file: bytes = await request.json()
    imageSrc = json_file['imageSrc']
    image_np = stringToRGB(imageSrc)
    # cv2.imwrite(full_img_path, image_np)

    NUM_HANDS = 100
    processed_frame, size, padded_size = preprocess_from_array(image_np)
    with torch.no_grad():
        output = detector(processed_frame)[0]
    boxes = output["boxes"][:NUM_HANDS]
    scores = output["scores"][:NUM_HANDS]
    
    scores = scores[:min(NUM_HANDS, len(boxes))]
    i = np.argmax(scores)

    if scores[i] > 0.1:
        width, height = size
        padded_width, padded_height = padded_size
        scale = max(width, height) / 320

        padding_w = abs(padded_width - width) // (2 * scale)
        padding_h = abs(padded_height - height) // (2 * scale)

        x1 = int((boxes[i][0] - padding_w) * scale)
        y1 = int((boxes[i][1] - padding_h) * scale)
        x2 = int((boxes[i][2] - padding_w) * scale)
        y2 = int((boxes[i][3] - padding_h) * scale)
        # Crop the region within the bounding box
        cropped_frame = image_np[y1:y2, x1:x2]

        cv2.imwrite(img_path, cropped_frame)

        processed_frame, size, padded_size = preprocess(img_path)
        with torch.no_grad():
            output = classifier_model(processed_frame)
        output = torch.argmax(output["gesture"])

        return {"content": f"{gestures[output]}"}
        
    return {"content": "no_gesture"}

if __name__ == "__main__":
    uvicorn.run(app, port=8000)