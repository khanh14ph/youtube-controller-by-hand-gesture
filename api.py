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
import argparse
import logging
import time
from typing import Optional, Tuple

import albumentations as A
import cv2
import mediapipe as mp
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

from detector.model import TorchVisionModel
from detector.ssd_mobilenetv3 import SSDLiteMobilenet_small
from classifier import utils

current_folder = os.path.dirname(os.path.realpath(__file__))

IMAGES = (".jpeg", ".jpg", ".jp2", ".png", ".tiff", ".jfif", ".bmp", ".webp", ".heic")
threshold=0.5
targets = {
    1: "dislike",
    2: "like",
    3: "mute",
    4: "ok",
    5: "palm",
    6: "peace",
    7: "stop",
    8: "two up",
    9: "no gesture",
    
}

parser = argparse.ArgumentParser(description='Model Live')
parser.add_argument("--detector_path", default="detector.pth", type=str)
parser.add_argument("--classifier_path", default="classifier.pth", type=str)
args = parser.parse_args()
conf = OmegaConf.load("mobilenet_large_v3_config.yaml")
from detector.ssd_mobilenetv3 import SSDLiteMobilenet_large
detectors_list = {
    "SSDLiteMobileNetV3Large": SSDLiteMobilenet_large,
}

def build_model(config):
    model_name = config.model.name
    model_config = {"num_classes": 34, "pretrained": config.model.pretrained}
    if model_name in detectors_list:
        model_config["num_classes"] += 1
        model_config.update(
            {
                "pretrained_backbone": config.model.pretrained_backbone,
                "img_size": config.dataset.img_size,
                "img_mean": config.dataset.img_mean,
                "img_std": config.dataset.img_std,
            }
        )
        model = detectors_list[model_name](**model_config)
        model.type = "detector"
    else:
        raise Exception(f"Unknown model {model_name}")
    return model
detector = build_model(conf)
def get_transform_for_inf(transform_config):
        """
        Create list of transforms from config
        Parameters
        ----------
        transform_config: DictConfig
            config with test transforms
        """
        transforms_list = [getattr(A, key)(**params) for key, params in transform_config.items()]
        transforms_list.append(ToTensorV2())
        return A.Compose(transforms_list)
transform = get_transform_for_inf(conf.test_transforms)
if conf.model.checkpoint is not None:
        snapshot = torch.load(conf.model.checkpoint, map_location=torch.device("cpu"))
        detector.load_state_dict(snapshot["MODEL_STATE"])
detector.eval()
def preprocess(img: np.ndarray, transform) -> Tuple[Tensor, Tuple[int, int], Tuple[int, int]]:
        """
        Preproc image for model input
        Parameters
        ----------
        img: np.ndarray
            input image
        transform :
            albumentation transforms
        """
        height, width = img.shape[0], img.shape[1]
        transformed_image = transform(image=img)
        processed_image = transformed_image["image"] / 255.0

        return processed_image, (width, height)
def preprocess_classifier(path):
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
        image = image.resize((224,224))

        img_tensor = f.pil_to_tensor(image)
        img_tensor = f.convert_image_dtype(img_tensor)
        img_tensor = img_tensor[None, :, :, :]
        return img_tensor, (width, height), (padded_width, padded_height)


conf_classifier = OmegaConf.load(os.path.join(current_folder, "classifier", "config", "default.yaml"))
classifier_model=utils.build_model(
        model_name=conf_classifier.model.name,
        num_classes=len(targets),
        checkpoint=args.classifier_path,
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



img_folder = os.path.join(current_folder, "img")
img_path = os.path.join(img_folder, "receive.png")
import cv2
@app.post("/detector")
async def get_bounding_box(request: Request):

    NUM_HANDS = 100
    json_file: bytes = await request.json()
    imageSrc = json_file['imageSrc']
    image_np = stringToRGB(imageSrc)

    processed_frame, size, padded_size = preprocess_from_array(image_np)
    processed_image, _ = preprocess(image_np,transform)
    with torch.no_grad():
        output = detector([processed_image])[0]
    boxes = output["boxes"][:NUM_HANDS]
    scores = output["scores"][:NUM_HANDS]
            
    scores = scores[:min(NUM_HANDS, len(boxes))]
    i = np.argmax(scores)
    rx = ry = rw = rh = 0
    if scores[i] > threshold:
        width, height = size
        padded_width, padded_height = padded_size
        scale = max(width, height) / 224

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

    NUM_HANDS = 100
    processed_image, size = preprocess(image_np,transform)
    with torch.no_grad():
        output = detector([processed_image])[0]
    boxes = output["boxes"][:NUM_HANDS]
    scores = output["scores"][:NUM_HANDS]
    
    scores = scores[:min(NUM_HANDS, len(boxes))]
    i = np.argmax(scores)

    _, size, padded_size = preprocess_from_array(image_np)
    if scores[i] > threshold:
        width, height = size
        padded_width, padded_height = padded_size
        scale = max(width, height) / 224

        padding_w = abs(padded_width - width) // (2 * scale)
        padding_h = abs(padded_height - height) // (2 * scale)

        x1 = int((boxes[i][0] - padding_w) * scale)
        y1 = int((boxes[i][1] - padding_h) * scale)
        x2 = int((boxes[i][2] - padding_w) * scale)
        y2 = int((boxes[i][3] - padding_h) * scale)
        # Crop the region within the bounding box
        cropped_frame = image_np[y1:y2, x1:x2]

        cv2.imwrite(img_path, cropped_frame)

        processed_frame, size, padded_size = preprocess_classifier(img_path)
        with torch.no_grad():
            output = classifier_model(processed_frame)
        output = torch.argmax(output["gesture"])
        print(gestures[output])
        return {"content": f"{gestures[output]}"}
        
    return {"content": "no_gesture"}

if __name__ == "__main__":
    uvicorn.run(app, port=8000)
