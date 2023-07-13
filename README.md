# cv
## Installation
To install this extension, first you need to clone this repo
```
pip install -r requirements.txt
```
To download detector and classifier checkpoint, click [here](https://drive.google.com/drive/folders/1t1XHg6xevYqCHeHCy03jdX9NrsUZqm4W?usp=sharing)
## Usage
1. You need to run this command to open API:
```
python api.py --detector_path path/to/detector/checkpoint --classifier_path path/to/classifier/checkpoint
```
2. Open chrome browser extension and open developer mode. Click `Load unpacked` and select the directory you cloned  
![image](https://drive.google.com/uc?export=view&id=1L-Fu2zoda0VffFbXr66_nugSXV5lK89d)
3. Open a random youtube video and click the extension icon -> Movie Controller  
![image](https://drive.google.com/uc?export=view&id=1ewYCjjo3tnpOmGhBHwjJa4tjMx19P_ca)
4. Tick to checkbox and enjoy your video  
![image](https://drive.google.com/uc?export=view&id=1qiTBhaV52ofvT49G7W1NwhKMZN_dRHCn)

## Training resnet18, resnext101, mobilenetv3_smal, vit32
1. To start training, run:
   ```
   python -m classifier.run --command 'train' --path_to_config classifier/config/*.yaml
   ```
## Training MAE VIT
1. If you want to run ViT without MAE pretrain, just run
   ```
   python VIT_classifier.py
   ```
2. If you want to run ViT with MAE pretrain, 
   First, run:
   ```
   python MAE.py
   ```
   After finishing training, load the MAE checkpoint in the VIT_classifier.py file, and run:
   ```
   python VIT_classifier.py
   ```
