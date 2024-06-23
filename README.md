# Few-Shot-Forgery-Detection
## 112-2 Deep Learning Final Presentation


### We aims to 
#### Classify if a video is real or fake
#### Disscuss transfer learning performance on few-shot sample finetuning


## Dataset (FaceForensics++ & Celeb-DF)
![image](https://github.com/JenLungHsu/Few-Shot-Forgery-Detection/assets/79786516/731ba777-27ba-4518-92c7-18805b2c5c83)

---

## Detection Pipeline
![image](https://github.com/JenLungHsu/Few-Shot-Forgery-Detection/assets/79786516/3108df90-8072-4610-ac27-1cf73f5c4875)

## Proposed Pipeline
![image](https://github.com/JenLungHsu/Few-Shot-Forgery-Detection/assets/79786516/0093156c-9189-4173-a603-2152803680a0)


## Training Phase
![image](https://github.com/JenLungHsu/Few-Shot-Forgery-Detection/assets/79786516/0ec914ea-fef4-4980-8589-a9760d5938e1)

## Testing Phase
![image](https://github.com/JenLungHsu/Few-Shot-Forgery-Detection/assets/79786516/8bca1d16-ee18-4c62-90fe-ca080e6475a7)

---

## Experiment

### Detail
* model pretrained on face forensics ++  c23
* models use MesoNet(2018), XceptionNet(2016), EfficientNet(2019)
* finetune 1%, 5%, 10%, 50%, 100% Celeb-DF
    * training set: 1%, 5%, 10%, 50%, 100% 
    * validation set: 1%
    * testing set: all Celeb-DF official testing set
* Hyperparameter
    * Loss: Cross Entropy
    * Optimizer: Adam
    * Learning rate: 0.001
    * Scheduler: StepLR, step size=5
    * Epoch
![image](https://github.com/JenLungHsu/Few-Shot-Forgery-Detection/assets/79786516/0a59f681-0fdf-4117-9327-ee3d6a2c1423)

### Transfer Learning
![image](https://github.com/JenLungHsu/Few-Shot-Forgery-Detection/assets/79786516/95f87249-9135-461d-9e77-08ab13ab7c5c)


### Few Shot Learning
#### MesoNet
![image](https://github.com/JenLungHsu/Few-Shot-Forgery-Detection/assets/79786516/af033be1-b8e8-4423-9433-928b0448a3c7)
#### Xception
![image](https://github.com/JenLungHsu/Few-Shot-Forgery-Detection/assets/79786516/82aeb17f-190c-4c44-b8b4-be230434f8c6)
#### EfficientNetB4
![image](https://github.com/JenLungHsu/Few-Shot-Forgery-Detection/assets/79786516/d1a91e58-6d2a-47ec-a988-8e5049493931)

### Model Comparison on F1-Score
#### F1-Score

![image](https://github.com/JenLungHsu/Few-Shot-Forgery-Detection/assets/79786516/bf293d98-f40c-4d95-8066-4a9e81f0efe3)

#### Accuracy

![image](https://github.com/JenLungHsu/Few-Shot-Forgery-Detection/assets/79786516/262aad53-befa-4be1-a4d3-03a54916afad)

#### AUC

![image](https://github.com/JenLungHsu/Few-Shot-Forgery-Detection/assets/79786516/b881b063-341e-4278-baf7-f160b927efdb)

