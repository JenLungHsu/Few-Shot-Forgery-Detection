# Few-Shot-Forgery-Detection

[![Semester](https://img.shields.io/badge/Semester-Spring%202024-blue)]() [![Project](https://img.shields.io/badge/Project-Deep%20Learning%20Final%20Presentation-orange)]()

🚀 Check out the [slide](document/Slide.pdf) or [report](document/Report.pdf) for more detail.

## Goal
We aims to 
1. Classify if a video is real or fake
2. Disscuss transfer learning performance on few-shot sample finetuning


## Dataset (FaceForensics++ & Celeb-DF)
![dataset](src/dataset.png)

## Detection Pipeline
![detection pipeline](src/detection_pipeline.png)

## Proposed Pipeline
![proposed pipeline](src/proposed_pipeline.png)


## Training Phase

We first split the video into frames of images array, and then label the images with the same label as the video. 

We then randomize the order of the images and split the images into training set and validation set.

![training phase](src/training_phase.png)

## Testing Phase

In the testing phase, we feed the testing video into the model and get the prediction of each frame. We then calculate the average prediction of all frames in the video and use it as the final prediction of the video.

![testing phase](src/testing_phase.png)

## Experiment

### Detail
- model pretrained on face forensics ++  c23
    - models: MesoNet(2018), XceptionNet(2016), EfficientNet(2019)
- finetune 1%, 5%, 10%, 50%, 100% Celeb-DF
    - training set: 1%, 5%, 10%, 50%, 100% 
    - validation set: 1%
    - testing set: all Celeb-DF official testing set
- Hyperparameter
    - Loss: Cross Entropy
    - Optimizer: Adam
    - Learning rate: 0.001
    - Scheduler: StepLR, step size=5
    - Epoch
    
    |Shot|1%|5%|10%|50%|100%|
    |---|---|---|---|---|---|
    |Epoch|100|100|100|50|50|

### Transfer Learning
![transfer learning](src/transfer_learning.png)


### Few Shot Learning
#### MesoNet
![mesonet](src/mesonet.png)
#### Xception
![xception](src/xception.png)
#### EfficientNetB4
![efficientnet](src/efficientnet.png)

### Model Comparison on differnt Metrics
#### F1-Score

![f1-score](src/f1-score.png)

#### Accuracy

![accuracy](src/accuracy.png)

#### AUC

![auc](src/auc.png)

## Conclusion
- EfficientNet outperforms the other two models when testing few-shot samples

- The difference between real and fake images is really small, it’s very hard to distinguish them by using simple convolutional feature extraction method.

- We think Residual component plays an important role in this task
