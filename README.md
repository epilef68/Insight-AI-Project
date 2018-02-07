# Insight-AI-Project
This project demonstrates the different speed accuracy tradeoffs for different stages of the OpenPose model on an indoor living room environment. OpenPose stages are used to achieve both local information in the form of joint heatmaps and global information in the form of limb vectors fields ie. part affinity fields (paf). These fields are concatenated at the end of each stage and input into the next stage, by removing stages less global information is used and greater weight is placed on local information. This means that for unobstructed joints and low number of people accuracy is not lost with decreasing number of stages.
# Setup
# Prerequisites
* Python 3
* Coco API
* OpenCV
* Keras
* Tensorflow
* Numpy

# Model
Several modifications are made to OpenPose including removing stages 1-6 as well as using the original VGG16 as the feature extractor and Inception.

# Datasets
2017 Coco dataset is used for training, while testing is done on the consulting companies in home images (contact for testing data). 

# Citations
```
@inproceedings{cao2017realtime,
  author = {Zhe Cao and Tomas Simon and Shih-En Wei and Yaser Sheikh},
  booktitle = {CVPR},
  title = {Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields},
  year = {2017}
}

@inproceedings{simon2017hand,
  author = {Tomas Simon and Hanbyul Joo and Iain Matthews and Yaser Sheikh},
  booktitle = {CVPR},
  title = {Hand Keypoint Detection in Single Images using Multiview Bootstrapping},
  year = {2017}
}

@inproceedings{wei2016cpm,
  author = {Shih-En Wei and Varun Ramakrishna and Takeo Kanade and Yaser Sheikh},
  booktitle = {CVPR},
  title = {Convolutional pose machines},
  year = {2016}
}
```
