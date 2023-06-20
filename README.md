# video-action-recognition

## Pre-trained weights

-[Pretrained weights for tubelet size 1](https://drive.google.com/file/d/1L7Afvmmku_sHCjRZurGWSxiZDlXl_cuF/view?usp=sharing)

-[Pretrained weights for tubelet size of 2](https://drive.google.com/file/d/1q5R2aJ2ncW8hEEl1dYRiRbAhGlTekvPD/view?usp=sharing)


This code is an implementation of an activity recognition model that uses PyTorch for training and testing. The model takes in a video clip and predicts the activity being performed in the clip. This implementation uses the ActivityNet and UCF101 dataset, which has 21 activity classes.

## Installation
To use this code, you will need to have PyTorch and argparse installed. You can install PyTorch by following the instructions on the PyTorch website. argparse should be included with your Python installation.

## Usage
You can run the code by running the parse_opt() function. This function takes in several arguments that can be specified to customize the training and testing process. These arguments are:

- batch_size: the batch size for training (default: 32)
- test_batch_size: the batch size for testing (default: 1)
- frames_per_clip: the number of frames per clip (default: 32)
- lr: the learning rate for the optimizer (default: 0.001)
- num_classes: the number of activity classes (default: 21)
- annot_dir: the directory for the ActivityNet annotation file (default: "../anet_annotations/activitynet-21-category.json")
- patch_size: the size of each patch in the CNN (default: 16)
- epochs: the number of epochs to train for (default: 1)
- tt_split: the train-test split ratio (default: 0.8)
- image_size: the size of the input images (default: 224)
- dataset_dir: the directory for the dataset (default: "../../../scratch/work/pehlivs1/others_files/activitynet_frames_zipped")
- pr: the number of processors to use for data loading (default: 1)
- dataset: the dataset to use (default: "Anet")
- checkpoint_flag: flag for saving model checkpoints (default: 0)
- str: flag for printing training loss (default: 1)
- dropout: the dropout rate for the fully connected layers (default: 0.0)
- You can specify these arguments when running the code to customize the training and testing process. For example, you can run the code with the following command:


 python activity_recognition.py --batch_size 64 --lr 0.0001 --epochs 10
 This will train the model with a batch size of 64, a learning rate of 0.0001, and for 10 epochs.

## Data
The dataset used in this code is the ActivityNet and UCF101, which contains videos of various activities. You can download the dataset from the ActivityNet and UCF101 website.








