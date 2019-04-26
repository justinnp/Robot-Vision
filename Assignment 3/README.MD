# PA3 Action Recognition

Justin Powell

Robot Vision - CAP4453 - Spring 2019


## To run

Enter `python3 pa3_train.py` if you wish to train the model.

However, there are pre-trained models located in the `models` folder than can be used for testing.

Enter `python3 pa3_test.py` if you wish to test the model. This script will display the accuracy of the entire testing dataset as well as the accuracy of a random sample of images from the dataset.

### Note:

Make sure the data being used for testing or training is a .jpg image file and is in the following structure:

`root/action/image.jpg`

i.e. if you are training the model for the diving action, the data should be structured as:

`train/diving/image_name.jpg`

i.e. if you are testing the model for the skateboarding action, the data should be structured as:

`test/skateboarding/image_name.jpg`


## Project Folders and Files

`train` is the training dataset folder. Not included in zip.

`test` is the testing dataset folder. Not included in zip.

`CNN outputs` is the folder containing the output text files that were generated when training and testing the model.

`models` is the folder containing the pre-trained models, one with a .pt extension, another with a .pth extension.

`Screenshots` is the folder containing screenshots of pre-computed accuracies from the testing dataset as well as proof that the training script runs showing epoch and loss.

`pa3_train.py` is the training script.

`pa3_test.py` is the testing script.

`PA3_report.pdf` is the report outlining how the data is loaded and set, the classifier design, the feature extraction method and evaluation of the model.

`PA3.pdf` is the assignment guidelines and requirements.



