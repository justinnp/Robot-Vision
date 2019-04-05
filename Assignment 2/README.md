# Lucas Kanade

This project folder contains the code for a Lucas Kanade Optical Flow script determining the optical flow for two images as well as implementing a multi-resolution optical flow using Gaussian Pyramids.

The two inputs are `basketball1.png` and `basketball2.png`

The output folders are `Lucas Kanade Outputs` and `Lucas Kanade Pyramid Outputs`

The output images will be placed in their respective folders based on whether it was implemented with single resolution Lucas Kanade or Gaussian Pyramid Lucas Kanade for multi resolution.

To run the script:

`python3 lucas_kanade.py`

To exit the image screen:

Type `0` on the keyboard


# Convolutional Neural Networks

This project folder also contains the code for the CNN models as well as the script to test and train them.

Outputs are stored in `CNN Outputs`, which contains screenshots and stack traces from the terminal commands. 

Note:

The screenshot results and the output.txt results may display varying accuracy levels as they were the result of multiple test trainings.

To run:

`python3 main.py` 

and then add the desired arguments listed below

`--mode` mode to define which model to be used: int

`--batch-size` input batch size for training: int

`--hidden-size` hidden layer size for network: int

`weight-decay` Weight decay, used for L2 regularization: float

`--test-batch-size` input batch size for testing: int

`--epochs` number of epochs to train: int

`--lr` learning rate: float

`--momentum` SGD momentum: float

`--no-cuda` disables CUDA training: boolean

`--seed` random seed: int 

`--log-interval` how many batches to wait before logging training status: int

`--save-model` For Saving the current Model: boolean

