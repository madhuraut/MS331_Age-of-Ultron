# Emotion Detection


## About this module

This architecture combines the use of residual modules and depth-wise separable convolutions.
This model relies on the idea of eliminating completely the fully connected layers from CNN and the inclusion of the combined depth-wise separable convolutions and residual modules. 
The model is trained on FER2013 dataset.
This architecture was trained with ADAM optimizer.
Adam is an optimization algorithm that can be used instead of the classical stochastic gradient descent procedure to update network weights iterative based in training data


### Depth wise Convolutions
It separates the spatial cross-correlations from the channel cross correlations.
It is the channel-wise n×n spatial convolution.

### Point wise Convolutions
It separates the spatial cross-correlations from the channel cross correlations.
It is the 1×1 convolution to change the dimension.




## How to use

`python emotion_detection.py`

