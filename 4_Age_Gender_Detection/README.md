### Age/Gender Detection


### About this module

It is a compact soft stagewise regression network for Age and gender estimation with compact model size. 
SSR-Net performs multiclass Classification, then turns result into regression by calculating expected values.
SSR-Net assigns a dynamic range to each age class according to the input images.
We are using Age estimation model Only. 
“SSR-Net” expects input to be a tensor of size: `N x 64 x 64 x 3`, where N is the number of faces, 64x64 is the height and width correspondingly and 3 stands for RGB. 
Whereas Levi and Tal Hassner expect input to be a tensor of size: `N x 3 x 227 x 227`, where N is the number of faces, 3 means channels of RGB and 227x227 is for height and width correspondingly.

### Binary Classification Task
Gender detection module will identify whether person present in camera range is male or female in real time.
The gender prediction is a binary classification task. 
The model outputs value between 0~1, where the higher the value, the more confidence the model think the face is a male.

### 3 Convolutional Layers
96 filters of size 3×7×7 pixels are applied to the input in the first convolutional layer, followed by a Rectified Linear Operator (ReLU), a max pooling layer taking the maximal value of 3 × 3 regions with two-pixel strides and a local response normalization layer. 
The 96 × 28 × 28 output of the previous layer is then processed by the second convolutional layer, containing 256 filters of size 96 × 5 × 5 pixels. Again, this is followed by ReLU, a max pooling layer and a local response normalization layer with the same hyper parameters as before. 
Finally, the third and last convolutional layer operates on the 256 × 14 × 14 blob by applying a set of 384 filters of size 256 × 3 × 3 pixels, followed by ReLU and a max pooling layer.

Finally, the output of the last fully connected layer is fed to a soft-max layer that assigns a probability for the given test image.


### How to use

`python age_detection.py`
`python gender_detection.py`


