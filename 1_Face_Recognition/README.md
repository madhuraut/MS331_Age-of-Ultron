# Face Recognition



## About this module

Deep cascaded multi-task framework exploits the inherent correlation between detection and alignment to boost up their performance. 
The framework leverages a cascaded architecture with three stages of carefully designed deep convolutional networks to predict face and landmark location in a coarse-to-fine manner. 
In addition, a new online hard sample mining strategy is used which further improves the performance in practice


### Stage 1
A fully convolutional network, called Proposal Network (P-Net) is built, to obtain the candidate facial windows and their bounding box regression vectors.
Then candidates are calibrated based on the estimated bounding box regression vectors. After that, a non-maximum suppression is used (NMS) to merge highly overlapped candidates.

### Stage 2
All candidates are fed to another CNN, called Refine Network (R-Net), which further rejects a large number of false candidates, performs calibration with bounding box regression, and conducts NMS

### Stage 3
Output Network (O-Net):This stage is similar to the second stage, but in this stage we aim to identify face regions with more supervision. 
In particular, the network will output five facial landmarks’ positions.


## FaceNet
It directly learns a mapping from face images to a compact Euclidean space where distances directly correspond to a measure of face similarity. Once this space has been produced, tasks such as face recognition, verification and clustering can be easily implemented using standard techniques with FaceNet embeddings as feature vectors.

### DEEP CONVOLUTIONAL NETWORK
To directly optimize the embedding itself, rather than an intermediate bottleneck layer as in deep face approach where they use sofmax loss to train the model.

### TRIPLETS
To train, FaceNet uses triplets of roughly aligned matching / non-matching face patches generated using a novel online triplet mining method (hard samples)

With the help of triplet loss facenet achieves better discriminative power against feature vectors. Thus, minimizing intra class variance and maximizing interclass distance.


## How to use

`python face_recognition_real_time.py`



