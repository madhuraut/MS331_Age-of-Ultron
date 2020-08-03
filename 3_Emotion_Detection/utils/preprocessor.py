import numpy as np
from imageio import imread
from skimage.transform import resize
##################
def input_preproc(s, v=True):
    s = s.astype('float32')
    s = s / 255.0
    if v:
        s = s - 0.5
        s = s * 2.0
    return s


