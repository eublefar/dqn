import numpy as np

def to_grayscale(img):
    return np.mean(img, axis=2)

def downsample(img):
    return img[::2, ::2]

def preprocess(img):
    # return np.expand_dims(to_grayscale(downsample(img)), axis = 2 )
    return to_grayscale(downsample(img))

# command + python3 main.py --batch-size 30 --env-name BreakoutDeterministic-v4
