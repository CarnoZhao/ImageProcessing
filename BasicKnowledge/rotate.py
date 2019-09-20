import numpy as np
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
from math import cos, sin

dog = plt.imread('/home/tongxueqing/zhaox/PythonSkills/ImageProcessing/dog.png')

rotate = lambda theta: np.ndarray(
    [
        [ cos(theta), sin(theta), 0],
        [-sin(theta), cos(theta), 0],
        [          0,          0, 1]
    ])

