# -*- coding: utf-8 -*-
"""HPC_LW3.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1_9BLffHioN7LMJo--9A8HfSLJE0Dr0E2
"""

from numba import cuda
import matplotlib.pyplot as plt
import numpy as np
import time
import math
from PIL import Image
import pandas as pd

from google.colab import drive
drive.mount('/content/drive')

im = plt.imread("/content/drive/MyDrive/Colab Notebooks/image1.jpg")

shape = np.shape(im)
shape

devdata = cuda.to_device(im)
devOuput = cuda.device_array(shape, np.uint8)

@cuda.jit
def grayscale(src, dst):
  tidx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
  tidy = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y

  g = np.uint8((src[tidx, tidy, 0] + src[tidx, tidy, 1] + src[tidx, tidy, 2]) / 3)
  dst[tidx, tidy, 0] = dst[tidx, tidy, 1] = dst[tidx, tidy, 2] = g

def testfunc(x,y):
  t1 = time.time()
  blockSize = (x, y)
  gridSize = (math.ceil(shape[0]/blockSize[0]),math.ceil(shape[1]/blockSize[1]))
  grayscale[gridSize, blockSize](devdata, devOuput)
  t2 = time.time()
  t = t2 - t1

  return t

resultname = []
listi = [1,2,4,8,16,24,32,64,128,512,1024]
listj = [1,2,4,8,16,24,32,64,128,512,1024]
listi2 = []
listj2 = []

for i in listi:
  for j in listj:
      if (i*j)<=1024:
        resultname.append((i,j))

resultname

result = []

for i in resultname:
    t3 = testfunc(i[0], i[1])
    result.append(t3)

l = list(range(len(result)))

plt.bar(l,result)
plt.yscale("log")