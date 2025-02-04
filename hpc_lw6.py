# -*- coding: utf-8 -*-
"""HPC_LW6.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/12_O7PHovQcSWAsbMfYGasEdUAjZTKQZE
"""

from numba import cuda
import matplotlib.pyplot as plt
import numpy as np
import time
import math
from PIL import Image

from google.colab import drive
drive.mount('/content/drive')

im = plt.imread("/content/drive/MyDrive/Colab Notebooks/image2.jpg")

shape = np.shape(im)

"""#LW6a"""

@cuda.jit
def grayscalebinarlization1(src, dst, threshold):
  tidx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
  tidy = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
  
  v = (src[tidx,tidy,0]+src[tidx,tidy,1]+src[tidx,tidy,2])/3

  if v >= threshold:
    v = 255
  else:
    v = 0

  dst[tidx, tidy, 0 ] = np.uint8(v)
  dst[tidx, tidy, 1 ] = np.uint8(v)
  dst[tidx, tidy, 2 ] = np.uint8(v)

devdata = cuda.to_device(im)
devOuput = cuda.device_array(shape, np.uint8)
blockSize = (32,32)

gridSize = (math.ceil(shape[0]/blockSize[0]),math.ceil(shape[1]/blockSize[1]))
grayscalebinarlization1[gridSize, blockSize](devdata, devOuput, 50)

result = devOuput.copy_to_host()
imgpu = Image.fromarray(result)
imgpu.save("/content/drive/MyDrive/Colab Notebooks/image_bw_GPU.jpeg")

@cuda.jit
def grayscalebinarlization2(src, dst, threshold):
  tidx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
  tidy = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
     
  v = (src[tidx,tidy,0]+src[tidx,tidy,1]+src[tidx,tidy,2])/3

  relv = v/255
  relvthreshold = threshold/255

  bv = math.ceil(relv-relvthreshold)
  bv = bv*255

  dst[tidx, tidy, 0 ] = bv
  dst[tidx, tidy, 1 ] = bv
  dst[tidx, tidy, 2 ] = bv

devdata = cuda.to_device(im)
devOuput = cuda.device_array(shape, np.uint8)
blockSize = (32,32)

gridSize = (math.ceil(shape[0]/blockSize[0]),math.ceil(shape[1]/blockSize[1]))
grayscalebinarlization2[gridSize, blockSize](devdata, devOuput, 50)

result = devOuput.copy_to_host()
imgpu = Image.fromarray(result)
imgpu.save("/content/drive/MyDrive/Colab Notebooks/image_bw_GPU2.jpeg")



"""#LW6B"""

@cuda.jit
def brightnesscontrol(src, dst, brightnesslevel):
  tidx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
  tidy = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y

  r = src[tidx, tidy, 0] + brightnesslevel
  g = src[tidx, tidy, 1] + brightnesslevel
  b = src[tidx, tidy, 2] + brightnesslevel
  
  if r > 255:
    r = 255
  elif r <0:
    r = 0
  
  if g > 255:
    g = 255
  elif g <0:
    g = 0

  if b > 255:
    b = 255
  elif b <0:
    b = 0

  dst[tidx, tidy, 0] = np.uint8(r)
  dst[tidx, tidy, 1] = np.uint8(g)
  dst[tidx, tidy, 2] = np.uint8(b)

devdata = cuda.to_device(im)
devOuput = cuda.device_array(shape, np.uint8)
blockSize = (32,32)

gridSize = (math.ceil(shape[0]/blockSize[0]),math.ceil(shape[1]/blockSize[1]))
brightnesscontrol[gridSize, blockSize](devdata, devOuput, 80)

result = devOuput.copy_to_host()
imgpu = Image.fromarray(result)
imgpu.save("/content/drive/MyDrive/Colab Notebooks/image_brightness_GPU.jpeg")



"""#LW6c"""

im1 = plt.imread("/content/drive/MyDrive/Colab Notebooks/image7.jpg")
im2 = plt.imread("/content/drive/MyDrive/Colab Notebooks/image8.jpg")

@cuda.jit
def mergepic(src1, src2, dst, threshold):
  tidx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
  tidy = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y

  r = src1[tidx, tidy, 0]*threshold+src2[tidx, tidy, 0]*(1-threshold)
  g = src1[tidx, tidy, 1]*threshold+src2[tidx, tidy, 1]*(1-threshold)
  b = src1[tidx, tidy, 2]*threshold+src2[tidx, tidy, 2]*(1-threshold)

  dst[tidx, tidy, 0] = np.uint8(r)
  dst[tidx, tidy, 1] = np.uint8(g)
  dst[tidx, tidy, 2] = np.uint8(b)

devdata1 = cuda.to_device(im1)
devdata2 = cuda.to_device(im2)
shape = np.shape(im1)
devOuput = cuda.device_array(shape, np.uint8)
blockSize = (32,32)
gridSize = (math.ceil(shape[0]/blockSize[0]),math.ceil(shape[1]/blockSize[1]))

mergepic[gridSize, blockSize](devdata1, devdata2, devOuput, 0.5)

result = devOuput.copy_to_host()
imgpu = Image.fromarray(result)
imgpu.save("/content/drive/MyDrive/Colab Notebooks/merge_GPU.jpeg")