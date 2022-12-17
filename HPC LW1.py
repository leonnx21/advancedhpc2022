# -*- coding: utf-8 -*-
	
from numba import cuda
print(cuda.gpus)

cuda.is_available()

cuda.detect()

cuda.gpus

from numba.cuda.cudadrv import enums
device = cuda.get_current_device()
attribs= [name.replace("CU_DEVICE_ATTRIBUTE_", "") for name in dir(enums) if name.startswith("CU_DEVICE_ATTRIBUTE_")]
attribs

print(cuda.current_context().get_memory_info())

print(getattr(device,'CLOCK_RATE'))

print(getattr(device,'MULTIPROCESSOR_COUNT'))

print(getattr(device,'WARP_SIZE'))

