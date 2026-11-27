# ============================================================================
# Glassbox-100M Pure From-Scratch C++17 & CUDA C++17 Build System
# Zero PyTorch. Zero TensorFlow. Zero cuDNN.
# ============================================================================

CXX      ?= g++
NVCC     ?= nvcc
ARCH     ?= sm_89
USE_CUBLAS ?= 1

CXXFLAGS := -O3 -march=native -fno-math-errno -fno-trapping-math -std=c++17 -Wall -Wextra
NVCCFLAGS := -O3 -std=c++17 -arch=$(ARCH) --use_fast_math

ifeq ($(USE_CUBLAS),1)
NVCCFLAGS += -DGBX_USE_CUBLAS
CUDA_LIBS := -lcublas
else
CUDA_LIBS :=
endif

.PHONY: all cpu cuda clean

all: cpu

# Native Pure C++17 CPU Trainer (runs on any x86_64/ARM64 CPU)
cpu: glassbox_cpu_train

glassbox_cpu_train: train_glassbox_cpu.cpp glassbox_format.hpp ../glassbox/csrc/kernels.cpp
	$(CXX) $(CXXFLAGS) train_glassbox_cpu.cpp ../glassbox/csrc/kernels.cpp -o $@

# Pure From-Scratch CUDA C++17 GPU Trainer (RTX 4090=sm_89, A100=sm_80, H100=sm_90)
cuda: glassbox_cuda_train

glassbox_cuda_train: train_glassbox_100m.cu kernels.cu glassbox_cuda.cuh glassbox_format.hpp
	$(NVCC) $(NVCCFLAGS) train_glassbox_100m.cu kernels.cu $(CUDA_LIBS) -o $@

# 100% pure custom CUDA GEMM (zero cuBLAS, zero external libs beyond cudart)
cuda-pure:
	$(MAKE) cuda USE_CUBLAS=0 ARCH=$(ARCH)

clean:
	rm -f glassbox_cpu_train glassbox_cuda_train
