#pragma once
#include "cuda_runtime.h"

struct Kernel
{
	unsigned size = 0;
	float* data = nullptr;
	__host__ __device__ inline float operator()(const int x, const int y) const
	{
		const int x_ = x + size / 2;
		const int y_ = y + size / 2;
		return data[y_ * size + x_];
	}
};

class KernelGenerator
{
public:
	Kernel generateBlurKernel(unsigned size) const;
	Kernel generateGaussianKernel(unsigned size, float sigma) const;
	Kernel generateEdgeKernel() const;
};

