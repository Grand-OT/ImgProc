#pragma once

struct Kernel
{
	unsigned size = 0;
	float* data = nullptr;
};

class KernelGenerator
{
public:
	Kernel generateBlurKernel(unsigned size) const;
	Kernel generateGaussianKernel(unsigned size, float sigma) const;
	Kernel generateEdgeKernel() const;
};

