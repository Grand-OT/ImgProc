#define _USE_MATH_DEFINES
#include "KernelGenerator.h"
#include <stdlib.h>
#include <math.h>
#include <cstring>

Kernel KernelGenerator::generateBlurKernel(unsigned size) const
{
    if (size % 2 == 0) {
        return {};
    }
    unsigned size2 = size * size;
    float* data = (float*)malloc(size2 * sizeof(float));
    for (unsigned i = 0; i < size2; ++i) {
        data[i] = 1.0 / size2;
    }

    return { size, data };
}

Kernel KernelGenerator::generateGaussianKernel(unsigned size, float sigma) const
{
    if (size % 2 == 0) {
        return {};
    }
    unsigned size2 = size * size;
    float* data = (float*)malloc(size2 * sizeof(float));
    float sum = 0;
    for (int mx = 0; mx < size; ++mx)
        for (int my = 0; my < size; ++my)
        {
            int x = mx - size / 2;
            int y = my - size / 2;
            sum += data[my * size + mx] = expf(-1. * (x * x + y * y) / (2 * sigma * sigma));
        }

    for (int mx = 0; mx < size; ++mx)
        for (int my = 0; my < size; ++my)
            data[my * size + mx] /= sum;
    return { size, data };
}

Kernel KernelGenerator::generateEdgeKernel() const
{
    /*
    * -1 -1 -1
    * -1  8 -1
    * -1 -1 -1
    */

    float* data = (float*)malloc(9 * sizeof(float));
    for (int i = 0; i < 9; ++i)
        data[i] = -1;
    data[4] = 8; //center element 

    return { 3, data };
}
