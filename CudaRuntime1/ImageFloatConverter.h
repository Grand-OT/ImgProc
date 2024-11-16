#pragma once
#include "Image.h"

class ImageFloatConverter
{
    template<typename S, typename D>
    Image<D> convert(Image<S> img);
public:
    Image<float> byteToFloat(Image<byte> img);
    Image<byte> floatToByte(Image<float> img);
};

template<typename S, typename D>
inline Image<D> ImageFloatConverter::convert(Image<S> img)
{
    Image<D> res;
    res.height = img.height;
    res.width = img.width;
    res.channels = img.channels;
    if (!img.data) {
        return res;
    }
    res.data = (D*)malloc(res.width * res.height * res.channels * sizeof(D));

    for (unsigned x = 0; x < img.width; ++x) {
        for (unsigned y = 0; y < img.height; ++y) {
            unsigned idx = (y * img.width + x) * img.channels;
            for (unsigned c = 0; c < img.channels; ++c) {
                res.data[idx + c] = img.data[idx + c];
            }
        }
    }

    return res;
}
