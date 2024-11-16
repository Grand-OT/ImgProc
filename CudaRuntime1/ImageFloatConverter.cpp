#include "ImageFloatConverter.h"
#include <stdlib.h>


Image<float> ImageFloatConverter::byteToFloat(Image<byte> img)
{
    return convert<byte, float>(img);
}

Image<byte> ImageFloatConverter::floatToByte(Image<float> img)
{
    return convert<float, byte>(img);

}
