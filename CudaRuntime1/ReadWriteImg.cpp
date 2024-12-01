#include "ReadWriteImg.h"
#include <vector>
#include "libpng/png.h"


Image<byte> ReadWriteImg::readImage(const std::string& imgPath) const
{
    FILE* file = fopen(imgPath.c_str(), "rb");
    if (!file) {
        return {};
    }

    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png) {
        fclose(file);
        return {};
    }

    png_infop info = png_create_info_struct(png);
    if (!info) {
        png_destroy_read_struct(&png, NULL, NULL);
        fclose(file);
        return {};
    }

    png_init_io(png, file);
    png_read_info(png, info);

    unsigned width = png_get_image_width(png, info);
    unsigned height = png_get_image_height(png, info);
    unsigned bit_depth = png_get_bit_depth(png, info);
    unsigned channels = png_get_channels(png, info);

    png_bytep pixels = (png_bytep)malloc(width * height * channels * sizeof(png_bytep));
    std::vector<png_bytep> row_pointers(height);
    for (unsigned y = 0; y < height; y++) {
        row_pointers[y] = (&pixels[y * width * channels]);
    }

    png_read_image(png, &row_pointers[0]);

    fclose(file);
    png_destroy_read_struct(&png, &info, NULL);

    return { width, height, channels, 0, 0, pixels };
}

bool ReadWriteImg::writeImage(Image<byte> img, const std::string& imgPath) const
{
    if (!img.data) {
        return false;
    }

    FILE* fp = fopen(imgPath.c_str(), "wb");
    if (!fp) {
        return false;
    }

    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png) {
        fclose(fp);
        return false;
    }

    png_infop info = png_create_info_struct(png);
    if (!info) {
        png_destroy_write_struct(&png, NULL);
        fclose(fp);
        return false;
    }

    if (setjmp(png_jmpbuf(png))) {
        png_destroy_write_struct(&png, &info);
        fclose(fp);
        return false;
    }

    const unsigned width = img.width;
    const unsigned height = img.height;
    const unsigned channels = img.channels;
    png_bytep image_data = img.data;
    png_init_io(png, fp);
    int color_type = 0; PNG_COLOR_TYPE_RGB;
    switch (channels)
    {
    case 3:
    {
        color_type = PNG_COLOR_TYPE_RGB;
        break;
    }
    case 4:
    {
        color_type = PNG_COLOR_TYPE_RGB_ALPHA;
        break;
    }
    case 1:
    {
        color_type = PNG_COLOR_TYPE_GRAY;
        break;
    }
    default:
        break;
    }
    png_set_IHDR(png, info, width, height, 8, color_type, PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_write_info(png, info);

    // Создаем буфер для записи
    png_bytep row = (png_bytep)malloc(channels * width * sizeof(png_byte));
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = (y * width + x) * channels;
            for (int c = 0; c < channels; ++c) {
                row[x * channels + c] = image_data[idx + c];
            }
        }
        png_write_row(png, row);
    }

    free(row);
    png_write_end(png, NULL);
    png_destroy_write_struct(&png, &info);
    fclose(fp);
    return true;
}