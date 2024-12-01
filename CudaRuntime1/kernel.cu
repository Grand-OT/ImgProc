#include "cuda_runtime.h"
#include "cuda_texture_types.h"
#include "device_launch_parameters.h"
#include "ImageFloatConverter.h"
#include "KernelGenerator.h"
#include "ReadWriteImg.h"
#include <algorithm>
#include <cuda_runtime_api.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <stdio.h>
#include <texture_fetch_functions.h>
#include <texture_types.h>
#include <thread>

#define INITIALIZE_ENTITIES(i)  auto& img = images[i]; \
                                auto& resImg = resImages[i]; \
                                auto& imgD = imgDs[i]; \
                                auto& resD = resDs[i]; \
                                auto& kernelD = kernelDs[i]; \
                                auto& blockDim = blockDims[i]; \
                                auto& gridDim = gridDims[i]; \
                                auto& memSize = memSizes[i]; \
                                auto& dimParams = dimParamsS[i]; \
                                auto& texObj = texObjs[i];

struct DimParams
{
    dim3 gridDim, blockDim;
    unsigned sharedMemSize = 0;
};

struct Measurements
{
    using MeasRes = std::map<int, double>;
    std::mutex mtx;
    MeasRes meas[4];
};

Measurements results[2];

void writeToFile(const std::string& filename);

template <class KerType, class... KerArgs>
float testKernel(KerType ker, const DimParams& dimParams, KerArgs&&... _Args)
{
    auto gridSize = dimParams.gridDim;
    auto blockSize = dimParams.blockDim;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // запись события
    cudaEventRecord(start, 0);
    if (dimParams.sharedMemSize == 0)
        ker << <gridSize, blockSize >> > (std::forward<KerArgs>(_Args)...);
    else
        ker << <gridSize, blockSize, dimParams.sharedMemSize >> > (std::forward<KerArgs>(_Args)...);
    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return elapsedTime;
}

std::pair<Image<byte>::Images, Image<byte>::Images> splitImages(const Image<byte>& src,
    const Image<byte>& dest, int devCount);

cudaTextureObject_t create3DTextureObject(const Image<byte>& img);


__global__ 
void conv(const Image<byte> source, Image<byte> res, const Kernel kernel)
{
    
    const int resx = blockIdx.x * blockDim.x + threadIdx.x;
    const int resy = blockIdx.y * blockDim.y + threadIdx.y;
    if (resx > res.width || resy > res.height)
        return;
    const int x = resx;
    const int y = resy + source.botPad;
    const int ch = blockIdx.z;
    if (ch != 3)
    {
        float sum = 0.0;
        const int kernelSize = kernel.size;
        for (int cx = -kernelSize / 2; cx < kernelSize / 2 + 1; ++cx) {
            for (int cy = -kernelSize / 2; cy < kernelSize / 2 + 1; ++cy) {
                if (x + cx >= 0 && x + cx < source.width && y + cy >= 0 && y + cy < source.height) {
                    sum += kernel(cx, cy) * source(x + cx, y + cy, ch);
                }
            }
        }
        res(resx, resy, ch) = sum;
    }
    else
    {
        res(resx, resy, ch) = source(x, y, ch);
    }
}

__global__
void convTex(cudaTextureObject_t tex, unsigned firstLine, unsigned imgWidth,
    unsigned imgHeight, unsigned imgCh, Image<byte> res, const Kernel kernel)
{
    const int resx = blockIdx.x * blockDim.x + threadIdx.x;
    const int resy = blockIdx.y * blockDim.y + threadIdx.y;
    if (resx > res.width || resy > res.height)
        return;
    const int x = resx;
    const int y = resy + firstLine;
    const int ch = blockIdx.z;
    if (ch != 3)
    {
        float sum = 0.0;

        const int kernelSize = kernel.size;
        for (int cx = -kernelSize / 2; cx < kernelSize / 2 + 1; ++cx) {
            for (int cy = -kernelSize / 2; cy < kernelSize / 2 + 1; ++cy) {
                /*if (x + cx >= 0 && x + cx < imgWidth && y + cy >= 0 && y + cy < imgHeight)*/ {
                    byte imgVal = tex3D<byte>(tex, x + cx, y + cy, ch);
                    sum += kernel(cx, cy) * imgVal;
                }
            }
        }
        res(resx, resy, ch) = sum;
    }
    else
    {
        res(resx, resy, ch) = tex3D<byte>(tex, x, y, ch);
    }
}

__global__
void convShKernel(const Image<byte> source, Image<byte> res, const Kernel kernel)
{
    __shared__ float kernelSh[32 * 32];
    const int resx = blockIdx.x * blockDim.x + threadIdx.x;
    const int resy = blockIdx.y * blockDim.y + threadIdx.y;
    if (resx > res.width || resy > res.height)
        return;
    const int x = resx;
    const int y = resy + source.botPad;
    const int ch = blockIdx.z;

    if (threadIdx.x < kernel.size && threadIdx.y < kernel.size && ch == 0)
    {
        const int idx = threadIdx.y * kernel.size + threadIdx.x;
        kernelSh[idx] = kernel.data[idx];
    }
    __syncthreads();
    if (ch != 3)
    {
        float sum = 0.0;

        const int kernelSize = kernel.size;
        for (int cx = -kernelSize / 2; cx < kernelSize / 2 + 1; ++cx) {
            for (int cy = -kernelSize / 2; cy < kernelSize / 2 + 1; ++cy) {
                if (x + cx >= 0 && x + cx < source.width && y + cy >= 0 && y + cy < source.height) {
                    const int kernelIdx = (cy + kernel.size / 2) * kernel.size + cx + kernel.size / 2;
                    sum += kernelSh[kernelIdx] * source(x + cx, y + cy, ch);
                }
            }
        }
        res(resx, resy, ch) = sum;
    }
    else
    {
        res(resx, resy, ch) = source(x, y, ch);
    }
}

__global__ 
void convSh(const Image<byte> source, Image<byte> res, const Kernel kernel)
{
    const int kernelSize = kernel.size;
    const int kernelRadius = (kernelSize - 1) / 2;
    const int x0 = blockIdx.x * blockDim.x;
    const int y0 = blockIdx.y * blockDim.y;
    const int resx = blockIdx.x * blockDim.x + threadIdx.x;
    const int resy = blockIdx.y * blockDim.y + threadIdx.y;
    if (resx > res.width || resy > res.height)
        return;
    const int ch = blockIdx.z;
    const int tileWidth = blockDim.x + kernelSize - 1;
    const int tileHeight = blockDim.y + kernelSize - 1;
    const int tileSize = tileWidth * tileHeight;
    const int imgWidth = source.width;
    const int imgHeight = source.height;

    extern __shared__ byte blockTile[];
    const int blockSize = blockDim.x * blockDim.y;
    const int count = tileSize / blockSize;
    const int threadNum = threadIdx.y * blockDim.x + threadIdx.x;
    int curTileIdx;
    for (int i = 0; i < count; ++i)
    {
        curTileIdx = threadNum * count + i;
        const int imgX = x0 - kernelRadius + curTileIdx % tileWidth;
        const int imgY = y0 - kernelRadius + curTileIdx / tileWidth + source.botPad;
        if (imgX < 0 || imgX >= source.width || imgY < 0 || imgY >= source.height)
            blockTile[curTileIdx] = 0;
        else
            blockTile[curTileIdx] = source(imgX, imgY, ch);
    }
    curTileIdx = count * blockSize + threadNum;
    if (tileSize > curTileIdx)
    {
        const int imgX = x0 - kernelRadius + curTileIdx % tileWidth;
        const int imgY = y0 - kernelRadius + curTileIdx / tileWidth;
        if (imgX < 0 || imgX >= imgWidth || imgY < 0 || imgY >= imgHeight)
            blockTile[count * blockSize + threadNum] = 0;
        else
            blockTile[count * blockSize + threadNum] = source(imgX, imgY, ch);
    }
    const int invId = blockSize - threadNum - 1;
    float* kernelSh = (float*)(blockTile + tileSize * sizeof(byte));
    if (invId < kernelSize * kernelSize)
    {
        kernelSh[invId] = kernel.data[invId];
    }
    __syncthreads();
    if (ch != 3)
    {
        float sum = 0.0;

        for (int cx = -kernelRadius; cx < kernelRadius + 1; ++cx) {
            for (int cy = -kernelRadius; cy < kernelRadius + 1; ++cy) {
                {
                    int tileIdx = (kernelRadius + cy + threadIdx.y) * tileWidth +
                        cx + threadIdx.x + kernelRadius;
                    int kernelIdx = (cy + kernelRadius) * kernelSize + cx + kernelRadius;
                    sum += kernelSh[kernelIdx] * blockTile[tileIdx];
                }
            }
        }
        res(resx, resy, ch) = sum;
    }
    else
    {
        int tileIdx = (kernelRadius + threadIdx.y) * tileWidth +
            threadIdx.x + kernelRadius;
        res(resx, resy, ch) = blockTile[tileIdx];
    }
}

int deviceCount;

void handleConvolutionSize(int size);



int main()
{
    cudaGetDeviceCount(&deviceCount);
    deviceCount = 1;
    handleConvolutionSize(3);
    handleConvolutionSize(5);
    handleConvolutionSize(7);
    handleConvolutionSize(11);
    handleConvolutionSize(13);
    handleConvolutionSize(15);
    handleConvolutionSize(17);
    handleConvolutionSize(19);
    handleConvolutionSize(21);
    handleConvolutionSize(23);
    handleConvolutionSize(25);
    handleConvolutionSize(27);
    handleConvolutionSize(29);
    handleConvolutionSize(31);
    writeToFile("..\\results.txt");
}

std::pair<Image<byte>::Images, Image<byte>::Images> splitImages(const Image<byte>& src, const Image<byte>& dest, int devCount)
{
    return std::pair<Image<byte>::Images, Image<byte>::Images>();
}

cudaTextureObject_t create3DTextureObject(const Image<byte>& img) {

    // Размеры 3D текстуры (ширина, высота, число каналов)
    const int width = img.width,
        height = img.height,
        channels = img.channels;
    cudaExtent extent = make_cudaExtent(width, height, channels);

    // Описание формата канала
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<byte>();

    // Создание 3D массива
    cudaArray_t cuArray3D;
    cudaMalloc3DArray(&cuArray3D, &channelDesc, extent);

    // Преобразование данных в формат, совместимый с 3D массивом
    std::vector<byte> linearData(img.getSize());
    for (int z = 0; z < channels; ++z) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int idx = (y * width + x) * channels + z;
                int linearIdx = z * (width * height) + y * width + x;
                linearData[linearIdx] = img.data[idx];
            }
        }
    }

    // Копирование данных в 3D массив
    cudaMemcpy3DParms copyParams = {};
    copyParams.srcPtr = make_cudaPitchedPtr(
        linearData.data(), width * sizeof(unsigned char), width, height);
    copyParams.dstArray = cuArray3D;
    copyParams.extent = extent;
    copyParams.kind = cudaMemcpyHostToDevice;
    cudaMemcpy3D(&copyParams);

    // Настройка ресурса текстуры
    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray3D;

    // Настройка параметров текстуры
    cudaTextureDesc texDesc = {};
    texDesc.addressMode[0] = cudaAddressModeClamp; // Адресация по ширине
    texDesc.addressMode[1] = cudaAddressModeClamp; // Адресация по высоте
    texDesc.addressMode[2] = cudaAddressModeClamp; // Адресация по глубине (каналы)
    texDesc.filterMode = cudaFilterModePoint;     // Без интерполяции
    texDesc.readMode = cudaReadModeElementType;    // Чтение байтов без нормализации
    texDesc.normalizedCoords = false;             // Не нормализовать координаты

    // Создание текстурного объекта
    cudaTextureObject_t texObject = 0;
    cudaCreateTextureObject(&texObject, &resDesc, &texDesc, nullptr);

    return texObject;
}

void handleConvolutionSize(int size)
{
    
    ReadWriteImg reader;
    const std::string base = "..\\";  // Получаем базовую директорию
    const std::string img_input_name = "test.png";
    const std::string img_output_name = "res.png";

    std::string full_path_to_image = base + img_input_name;
    std::string full_path_to_result = base + img_output_name;

    // Используем полный путь к изображению
    auto _img = reader.readImage(full_path_to_image);

    auto _resImg = _img.createSimilar();
    Kernel ker = KernelGenerator().generateBlurKernel(size);
    //Kernel ker = KernelGenerator().generateEdgeKernel();
    auto images = _img.splitWithOverlap(deviceCount, ker.size / 2);
    auto resImages = _resImg.splitImage(deviceCount);

    std::vector<byte*> imgDs, resDs;
    std::vector<float*> kernelDs;
    std::vector<dim3> blockDims, gridDims;
    std::vector<unsigned> memSizes;
    std::vector<DimParams> dimParamsS;
    std::vector<cudaTextureObject_t> texObjs;

    for (int i = 0; i < deviceCount; ++i)
    {
        imgDs.emplace_back();
        resDs.emplace_back();
        kernelDs.emplace_back();
        blockDims.emplace_back();
        gridDims.emplace_back();
        memSizes.emplace_back();
        dimParamsS.emplace_back();
        texObjs.emplace_back();
        INITIALIZE_ENTITIES(i);
        cudaSetDevice(i);


        cudaMalloc(&imgD, img.getSize());
        cudaMemcpy(imgD, img.data, img.getSize(), cudaMemcpyHostToDevice);

        cudaMalloc(&resD, img.getSize());

        unsigned size2 = ker.size * ker.size * sizeof(float);
        cudaMalloc(&kernelD, size2);
        auto res = cudaMemcpy(kernelD, ker.data, size2, cudaMemcpyHostToDevice);

        unsigned numOfThreadsPerDim = 32;

        texObj = create3DTextureObject(img);

        blockDim.x = std::min(numOfThreadsPerDim, img.width);
        blockDim.y = std::min(numOfThreadsPerDim, img.height);
        gridDim.x = (img.width + blockDim.x - 1) / blockDim.x;
        gridDim.y = (img.height + blockDim.y - 1) / blockDim.y;
        gridDim.z = img.channels;

        memSize = (numOfThreadsPerDim + ker.size - 1) *
            (numOfThreadsPerDim + ker.size - 1) * sizeof(byte) +
            size2;

        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, i);
        if (memSize > deviceProp.sharedMemPerBlock)
        {
            printf("Not enough shared memory\n");
            return;
        }

        dimParams.blockDim = blockDim;
        dimParams.gridDim = gridDim;
    }
    std::vector<std::thread> workers;
    std::vector<std::mutex> mutexes(deviceCount);
    for (int i = 0; i < deviceCount; ++i)
    {
        workers.emplace_back([&, i]() {
            auto& img = images[i];
            auto& resImg = resImages[i];
            auto& imgD = imgDs[i];
            auto& resD = resDs[i];
            auto& kernelD = kernelDs[i];
            auto& blockDim = blockDims[i];
            auto& gridDim = gridDims[i];
            auto& memSize = memSizes[i];
            auto& dimParams = dimParamsS[i];
            auto& texObj = texObjs[i];
            cudaSetDevice(i);
            std::unique_lock<std::mutex> lk(mutexes[0]);
            //test common memory convolution
            Image<byte> sourceD = img;
            sourceD.data = imgD;
            auto resImgD = resImg;
            resImgD.data = resD;
            auto kerD = ker;
            kerD.data = kernelD;
            float elapsedTimeCommonMemory1 = testKernel(conv, dimParams, sourceD, resImgD, kerD);
            printf("Elapsed time for common memory %d: %f\n", i, elapsedTimeCommonMemory1);

            //test texture memory convolution
            float elapsedTimeCommonMemory2 = testKernel(convTex, dimParams, texObj, img.botPad, img.width,
                img.height, img.channels, resImgD, kerD);
            printf("Elapsed time for texture memory %d: %f\n", i, elapsedTimeCommonMemory2);

            ////test shared memory convolution
            float elapsedTimeCommonMemory3 = testKernel(convShKernel, dimParams, sourceD, resImgD, kerD);
            printf("Elapsed time for shared kernel %d: %f\n", i, elapsedTimeCommonMemory3);

            ////test shared memory convolution
            dimParams.sharedMemSize = memSize;
            float elapsedTimeCommonMemory4 = testKernel(convSh, dimParams, sourceD, resImgD, kerD);
            printf("Elapsed time for shared memory %d: %f\n", i, elapsedTimeCommonMemory4);



            {
                std::unique_lock<std::mutex> lk(results[i].mtx);
                results[i].meas[0][kerD.size] = elapsedTimeCommonMemory1;
                results[i].meas[1][kerD.size] = elapsedTimeCommonMemory2;
                results[i].meas[2][kerD.size] = elapsedTimeCommonMemory3;
                results[i].meas[3][kerD.size] = elapsedTimeCommonMemory4;
            }
            cudaMemcpy(resImg.data, resD, resImg.getSize(), cudaMemcpyDeviceToHost);

            cudaFree(imgD);
            cudaFree(resD);
            cudaFree(kernelD);
            cudaDestroyTextureObject(texObj);
            });
    }

    for (int i = 0; i < deviceCount; ++i)
        workers[i].join();
    //auto resImg = _img;
    //reader.writeImage(_resImg, (base + img_output_name));
    _resImg.clear();
    _img.clear();
    free(ker.data);
    return;
}

void writeToFile(const std::string& filename) {
    std::ofstream outFile(filename);
    if (!outFile) {
        std::cerr << "Ошибка открытия файла для записи!" << std::endl;
        return;
    }

    for (size_t i = 0; i < sizeof(results) / sizeof(Measurements); ++i) {
        Measurements& currResults = results[i];
        std::lock_guard<std::mutex> lock(currResults.mtx);

        for (size_t j = 0; j < 4; ++j) {
            const Measurements::MeasRes& measMap = currResults.meas[j];
            for (const auto& entry : measMap) {
                outFile << i << "," << j << "," << entry.first << "," << entry.second << std::endl;
            }
        }
    }

    outFile.close();
    std::cout << "Данные успешно записаны в файл '" << filename << "'" << std::endl;
}