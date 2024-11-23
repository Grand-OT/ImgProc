#pragma once
#include "cuda_runtime.h"
#include <algorithm>
#include <stdlib.h>
#include <vector>

using byte = unsigned char;

#define GET_IMAGE_INTENSITY(img, x, y, ch)  img.data[(y * img.width + x) * img.channels + ch]

template <typename T>
struct Image
{
	using Images = std::vector<Image<T>>;
	unsigned width = 0, height = 0, channels = 0;
	unsigned topPad = 0, botPad = 0;
	T* data = nullptr;
	void clear();
	Image<T> clone() const;
	Image<T> createSimilar() const;
	Images splitImage(int nParts) const;
	Images splitWithOverlap(int nParts, int overlap);
	void constructFromParts(const Images& images);
	__host__ __device__ inline const T operator()(const unsigned x, const unsigned y, const unsigned ch) const;
	__host__ __device__ inline T& operator()(const unsigned x, const unsigned y, const unsigned ch);
	unsigned getSize() const;
};

template<typename T>
inline void Image<T>::clear()
{
	if (data)
		free(data);
	width = 0;
	height = 0;
	channels = 0;
}

template<typename T>
inline Image<T> Image<T>::clone() const
{
	auto newImg = createSimilar();
	memcpy(newImg.data, data, getSize());
	return newImg;
}

template<typename T>
inline Image<T> Image<T>::createSimilar() const
{
	T* ptr = nullptr;
	if (data) {
		ptr = (T*)malloc(getSize());
	}

	return{ width, height, channels, botPad, topPad, ptr };
}

template<typename T>
inline typename Image<T>::Images Image<T>::splitImage(int nParts) const
{
	if (nParts <= 0)
		return {};
	int partHeight = height / nParts;
	int rem = height % nParts;
	Image<T>::Images res;
	T* curDataPtr = data;
	for (int i = 0; i < nParts; ++i)
	{
		Image<T> newImage;
		newImage.data = curDataPtr;
		newImage.width = width;
		newImage.height = partHeight + (i < rem);
		newImage.channels = channels;
		curDataPtr += newImage.height * newImage.width * newImage.channels;
		res.push_back(newImage);
	}
	return res;
}

template<typename T>
inline typename Image<T>::Images Image<T>::splitWithOverlap(int nParts, int overlap)
{
	if (nParts <= 0)
		return {};
	int partHeight = height / nParts;
	int rem = height % nParts;
	Image<T>::Images res;
	int curHeight = 0;
	for (int i = 0; i < nParts; ++i)
	{
		Image<T> newImage;
		int blockStartHeight = max(curHeight - overlap, 0);
		newImage.botPad = curHeight - blockStartHeight;
		curHeight += partHeight + (i < rem);
		int blockEndHeight = min(curHeight + overlap, height);
		newImage.topPad = blockEndHeight - curHeight;
		newImage.width = width;
		newImage.height = blockEndHeight - blockStartHeight;
		newImage.channels = channels;
		newImage.data = data + blockStartHeight * width * channels;
		res.push_back(newImage);
	}
	return res;
}

template<typename T>
inline void Image<T>::constructFromParts(const Images& images)
{
	clear();
	if (images.size() == 0)
		return;
	int totalHeight = 0;
	int curWidth = images[0].width;
	int curChannels = images[0].channels;
	for (const auto& image : images)
	{
		totalHeight += image.height;
		if (curWidth != image.width)
			return;
		if (curChannels != image.channels)
			return;
	}
	data = (T*)malloc(totalHeight * curWidth * curChannels * sizeof(T));
	width = curWidth;
	height = totalHeight;
	channels = curChannels;
	T* curData = data;
	for (const auto& image : images)
	{
		const unsigned size = image.getSize();
		memcpy(curData, image.data, size);
		curData += size / sizeof(T);
	}
}

template<typename T>
inline const T Image<T>::operator()(const unsigned x, const unsigned y, const unsigned ch) const
{
	return data[(y * width + x) * channels + ch];
}

template<typename T>
inline T& Image<T>::operator()(const unsigned x, const unsigned y, const unsigned ch)
{
	return data[(y * width + x) * channels + ch];
}

template<typename T>
inline unsigned Image<T>::getSize() const
{
	return width * height * channels * sizeof(T);
}

