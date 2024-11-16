#pragma once
#include <string>
#include "Image.h"

class ReadWriteImg
{
public:
	Image<byte> readImage(const std::string& imgPath) const;
	bool writeImage(Image<byte> img, const std::string& imgPath) const;
};

