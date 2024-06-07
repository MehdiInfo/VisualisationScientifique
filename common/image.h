#ifndef __IMAGE_H__
#define __IMAGE_H__

#include <string>
#include <cstdlib>
#include <stdint.h>
#include <iostream>
#include "stb_image.h"
#include "stb_image_write.h"
#include <fstream>
#include <cstdint>


struct color_t{
    unsigned char r,g,b;
};

struct image_t{
    int width,height,bpp;
    unsigned char *data;

    void loadImage(const std::string &path);
    void createWhiteImage(const int w, const int h);
    void freeImage();
};

void saveBMP(const char* filePath, uint32_t width, uint32_t height, const uint32_t* pixels) ;
#endif