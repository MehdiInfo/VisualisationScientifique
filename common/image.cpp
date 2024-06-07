#define STB_IMAGE_IMPLEMENTATION
#include "image.h"

void image_t::loadImage(const std::string &path){
    data = stbi_load(path.c_str(), &width, &height, &bpp, 4);
}
void image_t::createWhiteImage(const int w, const int h){;
    width = w;
    height = h;
    bpp = 3;
    data = (unsigned char*)malloc(width*height*4);
    memset(data,255,w*h*3);
}
void image_t::freeImage(){
    stbi_image_free(data);
}

void saveBMP(const char* filePath, uint32_t width, uint32_t height, const uint32_t* pixels) {
    // BMP header
    const uint32_t bmpHeaderSize = 14;
    const uint32_t dibHeaderSize = 40;
    const uint32_t fileSize = bmpHeaderSize + dibHeaderSize + width * height * 3;
    uint8_t bmpHeader[bmpHeaderSize] = { 'B', 'M', 0, 0, 0, 0, 0, 0, 0, 0, bmpHeaderSize + dibHeaderSize, 0, 0, 0 };
    uint8_t dibHeader[dibHeaderSize] = { dibHeaderSize, 0, 0, 0, width & 0xFF, (width >> 8) & 0xFF, (width >> 16) & 0xFF, (width >> 24) & 0xFF, height & 0xFF, (height >> 8) & 0xFF, (height >> 16) & 0xFF, (height >> 24) & 0xFF, 1, 0, 24, 0 };

    // Open file for writing
    std::ofstream bmpFile(filePath, std::ios::binary);
    if (!bmpFile) {
        throw std::runtime_error("Failed to open BMP file for writing");
    }

    // Write headers
    bmpFile.write(reinterpret_cast<const char*>(bmpHeader), bmpHeaderSize);
    bmpFile.write(reinterpret_cast<const char*>(dibHeader), dibHeaderSize);

    // Write pixel data
    for (uint32_t y = 0; y < height; ++y) {
        for (uint32_t x = 0; x < width; ++x) {
            uint32_t i = y * width + x;
            uint8_t blue = (pixels[i] >> 16) & 0xFF;
            uint8_t green = (pixels[i] >> 8) & 0xFF;
            uint8_t red = pixels[i] & 0xFF;
            bmpFile.write(reinterpret_cast<const char*>(&blue), 1); // Blue channel
            bmpFile.write(reinterpret_cast<const char*>(&green), 1); // Green channel
            bmpFile.write(reinterpret_cast<const char*>(&red), 1); // Red channel
        }
    }
}