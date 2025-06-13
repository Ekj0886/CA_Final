#include <bits/stdc++.h>
#include "BMP.h"

#define F first 
#define S second

using namespace std;

// Load BMP image with correct row-padding handling
bool BMP::LoadBMP(string filePath) {
    ifstream file(filePath, ios::binary);
    if (!file) {
        cerr << "Error opening file: " << filePath << endl;
        return false;
    }

    // Read BMP headers
    file.read(reinterpret_cast<char*>(&fileheader), sizeof(BMPFileHeader));
    if (fileheader.fileType != 0x4D42) { // 'BM'
        cerr << "Not a valid BMP file" << endl;
        return false;
    }

    file.read(reinterpret_cast<char*>(&infoheader), sizeof(BMPInfoHeader));
    if (infoheader.bitsPerPixel != 24 && infoheader.bitsPerPixel != 32) {
        cerr << "Only 24-bit/32-bit BMP files are supported." << endl;
        return false;
    }

    size_t bytesPerPixel = infoheader.bitsPerPixel / 8;
    size_t rowSize = (infoheader.width * bytesPerPixel + 3) & ~3; // padded row
    size_t pixelRowBytes = infoheader.width * bytesPerPixel;
    int height = abs(infoheader.height);
    BMP_pixel.resize(height * pixelRowBytes);

    file.seekg(fileheader.dataOffset, ios::beg);

    // Read each row, skip padding
    for (int row = 0; row < height; ++row) {
        file.read(reinterpret_cast<char*>(&BMP_pixel[row * pixelRowBytes]), pixelRowBytes);
        file.ignore(rowSize - pixelRowBytes);
    }

    file.close();
    return true;
}

// Dump BMP image with correct padding
bool BMP::DumpImageToBMP(const string& file_name) {
    ofstream bmp_file(file_name, ios::binary);
    if (!bmp_file) {
        cerr << "Error: Could not open file for writing: " << file_name << endl;
        return false;
    }

    int height = abs(infoheader.height);
    int width = infoheader.width;
    int bpp = infoheader.bitsPerPixel;
    int bytesPerPixel = bpp / 8;
    int row_padding = (4 - (width * bytesPerPixel) % 4) % 4;
    int pixelRowBytes = width * bytesPerPixel;

    fileheader.fileSize = 14 + 40 + (pixelRowBytes + row_padding) * height;
    fileheader.dataOffset = 14 + 40;
    infoheader.headerSize = 40;
    infoheader.planes = 1;
    infoheader.compression = 0;
    infoheader.imageSize = (pixelRowBytes + row_padding) * height;

    bmp_file.write(reinterpret_cast<const char*>(&fileheader), sizeof(fileheader));
    bmp_file.write(reinterpret_cast<const char*>(&infoheader), sizeof(infoheader));

    // Write rows in BGR(A) with padding
    for (int row = 0; row < height; ++row) {
        const char* rowPtr = reinterpret_cast<const char*>(&BMP_pixel[row * pixelRowBytes]);
        bmp_file.write(rowPtr, pixelRowBytes);
        bmp_file.write("\0\0\0\0", row_padding);  // safe even for 0–3 bytes
    }

    bmp_file.close();
    return true;
}
