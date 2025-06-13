#ifndef IMAGE_H
#define IMAGE_H
#include <bits/stdc++.h>
#include "BMP.h"

#define F first 
#define S second

using namespace std;

struct PIXEL {
    uint8_t R, G, B, A;
    float Y, Cb, Cr;
    PIXEL(uint8_t r = 0, uint8_t g = 0, uint8_t b = 0, uint8_t a = 0) : R(r), G(g), B(b), A(a) {}
};

class IMAGE {

private:
    string name;
    BMP bmp_image;
    vector<vector<PIXEL>> pixel;
    int W;
    int H;
    int bit;
    
public:
    // constructor
    IMAGE() {};
    IMAGE(string file) { LoadImage(file); name = file; To_YCbCr(); }
    
    // member function
    void LoadImage(string);
    int  GetW() { return W; }    
    int  GetH() { return H; }
    int  GetBit() { return bit; }
    float Clamp(float, float, float);
    void To_YCbCr();
    void To_RGB();
    void DumpImage(string);
    vector<vector<PIXEL>>& GetPixel() { return pixel; }

    // Gaussian Blur using 2D convolution + Sharpness Enhancement
    void EnhanceSharpness_2D(int);
    void GaussianBlur2D(int k, float sigma);
    
    // Gaussian Blur using 1D convolution + Sharpness Enhancement
    void GaussianBlur(int k, float sigma);
    void EnhanceSharpness_1D(int);

    // CUDA functions
    void EnhanceSharpnessCUDA(int level);

};

#endif
