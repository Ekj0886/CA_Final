#include <bits/stdc++.h>
#include "IMAGE.h"

#define F first 
#define S second

using namespace std;

// functions defined in header file

// Basic functions for IMAGE class
void IMAGE::LoadImage(string file) {
    if(!bmp_image.LoadBMP(file) && file != "-h") {
        cerr << "Failed to load image" << endl;
        exit(0);
    }

    W = bmp_image.GetW();
    H = bmp_image.GetH();
    bit = bmp_image.GetBit();

    // load 
    pixel.resize(H, vector<PIXEL>(W));

    // Get raw pixel data from the BMP image (assuming RGB 8-bit per channel)
    vector<uint8_t> raw_data = bmp_image.GetPixel(); 

    // Assuming each pixel is stored as 3 consecutive bytes (R, G, B)
    int index = 0;  // Index in the raw data array
    for (int i = 0; i < H; ++i) {
        for (int j = 0; j < W; ++j) {
            if(bit == 24) {
                // Read RGB values from the raw pixel data
                uint8_t B = raw_data[index++];
                uint8_t G = raw_data[index++];
                uint8_t R = raw_data[index++];
                // Assign the RGB values to the PIXEL struct
                pixel[i][j] = PIXEL(R, G, B, 0);
            }
            else {
                // Read RGB values from the raw pixel data
                uint8_t B = raw_data[index++];
                uint8_t G = raw_data[index++];
                uint8_t R = raw_data[index++];
                uint8_t A = raw_data[index++];
                // Assign the RGB values to the PIXEL struct
                pixel[i][j] = PIXEL(R, G, B, A);
            }
        }
    }

    std::cout << "-- Load Image " << file << endl;

}

float IMAGE::Clamp(float d, float low, float high) {
    if(d < low) return low;
    if(d > high) return high;
    return d;
}

void IMAGE::To_YCbCr() {
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {
            float R = static_cast<float>(pixel[i][j].R);
            float G = static_cast<float>(pixel[i][j].G);
            float B = static_cast<float>(pixel[i][j].B);

            // Rec. 601 conversion
            float Y  =  0.299 * R + 0.587 * G + 0.114 * B;
            float Cb = -0.168736 * R - 0.331264 * G + 0.5 * B + 128;
            float Cr =  0.5 * R - 0.418688 * G - 0.081312 * B + 128;
            
            pixel[i][j].Y  = Clamp(Y, 0, 255);
            pixel[i][j].Cb = Clamp(Cb, 0, 255);
            pixel[i][j].Cr = Clamp(Cr, 0, 255);
        }
    }
}

void IMAGE::To_RGB() {
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {

            pixel[i][j].Y  = Clamp(pixel[i][j].Y, 0, 255);
            float Y  = static_cast<float>(pixel[i][j].Y);
            float Cb = static_cast<float>(pixel[i][j].Cb) - 128;
            float Cr = static_cast<float>(pixel[i][j].Cr) - 128;

            // Convert YCbCr back to RGB with precise coefficients
            int R = static_cast<int>(Y + 1.402 * Cr);
            int G = static_cast<int>(Y - 0.344136 * Cb - 0.714136 * Cr);
            int B = static_cast<int>(Y + 1.772 * Cb);

            // Clamp and assign back to the pixel
            pixel[i][j].R = static_cast<uint8_t>(std::min(255, std::max(0, R)));
            pixel[i][j].G = static_cast<uint8_t>(std::min(255, std::max(0, G)));
            pixel[i][j].B = static_cast<uint8_t>(std::min(255, std::max(0, B)));
        }
    }
}

void IMAGE::DumpImage(string file) {

    bmp_image.UpdateWH(W, H);

    vector<uint8_t> new_pixel;
    if(bit == 24) {
        for(int i = 0; i < H; i++) {
            for(int q = 0; q < W; q++) {
                new_pixel.push_back(uint8_t(pixel[i][q].B));
                new_pixel.push_back(uint8_t(pixel[i][q].G));
                new_pixel.push_back(uint8_t(pixel[i][q].R));
            }
        }
    }
    else {
        for(int i = 0; i < H; i++) {
            for(int q = 0; q < W; q++) {
                new_pixel.push_back(uint8_t(pixel[i][q].B));
                new_pixel.push_back(uint8_t(pixel[i][q].G));
                new_pixel.push_back(uint8_t(pixel[i][q].R));
                new_pixel.push_back(uint8_t(pixel[i][q].A));
            }
        }
    }
    bmp_image.UpdatePixel(new_pixel);
    bmp_image.DumpImageToBMP(file);
    // std::cout << "-- Dump Image " << name << " to " << file << endl;
}


// Gaussian Blur using 2D convolution + Sharpness Enhancement
void IMAGE::GaussianBlur2D(int k, float sigma) {
    int half = k / 2;

    // Generate 2D Gaussian kernel
    vector<vector<float>> kernel(k, vector<float>(k));
    float total = 0.0f;
    for (int i = -half; i <= half; ++i) {
        for (int j = -half; j <= half; ++j) {
            float value = exp(-(i * i + j * j) / (2 * sigma * sigma));
            kernel[i + half][j + half] = value;
            total += value;
        }
    }

    // Normalize kernel
    for (int i = 0; i < k; ++i)
        for (int j = 0; j < k; ++j)
            kernel[i][j] /= total;

    // Make a copy of the original Y channel
    vector<vector<float>> Y_orig(H, vector<float>(W));
    for (int i = 0; i < H; ++i)
        for (int j = 0; j < W; ++j)
            Y_orig[i][j] = pixel[i][j].Y;

    // Apply 2D convolution
    for (int i = 0; i < H; ++i) {
        for (int j = 0; j < W; ++j) {
            float sum = 0.0f;

            for (int u = -half; u <= half; ++u) {
                for (int v = -half; v <= half; ++v) {
                    int y = i + u;
                    int x = j + v;
                    x = min(max(x, 0), W - 1);
                    y = min(max(y, 0), H - 1);
                    sum += Y_orig[y][x] * kernel[u + half][v + half];
                }
            }

            pixel[i][j].Y = Clamp(sum, 0.0f, 255.0f);
        }
    }

    To_RGB();
}

void IMAGE::EnhanceSharpness_2D(int level) {
    IMAGE LPF_Image = *this;
    LPF_Image.GaussianBlur2D(13, 1);
    for(int i = 0; i < H; i++) {
        for(int j = 0; j < W; j++) {
            pixel[i][j].Y += level * (pixel[i][j].Y - LPF_Image.pixel[i][j].Y);
        }
    }
    To_RGB();
}


// Gaussian Blur using 1D convolution + Sharpness Enhancement
void IMAGE::GaussianBlur(int k, float sigma) {

    // Generate 1D Gaussian kernel
    vector<float> kernel(k); 
    k /= 2;
    float total = 0;
    for(int i = -k; i <= k; i++) {
        kernel[i+k] = exp(-1 * (i * i) / (2 * sigma * sigma) );
        total += kernel[i+k];
    }

    for(auto& kernel_value : kernel) {
        kernel_value /= total;
    }

    // Perform horizontal convolution 
    for(int i = 0; i < H; i++) {
        for(int j = 0; j < W; j++) {
            
            float Xconv = 0;
            float Xsum = 0; 
            for(int r = -k; r <= k; r++) {
                int x = r + j;
                int y = i;

                if(x < 0 || y < 0 || x >= W || y >= H) continue; // zero padding
                
                Xconv += pixel[y][x].Y * kernel[r + k];
                Xsum += kernel[r + k];
            }

            pixel[i][j].Y = Xconv / Xsum;
            pixel[i][j].Y  = Clamp(pixel[i][j].Y, 0, 255);
        }
    }

    // Perform vertical convolution 
    for(int i = 0; i < H; i++) {
        for(int j = 0; j < W; j++) {
            
            float Yconv = 0;
            float Ysum = 0;
            for(int r = -k; r <= k; r++) {
                int x = j;
                int y = r + i;

                if(x < 0 || y < 0 || x >= W || y >= H) continue; // zero padding
                
                Yconv += pixel[y][x].Y * kernel[r + k];
                Ysum += kernel[r + k];
            }

            pixel[i][j].Y = Yconv / Ysum;
            pixel[i][j].Y  = Clamp(pixel[i][j].Y, 0, 255);
        }
    }

    To_RGB();
}

void IMAGE::EnhanceSharpness_1D(int level) {
    IMAGE LPF_Image = *this;
    LPF_Image.GaussianBlur(13, 1);
    for(int i = 0; i < H; i++) {
        for(int j = 0; j < W; j++) {
            pixel[i][j].Y += level * (pixel[i][j].Y - LPF_Image.pixel[i][j].Y);
        }
    }
    To_RGB();
}


// CUDA Optimized functions
vector<float> genGaussian2DKernel(int k, float sigma) {
    vector<float> kernel(k * k);
    int half = k / 2;
    float sum = 0;
    for (int y = -half; y <= half; y++) {
        for (int x = -half; x <= half; x++) {
            float val = exp(-(x * x + y * y) / (2 * sigma * sigma));
            kernel[(y + half) * k + (x + half)] = val;
            sum += val;
        }
    }
    for (auto& v : kernel) v /= sum;
    return kernel;
}

__global__ void gaussianBlur2DShared(float* input, float* output, float* kernel, int k, int W, int H) {
    extern __shared__ float shared[];

    int tx = threadIdx.x, ty = threadIdx.y;
    int x = blockIdx.x * blockDim.x + tx;
    int y = blockIdx.y * blockDim.y + ty;
    int half = k / 2;

    int smW = blockDim.x + k - 1;
    int smH = blockDim.y + k - 1;

    // Load to shared
    for (int dy = ty; dy < smH; dy += blockDim.y) {
        for (int dx = tx; dx < smW; dx += blockDim.x) {
            int gx = blockIdx.x * blockDim.x + dx - half;
            int gy = blockIdx.y * blockDim.y + dy - half;
            gx = min(max(gx, 0), W - 1);
            gy = min(max(gy, 0), H - 1);
            shared[dy * smW + dx] = input[gy * W + gx];
        }
    }

    __syncthreads();

    if (x >= W || y >= H) return;

    float sum = 0, wsum = 0;
    for (int j = 0; j < k; ++j) {
        for (int i = 0; i < k; ++i) {
            float val = shared[(ty + j) * smW + (tx + i)];
            float w = kernel[j * k + i];
            sum += val * w;
            wsum += w;
        }
    }
    output[y * W + x] = sum / wsum;
}

__global__ void enhanceSharpnessKernel(float* srcY, float* blurredY, float* outY, int level, int W, int H) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= W || y >= H) return;

    int idx = y * W + x;
    float diff = srcY[idx] - blurredY[idx];
    float val = srcY[idx] + level * diff;
    outY[idx] = min(255.0f, max(0.0f, val));
}

void IMAGE::EnhanceSharpnessCUDA(int level) {
    // int size = H * W * sizeof(float);
    int k = 13;
    float sigma = 1.0f;
    auto kernel = genGaussian2DKernel(k, sigma);

    // Allocate with managed memory
    float *Y, *Y_blur, *Y_out, *K;
    cudaMallocManaged(&Y, H * W * sizeof(float));
    cudaMallocManaged(&Y_blur, H * W * sizeof(float));
    cudaMallocManaged(&Y_out, H * W * sizeof(float));
    cudaMallocManaged(&K, k * k * sizeof(float));

    for (int i = 0; i < H; ++i)
        for (int j = 0; j < W; ++j)
            Y[i * W + j] = pixel[i][j].Y;
    memcpy(K, kernel.data(), k * k * sizeof(float));

    dim3 block(16, 16);
    dim3 grid((W + 15) / 16, (H + 15) / 16);
    int sharedMemSize = (block.x + k - 1) * (block.y + k - 1) * sizeof(float);

    gaussianBlur2DShared<<<grid, block, sharedMemSize>>>(Y, Y_blur, K, k, W, H);
    cudaDeviceSynchronize();

    enhanceSharpnessKernel<<<grid, block>>>(Y, Y_blur, Y_out, level, W, H);
    cudaDeviceSynchronize();

    for (int i = 0; i < H; ++i)
        for (int j = 0; j < W; ++j)
            pixel[i][j].Y = Y_out[i * W + j];

    cudaFree(Y); cudaFree(Y_blur); cudaFree(Y_out); cudaFree(K);
    To_RGB();
}