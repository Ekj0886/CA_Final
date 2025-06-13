#include <bits/stdc++.h>
#include <getopt.h> 
#include <chrono>
#include "IMAGE.h"

#define F first 
#define S second

using namespace std;

int main(int argc, char *argv[]) {

    if(argc < 2) {
        cerr << "missing input or output file" << endl;
        return 0;
    }    

    string input_file  = argv[argc - 2];
    string output_file = argv[argc - 1]; 
    IMAGE* image = new IMAGE(input_file);

    static struct option long_options[] = {
        {"originimage",   required_argument, nullptr, 'o'},
        {"sharpness_CPU_2D",     required_argument, nullptr, 'd'},
        {"sharpness_CPU_1D",     required_argument, nullptr, 's'},
        {"sharpness_CUDA", required_argument, nullptr, 'c' },
        {nullptr,       0,                 nullptr,  0 } 
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "d:s:c:", long_options, nullptr)) != -1) {
        switch (opt) {
            case 'd': {
                auto t0 = chrono::high_resolution_clock::now();
                
                cout << "-- improve sharpness with CPU 2D " << input_file << endl;
                image->EnhanceSharpness_2D(atoi(optarg));

                auto t1 = chrono::high_resolution_clock::now();
                printf("[CPU_2D] %.3f ms\n", chrono::duration<float, std::milli>(t1 - t0).count());
                break;
            }
            case 's': {
                auto t0 = chrono::high_resolution_clock::now();
                
                cout << "-- improve sharpness " << input_file << endl;
                image->EnhanceSharpness_1D(atoi(optarg));

                auto t1 = chrono::high_resolution_clock::now();
                printf("[CPU_1D] %.3f ms\n", chrono::duration<float, std::milli>(t1 - t0).count());
                break;
            }
            case 'c': {
                auto t0 = chrono::high_resolution_clock::now();

                cout << "-- improve sharpness with CUDA " << input_file << endl;
                image->EnhanceSharpnessCUDA(atoi(optarg));

                auto t1 = chrono::high_resolution_clock::now();
                printf("[GPU] %.3f ms\n", chrono::duration<float, std::milli>(t1 - t0).count());
                break;
            }
            default:
                break;
        }
    }

    image->DumpImage(output_file);

    return 0;

}