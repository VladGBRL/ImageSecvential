#define _CRT_SECURE_NO_WARNINGS
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <iostream>
#include <chrono>

using namespace std;
using namespace std::chrono;

static unsigned char* resizeImage(unsigned char* input, int width, int height, int channels, int new_width, int new_height) {
    unsigned char* output = new unsigned char[new_width * new_height * channels];
    if (!output) {
        cerr << "Eroare: alocare memorie esuata!\n";
        return nullptr;
    }

    float x_ratio = float(width) / new_width;
    float y_ratio = float(height) / new_height;

    for (int y = 0; y < new_height; y++) {
        for (int x = 0; x < new_width; x++) {
            int px = int(x * x_ratio);
            int py = int(y * y_ratio);

            for (int c = 0; c < channels; c++) {
                output[(y * new_width + x) * channels + c] = input[(py * width + px) * channels + c];
            }
        }
    }
    return output;
}

int main() {
    int width, height, channels;

    auto start = high_resolution_clock::now();

    unsigned char* img = stbi_load("chemistry.jpg", &width, &height, &channels, 0);
    if (!img) {
        std::cerr << "Eroare la incarcarea imaginii!\n";
        return -1;
    }

    int new_width =7500;
    int new_height = 7500;


    unsigned char* resized_img = resizeImage(img, width, height, channels, new_width, new_height);


    stbi_write_jpg("output2.jpg", new_width, new_height, channels, resized_img, 100);
    if (!resized_img) {
        stbi_image_free(img);
        return -1;
    }

    stbi_image_free(img);
    delete[] resized_img;

    cout << "Imaginea a fost redimensionata si salvata!\n";
    auto end = high_resolution_clock::now();
    duration<double> elapsed = end - start;
    cout << "Timpul de redimensionare: " << elapsed.count() << " secunde\n";

    return 0;
}
