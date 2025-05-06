#define _CRT_SECURE_NO_WARNINGS
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <iostream>
#include <mpi.h>
#include <chrono>

using namespace std;
using namespace std::chrono;

unsigned char* resizeChunk(
    unsigned char* input, int width, int height, int channels,
    int start_y, int chunk_height, int new_width, int new_height
) {
    float x_ratio = float(width) / new_width;
    float y_ratio = float(height) / new_height;

    unsigned char* output = new unsigned char[chunk_height * new_width * channels];
    if (!output) {
        cerr << "Memory allocation failed!\n";
        return nullptr;
    }

    for (int y = 0; y < chunk_height; y++) {
        int global_y = start_y + y;
        int py = int(global_y * y_ratio);
        for (int x = 0; x < new_width; x++) {
            int px = int(x * x_ratio);
            for (int c = 0; c < channels; c++) {
                output[(y * new_width + x) * channels + c] =
                    input[(py * width + px) * channels + c];
            }
        }
    }

    return output;
}

int main(int argc, char** argv) {
    steady_clock::time_point start_time;
    start_time = steady_clock::now();
    MPI_Init(&argc, &argv);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int width, height, channels;
    int new_width = 7500;
    int new_height = 7500;
    unsigned char* img = nullptr;

    if (rank == 0) {
        img = stbi_load("chemistry.jpg", &width, &height, &channels, 0);
        if (!img) {
            cerr << "Failed to load image!\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&channels, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&new_width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&new_height, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int chunk_height = new_height / num_procs;
    int start_y = rank * chunk_height;
    if (rank == num_procs - 1) {
        chunk_height = new_height - start_y; 
    }

    unsigned char* local_resized_chunk = nullptr;
    if (rank == 0) {
        local_resized_chunk = resizeChunk(img, width, height, channels, start_y, chunk_height, new_width, new_height);
    }
    else {
   
        img = new unsigned char[width * height * channels];
    }

    MPI_Bcast(img, width * height * channels, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    if (rank != 0) {
        local_resized_chunk = resizeChunk(img, width, height, channels, start_y, chunk_height, new_width, new_height);
    }

    unsigned char* final_image = nullptr;
    int chunk_size = chunk_height * new_width * channels;
    if (rank == 0) {
        final_image = new unsigned char[new_width * new_height * channels];
    }

    int* recvcounts = nullptr;
    int* displs = nullptr;
    if (rank == 0) {
        recvcounts = new int[num_procs];
        displs = new int[num_procs];
        for (int i = 0; i < num_procs; i++) {
            int start = i * (new_height / num_procs);
            int height_chunk = (i == num_procs - 1) ? (new_height - start) : (new_height / num_procs);
            recvcounts[i] = height_chunk * new_width * channels;
            displs[i] = start * new_width * channels;
        }
    }

    MPI_Gatherv(
        local_resized_chunk,
        chunk_height * new_width * channels,
        MPI_UNSIGNED_CHAR,
        final_image,
        recvcounts,
        displs,
        MPI_UNSIGNED_CHAR,
        0,
        MPI_COMM_WORLD
    );

    if (rank == 0) {
        stbi_write_jpg("output_mpi3.jpg", new_width, new_height, channels, final_image, 100);
        cout << "Image resized and saved by rank 0.\n";
        delete[] final_image;
        delete[] recvcounts;
        delete[] displs;
    }

    delete[] local_resized_chunk;
    if (rank != 0) {
        delete[] img;
    }
    else {
        stbi_image_free(img);
    }

    MPI_Finalize();
    auto end_time = steady_clock::now();
    auto duration = duration_cast<seconds>(end_time - start_time).count();
    cout << "Total time taken: " << duration << " s\n";
    return 0;
}
