#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

const int WIDTH = 1024;
const int HEIGHT = 1024;
const int DELTA = 50;

// CPU реализация
void increaseBrightnessCPU(unsigned char* input, unsigned char* output, int width, int height, int delta) {
    for (int i = 0; i < width * height; ++i) {
        int temp = input[i] + delta;
        if (temp > 255) {
            output[i] = 255;
        } else {
            output[i] = temp;
        }
    }
}

// CUDA ядро
__global__ void increaseBrightness(unsigned char* input, unsigned char* output, int width, int height, int delta) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        int temp = input[idx] + delta;
        if (temp > 255) {
            output[idx] = 255;
        } else {
            output[idx] = temp;
        }
    }
}

// Сравнение результатов (будут переданы итоговые массивы)
bool compareResults(unsigned char* a, unsigned char* b, int size) {
    for (int i = 0; i < size; ++i)
        if (a[i] != b[i]) return false;
    return true;
}

int main() {
    size_t size = WIDTH * HEIGHT * sizeof(unsigned char);

    // Выделение памяти на хосте
    unsigned char *h_input = new unsigned char[WIDTH * HEIGHT];
    unsigned char *h_output_cpu = new unsigned char[WIDTH * HEIGHT];
    unsigned char *h_output_gpu = new unsigned char[WIDTH * HEIGHT];

    // Заполнение случайными значениями
    std::srand(std::time(0));
    for (int i = 0; i < WIDTH * HEIGHT; ++i)
        h_input[i] = std::rand() % 256;

    // Реализация на CPU
    auto start_cpu = std::chrono::high_resolution_clock::now();
    increaseBrightnessCPU(h_input, h_output_cpu, WIDTH, HEIGHT, DELTA);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpu_time = end_cpu - start_cpu;

    // Реализация на GPU
    unsigned char *d_input, *d_output;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);

    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((WIDTH + blockSize.x - 1) / blockSize.x, (HEIGHT + blockSize.y - 1) / blockSize.y);

    auto start_gpu = std::chrono::high_resolution_clock::now();
    increaseBrightness<<<gridSize, blockSize>>>(d_input, d_output, WIDTH, HEIGHT, DELTA);
    cudaDeviceSynchronize();
    auto end_gpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> gpu_time = end_gpu - start_gpu;

    cudaMemcpy(h_output_gpu, d_output, size, cudaMemcpyDeviceToHost);

    // Сравнение
    bool is_correct = compareResults(h_output_cpu, h_output_gpu, WIDTH * HEIGHT);
    std::cout << "Сравнение с CPU реализацией: " << std::endl;
    if (is_correct == true){
        std::cout << "Реализация одинаковая. " << std::endl;
    } else {
        std::cout << "Реализация разная. " << std::endl;
    }
    std::cout << "Время выполнения на CPU: " << cpu_time.count() << " сек. \n";
    std::cout << "Время выполнения на GPU: " << gpu_time.count() << " сек. \n";

    // Освобождение памяти
    delete[] h_input;
    delete[] h_output_cpu;
    delete[] h_output_gpu;
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}