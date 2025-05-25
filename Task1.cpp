#include <iostream> 
#include <cuda_runtime.h>
#include <chrono> 

__global__ void vectorAddCUDA(float *a, float *b, float *c, int size) { 
    int idx = threadIdx.x + blockIdx.x * blockDim.x; 
    if (idx < size) 
        c[idx] = a[idx] + b[idx]; 
} 

void vectorAddCPU(float *a, float *b, float *c, int N) {
    for (int i = 0; i < N; ++i) {
        c[i] = a[i] + b[i];
    }
}

int main() { 
    int N = 1000000; 
    size_t size = N * sizeof(float); 

    // Выделение памяти на хосте 
    float *h_a = new float[N]; 
    float *h_b = new float[N]; 
    float *h_c = new float[N]; 

    // Инициализация входных данных 
    for (int i = 0; i < N; ++i) { 
        h_a[i] = i*1.5f; 
        h_b[i] = i*2.5f; 
    } 

    auto start_countOnCPU = std::chrono::high_resolution_clock::now();
    vectorAddCPU(h_a, h_b, h_c, N);
    auto end_countOnCPU = std::chrono::high_resolution_clock::now();
    //Вывод первых 10 значений
    for (int k = 0; k < 10; k++) {
        std::cout << h_a[k] << " + " << h_b[k] << " = " << h_c[k] << std::endl;
    }
    std::cout << "Время выполнения сложения на CPU: " << std::chrono::duration<double>(end_countOnCPU - start_countOnCPU).count() << " сек." << std::endl;

    // Выделение памяти на устройстве 
    float *d_a, *d_b, *d_c; 
    cudaMalloc((void**)&d_a, size); 
    cudaMalloc((void**)&d_b, size); 
    cudaMalloc((void**)&d_c, size); 

    // Копирование данных с хоста на устройство 
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice); 
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice); 

    // Настройка сетки и блоков 
    int threadsPerBlock = 256; 
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock; 
    
    // Запуск ядра 
    auto start_countOnGPU = std::chrono::high_resolution_clock::now();
    vectorAddCUDA<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N); 
    cudaDeviceSynchronize();  // Ожидание завершения
    auto end_countOnGPU = std::chrono::high_resolution_clock::now();

    // Копирование результата обратно 
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost); 

    // Вывод первых 10 значений 
    for (int i = 0; i < 10; ++i) 
    std::cout << h_a[i] << " + " << h_b[i] << " = " << h_c[i] << std::endl;
    std::cout << "Время выполнения сложения на GPU: " << std::chrono::duration<double>(end_countOnGPU - start_countOnGPU).count() << " сек." << std::endl;

    // Очистка 
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c); 
    delete[] h_a; delete[] h_b; delete[] h_c; 
    return 0; 
}