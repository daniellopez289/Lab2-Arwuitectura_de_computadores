#include <stdio.h>
#include <cuda_runtime.h>

int main()
{
    int dev = 0;
    cudaDeviceProp p;
    cudaError_t err = cudaGetDeviceProperties(&p, dev);
    if (err != cudaSuccess)
    {
        printf("Error cudaGetDeviceProperties: %s\n", cudaGetErrorString(err));
        return 1;
    }

    int clock_khz = 0, mem_clock_khz = 0;
    cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, dev);
    cudaDeviceGetAttribute(&mem_clock_khz, cudaDevAttrMemoryClockRate, dev);

    printf("GPU Name: %s\n", p.name);
    printf("Compute Capability: %d.%d\n", p.major, p.minor);
    printf("Multiprocessors (SMs): %d\n", p.multiProcessorCount);
    printf("Warp Size: %d\n", p.warpSize);
    printf("Max Threads per Block: %d\n", p.maxThreadsPerBlock);
    printf("Max Threads per SM: %d\n", p.maxThreadsPerMultiProcessor);
    printf("Total Global Memory: %.2f GB (%zu bytes)\n",
           p.totalGlobalMem / (1024.0 * 1024.0 * 1024.0), p.totalGlobalMem);
    printf("Shared Memory per Block: %zu KB (%zu bytes)\n",
           p.sharedMemPerBlock / 1024, p.sharedMemPerBlock);
    printf("Shared Memory per SM: %zu KB (%zu bytes)\n",
           p.sharedMemPerMultiprocessor / 1024, p.sharedMemPerMultiprocessor);
    printf("Registers per Block: %d\n", p.regsPerBlock);
    printf("Registers per SM: %d\n", p.regsPerMultiprocessor);
    printf("L2 Cache Size: %.2f MB\n", p.l2CacheSize / (1024.0 * 1024.0));
    printf("Memory Bus Width: %d bits\n", p.memoryBusWidth);
    printf("GPU Clock Rate: %.0f MHz\n", clock_khz / 1000.0);
    printf("Memory Clock Rate: %.0f MHz\n", mem_clock_khz / 1000.0);
    return 0;
}