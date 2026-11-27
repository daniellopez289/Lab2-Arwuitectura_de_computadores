#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <float.h>
#include <string.h>
#include <cuda_runtime.h>

#define MAX_POINTS 40000000 // Máximo de puntos permitido

// Verifica errores de llamadas a la API de CUDA de forma clara
void check_cuda(cudaError_t err, const char *msg)
{
    if (err != cudaSuccess)
    {
        printf("Error CUDA (%s): %s\n", msg, cudaGetErrorString(err));
        exit(1);
    }
}

// Distancia euclidiana al cuadrado (CPU, GPU)
__host__ __device__ static inline float dist2(float px, float py, float cx, float cy)
{
    float dx = px - cx;
    float dy = py - cy;
    return dx * dx + dy * dy;
}

// Carga puntos (x,y) desde un archivo CSV sin encabezado
int load_csv(const char *filename, float **out_points)
{
    FILE *f = fopen(filename, "r");
    if (!f)
    {
        printf("Error: no se puede abrir %s\n", filename);
        return -1;
    }

    float *points = (float *)malloc((size_t)MAX_POINTS * 2 * sizeof(float));
    if (!points)
    {
        printf("Error reservando memoria en CPU\n");
        fclose(f);
        return -1;
    }

    int count = 0;
    float x, y;
    while (fscanf(f, "%f,%f", &x, &y) == 2)
    {
        if (count >= MAX_POINTS)
            break;
        points[2 * count] = x;
        points[2 * count + 1] = y;
        count++;
    }

    fclose(f);
    *out_points = points;
    return count;
}

// KERNEL GPU

// Cada hilo calcula el centroide más cercano para un único punto i
__global__ void assign_clusters_gpu(
    const float *points,    // tamaño N*2
    const float *centroids, // tamaño K*2
    int *labels,            // tamaño N
    int N, int K)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N)
        return; // Protege el último bloque contra accesos fuera de rango

    float px = points[2 * i];
    float py = points[2 * i + 1];

    float best_dist = FLT_MAX;
    int best_c = 0;

    for (int c = 0; c < K; c++)
    {
        float cx = centroids[2 * c];
        float cy = centroids[2 * c + 1];

        float d = dist2(px, py, cx, cy);
        if (d < best_dist)
        {
            best_dist = d;
            best_c = c;
        }
    }

    labels[i] = best_c;
}

// FUNCIONES CPU

void assign_clusters_cpu(
    const float *points,
    const float *centroids,
    int *labels,
    int N, int K)
{
    for (int i = 0; i < N; i++)
    {
        float px = points[2 * i];
        float py = points[2 * i + 1];

        float best_dist = FLT_MAX;
        int best_c = 0;

        for (int c = 0; c < K; c++)
        {
            float cx = centroids[2 * c];
            float cy = centroids[2 * c + 1];
            float d = dist2(px, py, cx, cy);
            if (d < best_dist)
            {
                best_dist = d;
                best_c = c;
            }
        }
        labels[i] = best_c;
    }
}

float update_centroids_cpu(
    const float *points,
    float *centroids,
    const int *labels,
    int N, int K)
{
    float *sum = (float *)calloc(K * 2, sizeof(float));
    int *count = (int *)calloc(K, sizeof(int));
    if (!sum || !count)
    {
        printf("Error reservando memoria en update_centroids_cpu\n");
        exit(1);
    }

    for (int i = 0; i < N; i++)
    {
        int c = labels[i];
        sum[2 * c] += points[2 * i];
        sum[2 * c + 1] += points[2 * i + 1];
        count[c]++;
    }

    float movement = 0.0f;
    for (int c = 0; c < K; c++)
    {
        float newx = centroids[2 * c];
        float newy = centroids[2 * c + 1];

        if (count[c] > 0)
        {
            newx = sum[2 * c] / count[c];
            newy = sum[2 * c + 1] / count[c];
        }

        float dx = newx - centroids[2 * c];
        float dy = newy - centroids[2 * c + 1];
        movement += dx * dx + dy * dy;

        centroids[2 * c] = newx;
        centroids[2 * c + 1] = newy;
    }

    free(sum);
    free(count);
    return movement;
}

void kmeans_cpu_ref(
    const float *points,
    float *centroids,
    int N, int K,
    int max_iters,
    float epsilon)
{
    int *labels = (int *)malloc((size_t)N * sizeof(int));
    if (!labels)
    {
        printf("Error reservando labels en CPU\n");
        exit(1);
    }

    for (int it = 0; it < max_iters; it++)
    {
        assign_clusters_cpu(points, centroids, labels, N, K);
        float movement = update_centroids_cpu(points, centroids, labels, N, K);
        if (movement < epsilon)
            break;
    }

    free(labels);
}

// K-MEANS GPU BÁSICO

void kmeans_gpu_basic(
    const float *points,
    float *centroids,
    int N, int K,
    int max_iters,
    float epsilon,
    int blockSize,
    float *out_total_ms,
    float *out_kernel_ms)
{
    int *labels = (int *)malloc((size_t)N * sizeof(int));
    if (!labels)
    {
        printf("Error reservando labels en host\n");
        exit(1);
    }

    // Eventos CUDA para medir tiempo real
    cudaEvent_t start_total, stop_total, start_k, stop_k;
    check_cuda(cudaEventCreate(&start_total), "EventCreate start_total");
    check_cuda(cudaEventCreate(&stop_total), "EventCreate stop_total");
    check_cuda(cudaEventCreate(&start_k), "EventCreate start_k");
    check_cuda(cudaEventCreate(&stop_k), "EventCreate stop_k");

    float total_kernel_ms = 0.0f;

    check_cuda(cudaEventRecord(start_total, 0), "Record start_total");

    // Reserva única en GPU para todo el algoritmo
    float *d_points = NULL, *d_centroids = NULL;
    int *d_labels = NULL;
    size_t points_bytes = (size_t)N * 2 * sizeof(float);
    size_t centroids_bytes = (size_t)K * 2 * sizeof(float);
    size_t labels_bytes = (size_t)N * sizeof(int);

    check_cuda(cudaMalloc((void **)&d_points, points_bytes), "cudaMalloc d_points");
    check_cuda(cudaMalloc((void **)&d_centroids, centroids_bytes), "cudaMalloc d_centroids");
    check_cuda(cudaMalloc((void **)&d_labels, labels_bytes), "cudaMalloc d_labels");

    // Los puntos no cambian: se copian a GPU una sola vez antes del ciclo
    check_cuda(cudaMemcpy(d_points, points, points_bytes, cudaMemcpyHostToDevice), "H2D d_points");

    int numBlocks = (N + blockSize - 1) / blockSize;
    printf("Configuracion GPU Basica: numBlocks = %d, blockSize = %d\n", numBlocks, blockSize);

    for (int it = 0; it < max_iters; it++)
    {
        // 1. Enviar centroides actuales a GPU
        check_cuda(cudaMemcpy(d_centroids, centroids, centroids_bytes, cudaMemcpyHostToDevice), "H2D d_centroids");

        // 2. Asignación paralela en GPU
        check_cuda(cudaEventRecord(start_k, 0), "Record start_k");
        assign_clusters_gpu<<<numBlocks, blockSize>>>(d_points, d_centroids, d_labels, N, K);
        check_cuda(cudaGetLastError(), "Kernel launch assign_clusters_gpu");
        check_cuda(cudaEventRecord(stop_k, 0), "Record stop_k");
        check_cuda(cudaEventSynchronize(stop_k), "Sync stop_k");

        float iter_k_ms = 0.0f;
        check_cuda(cudaEventElapsedTime(&iter_k_ms, start_k, stop_k), "Elapsed kernel");
        total_kernel_ms += iter_k_ms;

        // 3. Traer etiquetas de vuelta a CPU y recalcular centroides secuencialmente
        check_cuda(cudaMemcpy(labels, d_labels, labels_bytes, cudaMemcpyDeviceToHost), "D2H d_labels");
        float movement = update_centroids_cpu(points, centroids, labels, N, K);

        printf("Iteration %d - centroid movement = %.6f\n", it, movement);

        if (movement < epsilon)
        {
            printf("Converged after %d iterations.\n", it);
            break;
        }
    }

    check_cuda(cudaEventRecord(stop_total, 0), "Record stop_total");
    check_cuda(cudaEventSynchronize(stop_total), "Sync stop_total");
    check_cuda(cudaEventElapsedTime(out_total_ms, start_total, stop_total), "Elapsed total");
    *out_kernel_ms = total_kernel_ms;

    // Liberación limpia de recursos
    cudaEventDestroy(start_total);
    cudaEventDestroy(stop_total);
    cudaEventDestroy(start_k);
    cudaEventDestroy(stop_k);
    cudaFree(d_points);
    cudaFree(d_centroids);
    cudaFree(d_labels);
    free(labels);
}

// Compara los centroides finales GPU vs CPU con una tolerancia numérica
bool verify(const float *centroids_gpu, const float *centroids_cpu, int K, float tol)
{
    for (int c = 0; c < K; c++)
    {
        float dx = fabsf(centroids_gpu[2 * c] - centroids_cpu[2 * c]);
        float dy = fabsf(centroids_gpu[2 * c + 1] - centroids_cpu[2 * c + 1]);
        if (dx > tol || dy > tol)
        {
            printf("Mismatch en C%d: GPU=(%f, %f) vs CPU=(%f, %f)\n",
                   c, centroids_gpu[2 * c], centroids_gpu[2 * c + 1],
                   centroids_cpu[2 * c], centroids_cpu[2 * c + 1]);
            return false;
        }
    }
    return true;
}

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        printf("Usage: %s data.csv [blockSize]\n", argv[0]);
        return 1;
    }

    int blockSize = 256; // Valor por defecto
    if (argc >= 3)
    {
        blockSize = atoi(argv[2]);
        if (blockSize <= 0 || blockSize > 1024)
        {
            printf("Error: blockSize debe estar entre 1 y 1024\n");
            return 1;
        }
    }

    float *points = NULL;
    int N = load_csv(argv[1], &points);
    if (N <= 0)
        return 1;

    int K = 3;
    float centroids_gpu[6] = {0, 0, 5, 5, 10, 10};
    float centroids_cpu[6] = {0, 0, 5, 5, 10, 10};

    printf("Loaded %d points.\n", N);


    int max_iters = 10;
    float ep = 1e-4f;
    float total_ms = 0.0f, kernel_ms = 0.0f;

    // Ejecución en GPU (versión básica)
    kmeans_gpu_basic(points, centroids_gpu, N, K, max_iters, ep, blockSize, &total_ms, &kernel_ms);

    printf("\nFinal centroids (GPU Basic):\n");
    for (int c = 0; c < K; c++)
    {
        printf("C%d = (%f, %f)\n", c, centroids_gpu[2 * c], centroids_gpu[2 * c + 1]);
    }

    printf("\nTiempo total ciclo K-means (GPU Basic): %.4f ms\n", total_ms);
    printf("Tiempo acumulado solo kernel (GPU Basic): %.4f ms\n", kernel_ms);

    // 2. Verificación numérica contra referencia en CPU
    kmeans_cpu_ref(points, centroids_cpu, N, K, max_iters, ep);
    if (verify(centroids_gpu, centroids_cpu, K, 1e-4f))
    {
        printf("Verificacion contra CPU: CORRECTO\n");
    }
    else
    {
        printf("Verificacion contra CPU: ERROR\n");
    }

    free(points);
    return 0;
}