#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <float.h>
#include <string.h>
#include <cuda_runtime.h>

#define MAX_POINTS 10000000 // Máximo de puntos permitido
#define MAX_K 8             // Soporta K=3 y el análisis opcional K=2..8 en registros

// Verifica errores de CUDA de forma directa
void check_cuda(cudaError_t err, const char *msg)
{
    if (err != cudaSuccess)
    {
        printf("Error CUDA (%s): %s\n", msg, cudaGetErrorString(err));
        exit(1);
    }
}

// Distancia euclidiana al cuadrado (CPU y GPU)
__host__ __device__ static inline float dist2(float px, float py, float cx, float cy)
{
    float dx = px - cx;
    float dy = py - cy;
    return dx * dx + dy * dy;
}

// Carga puntos (x,y) desde un archivo CSV
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

// Guarda etiquetas de una iteración en CSV
void save_labels_csv(const char *prefix, int it, const float *points, const int *labels, int N)
{
    char fname[256];
    snprintf(fname, sizeof(fname), "%s_iter_%d.csv", prefix, it);
    FILE *f = fopen(fname, "w");
    if (!f)
    {
        printf("Error creando %s\n", fname);
        exit(1);
    }
    fprintf(f, "x,y,label\n");
    for (int i = 0; i < N; i++)
    {
        fprintf(f, "%.4f,%.4f,%d\n", points[2 * i], points[2 * i + 1], labels[i]);
    }
    fclose(f);
}

// KERNEL GPU

__global__ void assign_and_accumulate_gpu(
    const float *px,
    const float *py,
    const float *centroids,
    int *labels,
    float *d_sum_x,
    float *d_sum_y,
    int *d_count,
    int N, int K)
{
    __shared__ float s_centroids[MAX_K * 2];

    extern __shared__ float smem[];
    float *s_sum_x = smem;
    float *s_sum_y = smem + blockDim.x;
    int   *s_count = (int *)(smem + 2 * blockDim.x);

    int tid = threadIdx.x;

    // Carga cooperativa de centroides a memoria compartida
    for (int idx = tid; idx < 2 * K; idx += blockDim.x)
    {
        s_centroids[idx] = centroids[idx];
    }
    __syncthreads();

    // Acumuladores privados por hilo
    float my_sum_x[MAX_K];
    float my_sum_y[MAX_K];
    int   my_count[MAX_K];

    #pragma unroll
    for (int c = 0; c < MAX_K; c++)
    {
        my_sum_x[c] = 0.0f;
        my_sum_y[c] = 0.0f;
        my_count[c] = 0;
    }

    // Grid-stride loop con acceso coalescido (SoA)
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < N; i += stride)
    {
        float x = px[i];
        float y = py[i];

        float best_dist = FLT_MAX;
        int best_c = 0;

        for (int c = 0; c < K; c++)
        {
            float cx = s_centroids[2 * c];
            float cy = s_centroids[2 * c + 1];
            float d = dist2(x, y, cx, cy);
            if (d < best_dist)
            {
                best_dist = d;
                best_c = c;
            }
        }

        labels[i] = best_c;

        #pragma unroll
        for (int c = 0; c < MAX_K; c++)
        {
            if (c == best_c)
            {
                my_sum_x[c] += x;
                my_sum_y[c] += y;
                my_count[c] += 1;
            }
        }
    }

    // Reducción en árbol por bloque + 1 atomicAdd global por bloque y clúster
    for (int c = 0; c < K; c++)
    {
        s_sum_x[tid] = my_sum_x[c];
        s_sum_y[tid] = my_sum_y[c];
        s_count[tid] = my_count[c];
        __syncthreads();

        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
        {
            if (tid < s)
            {
                s_sum_x[tid] += s_sum_x[tid + s];
                s_sum_y[tid] += s_sum_y[tid + s];
                s_count[tid] += s_count[tid + s];
            }
            __syncthreads();
        }

        if (tid == 0 && s_count[0] > 0)
        {
            atomicAdd(&d_sum_x[c], s_sum_x[0]);
            atomicAdd(&d_sum_y[c], s_sum_y[0]);
            atomicAdd(&d_count[c], s_count[0]);
        }
        __syncthreads();
    }
}

// FUNCIONES CPU

void assign_clusters_cpu(const float *points, const float *centroids, int *labels, int N, int K)
{
    for (int i = 0; i < N; i++)
    {
        float px = points[2 * i];
        float py = points[2 * i + 1];
        float best_dist = FLT_MAX;
        int best_c = 0;

        for (int c = 0; c < K; c++)
        {
            float d = dist2(px, py, centroids[2 * c], centroids[2 * c + 1]);
            if (d < best_dist)
            {
                best_dist = d;
                best_c = c;
            }
        }
        labels[i] = best_c;
    }
}

float update_centroids_cpu(const float *points, float *centroids, const int *labels, int N, int K)
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

void kmeans_cpu_ref(const float *points, float *centroids, int N, int K, int max_iters, float epsilon)
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

//K-MEANS GPU

void kmeans_gpu_advanced(
    const float *points,
    float *centroids,
    int N, int K,
    int max_iters,
    float epsilon,
    int blockSize,
    int requestedBlocks,
    const char *export_prefix, // NULL en benchmark => cero overhead extra
    float *out_total_ms,
    float *out_kernel_ms)
{
    if (K > MAX_K)
    {
        printf("Error: K=%d excede MAX_K=%d\n", K, MAX_K);
        exit(1);
    }

    float *h_px = (float *)malloc((size_t)N * sizeof(float));
    float *h_py = (float *)malloc((size_t)N * sizeof(float));
    float *h_sum_x = (float *)malloc((size_t)K * sizeof(float));
    float *h_sum_y = (float *)malloc((size_t)K * sizeof(float));
    int   *h_count = (int *)malloc((size_t)K * sizeof(int));
    int   *h_labels = export_prefix ? (int *)malloc((size_t)N * sizeof(int)) : NULL;

    if (!h_px || !h_py || !h_sum_x || !h_sum_y || !h_count || (export_prefix && !h_labels))
    {
        printf("Error reservando memoria en host\n");
        exit(1);
    }

    for (int i = 0; i < N; i++)
    {
        h_px[i] = points[2 * i];
        h_py[i] = points[2 * i + 1];
    }

    FILE *f_cent = NULL;
    if (export_prefix)
    {
        char fname[256];
        snprintf(fname, sizeof(fname), "%s_centroids.csv", export_prefix);
        f_cent = fopen(fname, "w");
        if (!f_cent)
        {
            printf("Error creando %s\n", fname);
            exit(1);
        }
        fprintf(f_cent, "iter,cluster,x,y\n");
        // Guarda las semillas iniciales como iter = -1
        for (int c = 0; c < K; c++)
        {
            fprintf(f_cent, "-1,%d,%.6f,%.6f\n", c, centroids[2 * c], centroids[2 * c + 1]);
        }
    }

    float *d_px = NULL, *d_py = NULL, *d_centroids = NULL;
    float *d_sum_x = NULL, *d_sum_y = NULL;
    int   *d_labels = NULL, *d_count = NULL;

    size_t coord_bytes     = (size_t)N * sizeof(float);
    size_t centroids_bytes = (size_t)K * 2 * sizeof(float);
    size_t labels_bytes    = (size_t)N * sizeof(int);
    size_t k_float_bytes   = (size_t)K * sizeof(float);
    size_t k_int_bytes     = (size_t)K * sizeof(int);

    check_cuda(cudaMalloc((void **)&d_px, coord_bytes), "cudaMalloc d_px");
    check_cuda(cudaMalloc((void **)&d_py, coord_bytes), "cudaMalloc d_py");
    check_cuda(cudaMalloc((void **)&d_centroids, centroids_bytes), "cudaMalloc d_centroids");
    check_cuda(cudaMalloc((void **)&d_labels, labels_bytes), "cudaMalloc d_labels");
    check_cuda(cudaMalloc((void **)&d_sum_x, k_float_bytes), "cudaMalloc d_sum_x");
    check_cuda(cudaMalloc((void **)&d_sum_y, k_float_bytes), "cudaMalloc d_sum_y");
    check_cuda(cudaMalloc((void **)&d_count, k_int_bytes), "cudaMalloc d_count");

    check_cuda(cudaMemcpy(d_px, h_px, coord_bytes, cudaMemcpyHostToDevice), "H2D d_px");
    check_cuda(cudaMemcpy(d_py, h_py, coord_bytes, cudaMemcpyHostToDevice), "H2D d_py");

    int numSMs = 0;
    check_cuda(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0), "Get numSMs");

    int numBlocks = (requestedBlocks > 0) ? requestedBlocks : (32 * numSMs);
    int maxNeeded = (N + blockSize - 1) / blockSize;
    if (numBlocks > maxNeeded)
        numBlocks = maxNeeded;

    size_t shared_bytes = (size_t)blockSize * (2 * sizeof(float) + sizeof(int));

    printf("Configuracion GPU Avanzada: numSMs = %d, numBlocks = %d, blockSize = %d\n",
           numSMs, numBlocks, blockSize);

    cudaEvent_t start_total, stop_total, start_k, stop_k;
    check_cuda(cudaEventCreate(&start_total), "EventCreate start_total");
    check_cuda(cudaEventCreate(&stop_total), "EventCreate stop_total");
    check_cuda(cudaEventCreate(&start_k), "EventCreate start_k");
    check_cuda(cudaEventCreate(&stop_k), "EventCreate stop_k");

    float total_kernel_ms = 0.0f;

    check_cuda(cudaEventRecord(start_total, 0), "Record start_total");

    for (int it = 0; it < max_iters; it++)
    {
        check_cuda(cudaMemset(d_sum_x, 0, k_float_bytes), "Memset d_sum_x");
        check_cuda(cudaMemset(d_sum_y, 0, k_float_bytes), "Memset d_sum_y");
        check_cuda(cudaMemset(d_count, 0, k_int_bytes), "Memset d_count");

        check_cuda(cudaMemcpy(d_centroids, centroids, centroids_bytes, cudaMemcpyHostToDevice), "H2D d_centroids");

        check_cuda(cudaEventRecord(start_k, 0), "Record start_k");
        assign_and_accumulate_gpu<<<numBlocks, blockSize, shared_bytes>>>(
            d_px, d_py, d_centroids, d_labels, d_sum_x, d_sum_y, d_count, N, K);
        check_cuda(cudaGetLastError(), "Kernel launch assign_and_accumulate_gpu");
        check_cuda(cudaEventRecord(stop_k, 0), "Record stop_k");
        check_cuda(cudaEventSynchronize(stop_k), "Sync stop_k");

        float iter_k_ms = 0.0f;
        check_cuda(cudaEventElapsedTime(&iter_k_ms, start_k, stop_k), "Elapsed kernel");
        total_kernel_ms += iter_k_ms;

        check_cuda(cudaMemcpy(h_sum_x, d_sum_x, k_float_bytes, cudaMemcpyDeviceToHost), "D2H d_sum_x");
        check_cuda(cudaMemcpy(h_sum_y, d_sum_y, k_float_bytes, cudaMemcpyDeviceToHost), "D2H d_sum_y");
        check_cuda(cudaMemcpy(h_count, d_count, k_int_bytes, cudaMemcpyDeviceToHost), "D2H d_count");

        float movement = 0.0f;
        for (int c = 0; c < K; c++)
        {
            float newx = centroids[2 * c];
            float newy = centroids[2 * c + 1];

            if (h_count[c] > 0)
            {
                newx = h_sum_x[c] / h_count[c];
                newy = h_sum_y[c] / h_count[c];
            }

            float dx = newx - centroids[2 * c];
            float dy = newy - centroids[2 * c + 1];
            movement += dx * dx + dy * dy;

            centroids[2 * c] = newx;
            centroids[2 * c + 1] = newy;
        }

        // Exportación para GIF
        if (export_prefix)
        {
            check_cuda(cudaMemcpy(h_labels, d_labels, labels_bytes, cudaMemcpyDeviceToHost), "D2H d_labels export");
            save_labels_csv(export_prefix, it, points, h_labels, N);
            for (int c = 0; c < K; c++)
            {
                fprintf(f_cent, "%d,%d,%.6f,%.6f\n", it, c, centroids[2 * c], centroids[2 * c + 1]);
            }
        }

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

    if (f_cent) fclose(f_cent);
    cudaEventDestroy(start_total);
    cudaEventDestroy(stop_total);
    cudaEventDestroy(start_k);
    cudaEventDestroy(stop_k);
    cudaFree(d_px);
    cudaFree(d_py);
    cudaFree(d_centroids);
    cudaFree(d_labels);
    cudaFree(d_sum_x);
    cudaFree(d_sum_y);
    cudaFree(d_count);
    free(h_px);
    free(h_py);
    free(h_sum_x);
    free(h_sum_y);
    free(h_count);
    if (h_labels) free(h_labels);
}

bool verify(const float *centroids_gpu, const float *centroids_cpu, int K, float tol)
{
    float max_err = 0.0f;
    bool ok = true;

    for (int c = 0; c < K; c++)
    {
        float dx = fabsf(centroids_gpu[2 * c] - centroids_cpu[2 * c]);
        float dy = fabsf(centroids_gpu[2 * c + 1] - centroids_cpu[2 * c + 1]);
        if (dx > max_err) max_err = dx;
        if (dy > max_err) max_err = dy;

        if (dx > tol || dy > tol)
        {
            printf("Mismatch en C%d: GPU=(%f, %f) vs CPU=(%f, %f)\n",
                   c, centroids_gpu[2 * c], centroids_gpu[2 * c + 1],
                   centroids_cpu[2 * c], centroids_cpu[2 * c + 1]);
            ok = false;
        }
    }
    printf("Error absoluto maximo vs CPU: %.6e\n", max_err);
    return ok;
}

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        printf("Usage: %s data.csv [blockSize] [numBlocks] [exportPrefix|-] [K]\n", argv[0]);
        return 1;
    }

    int blockSize = 256;
    int requestedBlocks = 0;
    const char *export_prefix = NULL;
    int K = 3;

    if (argc >= 3)
    {
        blockSize = atoi(argv[2]);
        if (blockSize <= 0 || blockSize > 1024 || (blockSize & (blockSize - 1)) != 0)
        {
            printf("Error: blockSize debe ser potencia de 2 entre 1 y 1024\n");
            return 1;
        }
    }
    if (argc >= 4)
    {
        requestedBlocks = atoi(argv[3]);
        if (requestedBlocks < 0)
        {
            printf("Error: numBlocks debe ser >= 0 (0 = automatico)\n");
            return 1;
        }
    }
    if (argc >= 5 && strcmp(argv[4], "-") != 0)
    {
        export_prefix = argv[4];
    }
    if (argc >= 6)
    {
        K = atoi(argv[5]);
        if (K < 2 || K > MAX_K)
        {
            printf("Error: K debe estar entre 2 y %d\n", MAX_K);
            return 1;
        }
    }

    float *points = NULL;
    int N = load_csv(argv[1], &points);
    if (N <= 0)
        return 1;

    float centroids_gpu[MAX_K * 2];
    float centroids_cpu[MAX_K * 2];

    // Para K=3 conservamos las semillas del ejemplo
    // para otros K tomamos K puntos espaciados deterministamente del dataset.
    if (K == 3)
    {
        float init3[6] = {0, 0, 5, 5, 10, 10};
        memcpy(centroids_gpu, init3, sizeof(init3));
        memcpy(centroids_cpu, init3, sizeof(init3));
    }
    else
    {
        for (int c = 0; c < K; c++)
        {
            int idx = (int)(((long long)c * N) / K);
            centroids_gpu[2 * c]     = points[2 * idx];
            centroids_gpu[2 * c + 1] = points[2 * idx + 1];
            centroids_cpu[2 * c]     = points[2 * idx];
            centroids_cpu[2 * c + 1] = points[2 * idx + 1];
        }
    }

    printf("Loaded %d points (K = %d).\n", N, K);

    int max_iters = (K == 3) ? 10 : 50;
    float ep = 1e-4f;
    float total_ms = 0.0f, kernel_ms = 0.0f;

    kmeans_gpu_advanced(points, centroids_gpu, N, K, max_iters, ep, blockSize, requestedBlocks, export_prefix, &total_ms, &kernel_ms);

    printf("\nFinal centroids (GPU Advanced):\n");
    for (int c = 0; c < K; c++)
    {
        printf("C%d = (%f, %f)\n", c, centroids_gpu[2 * c], centroids_gpu[2 * c + 1]);
    }

    printf("\nTiempo total ciclo K-means (GPU Advanced): %.4f ms\n", total_ms);
    printf("Tiempo acumulado solo kernel (GPU Advanced): %.4f ms\n", kernel_ms);

    kmeans_cpu_ref(points, centroids_cpu, N, K, max_iters, ep);
    if (verify(centroids_gpu, centroids_cpu, K, 1e-3f))
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