#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <math.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>


__global__ void BackwardMask(char* input, char* backwardMask, int inputSize)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i == 0)
    {
        backwardMask[i] = 1;
    }
    else
    {
        backwardMask[i] = (input[i] != input[i - 1]);
    }
}

// Initially, input is backward mask
// Also, output will be written to input.
__global__ void ScanBackwardMask(char* input, int inputSize)
{
    //int i = blockIdx.x * blockDim.x + threadIdx.x;



}

/* KERNEL FUNCTIONS */
__global__ void scanKernel(int *data, const int len, int *out)
{
    const int threadCount = 1024;
    const int log_1024 = 10;

    int blockCount = gridDim.x;
    int tbid = blockIdx.x*threadCount + threadIdx.x;
    int id = threadIdx.x;
    int offset = blockIdx.x * threadCount;
    // max id 65535 * 1024 length

    //int val = data[id];
    //__syncthreads();

    for (int d = 0; d < log_1024 - 1; d++) {
        if (id % (1 << (d + 1)) == 0) {
            data[offset + id + (1 << (d + 1)) - 1] = data[offset + id + (1 << d) - 1] + data[offset + id + (1 << (d + 1)) - 1];
        }
        __syncthreads();
    }

    if (id == 0)
        data[offset + 1024 - 1] = 0;
    __syncthreads();

    for (int d = log_1024 - 1; d >= 0; d--) {
        if (id % (1 << (d + 1)) == 0) {
            int tmp = data[offset + id + (1 << d) - 1];
            data[offset + id + (1 << d) - 1] = data[offset + id + (1 << (d + 1)) - 1];
            data[offset + id + (1 << (d + 1)) - 1] = tmp + data[offset + id + (1 << (d + 1)) - 1];
        }
        __syncthreads();
    }

    //data[id] += val;
}

char* RLE_Arneback(char* input, int inputSize)
{
    // Transfer the input to GPU
    char* d_input;
    cudaMalloc((void**)&d_input, inputSize * sizeof(char));
    cudaMemcpy(d_input, input, inputSize * sizeof(char), cudaMemcpyHostToDevice);


    int blockSize = 256;
    int numberOfBlocks = inputSize / blockSize;

    if (blockSize * numberOfBlocks < inputSize)
    {
        numberOfBlocks++;
    }
    int numberOfThreads = blockSize * numberOfBlocks;

    // Create backwardMask array
    char* h_backwardMask = new char[inputSize];
    char* d_backwardMask;
    cudaMalloc((void**)&d_backwardMask, inputSize * sizeof(char));
    cudaMemcpy(d_backwardMask, h_backwardMask, inputSize * sizeof(char), cudaMemcpyHostToDevice);

    BackwardMask<<<numberOfBlocks, blockSize>>> (d_input, d_backwardMask, inputSize);
    cudaMemcpy(h_backwardMask, d_backwardMask, inputSize * sizeof(char), cudaMemcpyDeviceToHost);

    // implement own version and see how it will work
    char* h_scannedBackwardMask = new char[inputSize];
    char* d_scannedBackwardMask;
    cudaMalloc((void**)&d_scannedBackwardMask, inputSize * sizeof(char));
    cudaMemcpy(d_input, input, inputSize * sizeof(char), cudaMemcpyHostToDevice);

    //ScanBackwardMask<<<numberOfBlocks, blockSize>>> (d_input, d_backwardMask, inputSize);


    thrust::inclusive_scan(thrust::device, h_backwardMask, h_backwardMask + inputSize, h_scannedBackwardMask);

    //for (int i=0; i<inputSize; i++)
    //{
    //    printf("%d ", scannedBackwardMask[i]);
    //}

    /*
    cudaMemcpy(h_lengths, d_lengths, numberOfThreads * sizeof(int), cudaMemcpyDeviceToHost);

    // Start positions to write for all threads
    int* h_starts = new int[numberOfThreads];

    // Length of sequence (before merging results from threads)
    int len = 0;

    for (int i = 0; i < numberOfThreads; i++)
    {
        h_starts[i] = len;
        len += h_lengths[i];
    }

    int* d_starts;
    cudaMalloc((void **)&d_starts, numberOfThreads * sizeof(int));
    cudaMemcpy(d_starts, h_starts, numberOfThreads * sizeof(int), cudaMemcpyHostToDevice);

    // Arrays to keep results of the algorithm: letters and their counts
    int* d_counts;
    cudaMalloc((void **)& d_counts, len * sizeof(int));
    char* d_letters;
    cudaMalloc((void **)& d_letters, len * sizeof(char));

    RLE<<<numberOfBlocks, blockSize>>>(d_input, d_endsOfChunks, d_starts, d_counts, d_letters);

    int* h_counts = new int[len];
    cudaMemcpy(h_counts, d_counts, len * sizeof(int), cudaMemcpyDeviceToHost);
    char* h_letters = new char[len];
    cudaMemcpy(h_letters, d_letters, len * sizeof(char), cudaMemcpyDeviceToHost);

    // MERGING PHASE
    int levelOfMerges = 0;

    int mult = 1;
    while(mult < numberOfThreads)
    {
        mult *=2;
        levelOfMerges++;
    }

    for (int i=0; i< levelOfMerges; i++)
    {
        int step = power(2, i+1);

        for (int j = power(2, i); j<numberOfThreads; j += step)
        {
            int startIndOfj = h_starts[j];

            if (h_letters[startIndOfj-1] == h_letters[startIndOfj])
            {
                h_counts[startIndOfj] += h_counts[startIndOfj - 1];
                h_counts[startIndOfj - 1] = 0;
            }
        }
    }


    char* final = new char[3*len]; // We assume, that maximal count of letters in sequence is 99.
    int final_iter = 0;

    for (int i = 0; i < len; i++)
    {
        if (h_counts[i] != 0)
        {
            if (h_counts[i] == 1)
            {
                final[final_iter++] = h_letters[i];
            }
            else
            {
                char buf[10];
                sprintf(buf, "%d", h_counts[i]);

                for (int j = 0; j < strlen(buf); j++)
                {
                    final[final_iter++] = buf[j];
                }

                final[final_iter++] = h_letters[i];
            }
        }
    }

    final[final_iter] = 0;

    cudaFree(d_input);
    cudaFree(d_lengths);
    cudaFree(d_starts);
    cudaFree(d_counts);
    cudaFree(d_letters);
*/
    return new char[10];
}
