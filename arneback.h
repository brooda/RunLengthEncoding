#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <math.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>


#define CUDA_CALL(x) {cudaError_t cuda_error__ = (x); if (cuda_error__) printf("CUDA error: " #x " returned \"%s\"\n", cudaGetErrorString(cuda_error__));}

__global__ void BackwardMask(char* input, int* backwardMask, int inputSize)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < inputSize) {
        if (i == 0) {
            backwardMask[i] = 1;
        } else {
            backwardMask[i] = (input[i] != input[i - 1]);
        }
    }
}

__global__ void Compact(int* scannedBackwardMask, int* compactedBackwardMask, int* totalRuns, int inputSize)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < inputSize)
    {
        if (i == (inputSize - 1))
        {
            compactedBackwardMask[scannedBackwardMask[i]] = i + 1;
            *totalRuns = scannedBackwardMask[i];
            printf("TOTAL RUNS FROM kernel: %d", *totalRuns);
        }

        if (i == 0)
        {
            compactedBackwardMask[0] = 0;
        }
        else if (scannedBackwardMask[i] != scannedBackwardMask[i - 1])
        {
            compactedBackwardMask[scannedBackwardMask[i] - 1] = i;
        }
    }
}

char* RLE_Arneback(char* input, int inputSize)
{
    // Transfer the input to GPU
    char* d_input;
    cudaMalloc((void**)&d_input, inputSize * sizeof(char));
    cudaMemcpy(d_input, input, inputSize * sizeof(char), cudaMemcpyHostToDevice);


    int blockSize = 16;
    int numberOfBlocks = inputSize / blockSize;

    if (blockSize * numberOfBlocks < inputSize)
    {
        numberOfBlocks++;
    }

    //int numberOfThreads = blockSize * numberOfBlocks;

    // Create backwardMask array
    int* h_backwardMask = new int[inputSize];
    int* d_backwardMask;
    cudaMalloc((void**)&d_backwardMask, inputSize * sizeof(int));
    cudaMemcpy(d_backwardMask, h_backwardMask, inputSize * sizeof(int), cudaMemcpyHostToDevice);

    BackwardMask<<<numberOfBlocks, blockSize>>> (d_input, d_backwardMask, inputSize);
    cudaMemcpy(h_backwardMask, d_backwardMask, inputSize * sizeof(int), cudaMemcpyDeviceToHost);

    // I will use scan (prefix sum) from Thrust library
    int* h_scannedBackwardMask = new int[inputSize];
    int* d_scannedBackwardMask;
    cudaMalloc((void**)&d_scannedBackwardMask, inputSize * sizeof(int));

    thrust::inclusive_scan(thrust::device, d_backwardMask, d_backwardMask + inputSize, d_scannedBackwardMask);
    //cudaMemcpy(h_scannedBackwardMask, d_scannedBackwardMask, inputSize * sizeof(int), cudaMemcpyDeviceToHost);


    int* h_compactedBackwardMask = new int[inputSize];
    int* d_compactedBackwardMask;
    cudaMalloc((void**)&d_compactedBackwardMask, inputSize * sizeof(int));
    cudaMemcpy(d_compactedBackwardMask, h_compactedBackwardMask, inputSize * sizeof(int), cudaMemcpyHostToDevice);

    int h_totalRuns = 0;
    int* d_totalRuns;
    cudaMalloc(&d_totalRuns, sizeof(int));

    Compact<<<numberOfBlocks, blockSize>>> (d_scannedBackwardMask, d_compactedBackwardMask, d_totalRuns, inputSize);
    cudaDeviceSynchronize();

    cudaMemcpy(&h_totalRuns, d_totalRuns, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_totalRuns);

    cudaMemcpy(h_compactedBackwardMask, d_compactedBackwardMask, h_totalRuns * sizeof(int), cudaMemcpyDeviceToHost);


    printf("Total runs: %d \n", h_totalRuns);


    for (int i=0; i<h_totalRuns; i++ )
    {
        printf("%d ",h_compactedBackwardMask[i]);
    }
/*
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
