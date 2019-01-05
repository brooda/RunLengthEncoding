﻿#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <math.h>
// For each thread we initially are counting length of result, that will be returned by this thread.
// This is done not to use too much memory.
// Program can be speeded up when we will just write to output, that has length of input
// Inplace solution also possible - is introduced in this project.
__global__ void CountLenOfResults(char* text, int* lengths, int* endsOfChunks)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;


	int startIndex = 0;
	int endIndex = 0;

	endIndex = endsOfChunks[i];

	if (i != 0)
	{
		startIndex = endsOfChunks[i - 1] + 1;
	}

	// count length of results
	char first = text[startIndex];

	lengths[i] = 1;

	for (int j = startIndex + 1; j <= endIndex; j++)
	{
		if (text[j] != first)
		{
			first = text[j];
			lengths[i]++;
		}
	}
}


// Each threads work
// Returns:
// 		* list of letters
//		* list of count of those letters - stored as ints
__global__ void RLE(char* input, int* endsOfChunks, int* starts, int* counts, char* letters)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	int startIndex = 0;
	int endIndex = 0;

	endIndex = endsOfChunks[i];

	if (i != 0)
	{
		startIndex = endsOfChunks[i - 1] + 1;
	}

	// iterator, który będzie poruszał się po ciągu wyjściowym
	int iterator = starts[i];

	char first = input[startIndex];
	int curLength = 1;

	for (int j = startIndex + 1; j <= endIndex + 1; j++)
	{
		if (j==endIndex + 1 || input[j] != first)
		{
			counts[iterator] = curLength;
			letters[iterator] = first;

			iterator++;

			if (j != endIndex + 1)
			{
				first = input[j];
			}

			curLength = 1;
		}
		else
		{
			curLength++;
		}
	}
}


// All threads work on common memory. This function will count ends of chunks for threads.
// Thread i will work from pointer of end of chunk i-1 plus 1 to end of i.
int* EndsOfChunks(int inputSize, int numberOfThreads)
{
	// Each thread will work on such part of input (approximately).
	// Approximately, because number of elements may not be divisible by number of threads.
	// Program handles such possibility.
	int chunkSize = inputSize / numberOfThreads;

	// W tej tablicy będziemy przechowywali końcówki kawałków, którymi będą zajmowały się poszczególne wątki
	int* h_endsOfChunks = new int[numberOfThreads];

	int remains = inputSize - chunkSize * numberOfThreads;

	for (int i = 0; i < numberOfThreads - remains; i++)
	{
		h_endsOfChunks[i] = (i + 1) * chunkSize - 1;
	}

	for (int i = numberOfThreads - remains; i < numberOfThreads; i++)
	{
		h_endsOfChunks[i] = (i + 1) * chunkSize + i - (numberOfThreads - remains);
	}

	return h_endsOfChunks;
}

int power(int base, int exponent)
{
    int ret = 1;

    for (int i=0; i<exponent; i++)
    {
        ret *= base;
    }

    return ret;
}

void RLE_Parallel(char* input, int inputSize)
{
	// Transfer the input to GPU
	char* d_input;
	cudaMalloc((void**)&d_input, inputSize * sizeof(char));
	cudaMemcpy(d_input, input, inputSize * sizeof(char), cudaMemcpyHostToDevice);


	//int blockSize = 3;
	//int numberOfBlocks = 2;

	int blockSize = 256;
	int numberOfBlocks = 4;

	int numberOfThreads = blockSize * numberOfBlocks;

	int* h_endsOfChunks = EndsOfChunks(inputSize, numberOfThreads);

	// kopiujemy tablicę z indeksami końców chunków na GPU
	int* d_endsOfChunks;
	cudaMalloc((void**)&d_endsOfChunks, numberOfThreads * sizeof(int));
	cudaMemcpy(d_endsOfChunks, h_endsOfChunks, numberOfThreads * sizeof(int), cudaMemcpyHostToDevice);

	int* h_lengths = new int[numberOfThreads];
	int* d_lengths;
	cudaMalloc((void **)&d_lengths, numberOfThreads * sizeof(int));

	CountLenOfResults << <numberOfBlocks, blockSize>> > (d_input, d_lengths, d_endsOfChunks);
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


	//char* final = new char[3*len]; // We assume, that maximal count of letters in sequence is 99.
	int final_iter = 0;

	for (int i = 0; i < len; i++)
	{
		if (h_counts[i] != 0)
		{
			if (h_counts[i] == 1)
			{
				input[final_iter++] = h_letters[i];
			}
			else
			{
				char buf[10];
				sprintf(buf, "%d", h_counts[i]);

				for (int j = 0; j < strlen(buf); j++)
				{
                    input[final_iter++] = buf[j];
				}

                input[final_iter++] = h_letters[i];
			}
		}
	}

	input[final_iter] = 0;

	cudaFree(d_input);
	cudaFree(d_lengths);
	cudaFree(d_starts);
	cudaFree(d_counts);
	cudaFree(d_letters);

	//return final;
}
