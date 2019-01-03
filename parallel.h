#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

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

	for (int j = startIndex + 1; j < endIndex; j++)
	{
		if (text[j] != first)
		{
			first = text[j];
			lengths[i]++;
		}
	}
}

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

int* EndsOfChunks(int inputSize, int numberOfThreads)
{
	// Każdy z watków będzie zajmował się taką (około) częścią.
	// Około, bo rozmiar wejścia może nie być podzielny przez liczbę wątków.
	// Program jest przygotowany na taki wypadek.
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

char* RLE_Parallel(char* input, int inputSize)
{
	// Transfer the input to GPU
	char* d_input;
	cudaMalloc((void**)&d_input, inputSize * sizeof(char));
	cudaMemcpy(d_input, input, inputSize * sizeof(char), cudaMemcpyHostToDevice);

	int blockSize = 512;
	int numberOfBlocks = 64;

	int numberOfThreads = blockSize * numberOfThreads;

	int* h_endsOfChunks = EndsOfChunks(inputSize, numberOfThreads);

	// kopiujemy tablicę z indeksami końców chunków na GPU
	int* d_endsOfChunks;
	cudaMalloc((void**)&d_endsOfChunks, numberOfThreads * sizeof(int));
	cudaMemcpy(d_endsOfChunks, h_endsOfChunks, numberOfThreads * sizeof(int), cudaMemcpyHostToDevice);

	int* h_lengths = new int[numberOfThreads];
	int* d_lengths;
	cudaMalloc((void **)&d_lengths, numberOfThreads * sizeof(int));

	CountLenOfResults << <1, numberOfThreads >> > (d_input, d_lengths, d_endsOfChunks);
	cudaMemcpy(h_lengths, d_lengths, numberOfThreads * sizeof(int), cudaMemcpyDeviceToHost);

	// Są to pozycje startowe do zapisu dla poszczególnych wątków
	int* h_starts = new int[numberOfThreads];

	// Długość wyniku - ciągu skompresowanego (przed połączeniem)
	int len = 0;

	for (int i = 0; i < numberOfThreads; i++)
	{
		h_starts[i] = len;
		len += h_lengths[i];
	}

	int* d_starts;
	cudaMalloc((void **)&d_starts, numberOfThreads * sizeof(int));
	cudaMemcpy(d_starts, h_starts, numberOfThreads * sizeof(int), cudaMemcpyHostToDevice);

	// Tablica do przechowywania wyników algorytmu
	int* d_counts;
	cudaMalloc((void **)& d_counts, len * sizeof(int));	
	char* d_letters;
	cudaMalloc((void **)& d_letters, len * sizeof(char));

	if (inputSize < 512 * 4)
	{
		numberOfBlocks = 1;
		blockSize = 3;
	}

	RLE<<<numberOfBlocks, blockSize>>>(d_input, d_endsOfChunks, d_starts, d_counts, d_letters);

	int* h_counts = new int[len];
	cudaMemcpy(h_counts, d_counts, len * sizeof(int), cudaMemcpyDeviceToHost);
	char* h_letters = new char[len];
	cudaMemcpy(h_letters, d_letters, len * sizeof(char), cudaMemcpyDeviceToHost);

	char* final = new char[2 * len];

	for (int i = 1; i < numberOfThreads ; i++)
	{
		int prev = h_starts[i] - 1;
		int curr = h_starts[i];

		if (h_letters[prev] == h_letters[curr])
		{
			h_counts[prev] += h_counts[curr];
			h_counts[curr] = 0;
		}
	}

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

				for (int i = 0; i < strlen(buf); i++)
				{
					final[final_iter++] = buf[i];
				}

				final[final_iter++] = h_letters[i];
			}
		}
	}

	final[final_iter] = 0;

	// printf("%s \n", final);

	cudaFree(d_input);
	cudaFree(d_lengths);
	cudaFree(d_starts);
	cudaFree(d_counts);
	cudaFree(d_letters);

	return final;
}
