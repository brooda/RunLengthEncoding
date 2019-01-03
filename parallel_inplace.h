#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <math.h>
#include <cctype>

__global__ void RLE_inplace(char* input, int* endsOfChunks, int* starts, int* lengths, int* startNumbers, int* endNumbers, char* startLetters, char* endLetters)
{
	int i = threadIdx.x;

	// Ustalamy, którą częścią wejścia będzie zajmował się ten wątek
	int startIndex = 0;
	int endIndex = 0;
	endIndex = endsOfChunks[i];

	if (i != 0)
	{
		startIndex = endsOfChunks[i - 1] + 1;
	}

	starts[i] = startIndex;

	// iterator, który będzie poruszał się po ciągu wyjściowym
	int iterator = startIndex;

	char first = input[startIndex];
	int curLength = 1;

	endNumbers[i] = -1;

	int startWasSet = 0;

	for (int j = startIndex + 1; j <= endIndex + 1; j++)
	{
		if (j == endIndex + 1 || input[j] != first)
		{
			if (curLength != 1)
			{
				if (startWasSet == 0)
				{
					startNumbers[i] = curLength;
					startLetters[i] = first;

					startWasSet = 1;
				}
				else
				{
					endNumbers[i] = curLength;
					endLetters[i] = first;
				}

				int digits = 0;

				int num = curLength;
				while (num >= 1)
				{
					num /= 10;
					digits++;
				}

				for (int s = 0; s < digits; s++)
				{
					int div = 1;
					for (int k = 0; k < digits - s; k++)
					{
						div *= 10;
					}

					input[iterator++] = (curLength % div) + '0';
				}
			}

			// Wpisujemy literę
			input[iterator++] = first;

			first = input[j];

			curLength = 1;
		}
		else
		{
			curLength++;
		}
	}

	lengths[i] = iterator - startIndex;
}

void RLE_Parallel_Inplace(char* input, int inputSize)
{
	// Przenieś wejście na GPU.
	char* d_input;
	cudaMalloc((void**)&d_input, inputSize * sizeof(char));
	cudaMemcpy(d_input, input, inputSize * sizeof(char), cudaMemcpyHostToDevice);

	int numberOfThreads = 256;

	if (inputSize < 512)
	{
		numberOfThreads = 2;
	}

	int* h_endsOfChunks = EndsOfChunks(inputSize, numberOfThreads);

	// kopiujemy tablicę z indeksami końców chunków na GPU
	int* d_endsOfChunks;
	cudaMalloc((void**)&d_endsOfChunks, numberOfThreads * sizeof(int));
	cudaMemcpy(d_endsOfChunks, h_endsOfChunks, numberOfThreads * sizeof(int), cudaMemcpyHostToDevice);

	int* h_starts = new int[numberOfThreads];
	int* d_starts;
	int* h_lengths = new int[numberOfThreads];
	int* d_lengths;
	cudaMalloc((void**)&d_starts, numberOfThreads * sizeof(int));
	cudaMalloc((void**)&d_lengths, numberOfThreads * sizeof(int));

	// Tablice potrzebne do połączenia wyników wątków
	int* h_startNumbers = new int[numberOfThreads];
	int* h_endNumbers = new int[numberOfThreads];
	char* h_startLetters = new char[numberOfThreads];
	char* h_endLetters = new char[numberOfThreads];
	int* d_startNumbers;
	int* d_endNumbers;
	char* d_startLetters;
	char* d_endLetters;
	cudaMalloc((void**)&d_startNumbers, numberOfThreads * sizeof(int));
	cudaMalloc((void**)&d_endNumbers, numberOfThreads * sizeof(int));
	cudaMalloc((void**)&d_startLetters, numberOfThreads * sizeof(char));
	cudaMalloc((void**)&d_endLetters, numberOfThreads * sizeof(char));


	RLE_inplace << <1, numberOfThreads >> > (d_input, d_endsOfChunks, d_starts, d_lengths, d_startNumbers, d_endNumbers, d_startLetters, d_endLetters);
	
	cudaMemcpy(input, d_input, inputSize * sizeof(char), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_starts, d_starts, numberOfThreads * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_lengths, d_lengths, numberOfThreads * sizeof(int), cudaMemcpyDeviceToHost);

	cudaMemcpy(h_startNumbers, d_startNumbers, numberOfThreads * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_endNumbers, d_endNumbers, numberOfThreads * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_startLetters, d_startLetters, numberOfThreads * sizeof(char), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_endLetters, d_endLetters, numberOfThreads * sizeof(char), cudaMemcpyDeviceToHost);


	char* h_result;

	for (int i = 0; i < numberOfThreads; i++)
	{
		//printf("\n Start letter: %c, end letter: %c, start number: %d, end number: %d", h_startLetters[i], h_endLetters[i], h_startNumbers[i], h_endNumbers[i]);
	}


	cudaFree(d_input);
	cudaFree(d_endsOfChunks);
	cudaFree(d_starts);
	cudaFree(d_lengths);
	cudaFree(d_startNumbers);
	cudaFree(d_endNumbers);
	cudaFree(d_startLetters);
	cudaFree(d_endLetters);
}
