#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

char* RLE_Sequential(char* input, int inputSize)
{
	char* output;

	int outputSize = 0;
	output = (char*) malloc(outputSize * sizeof(char));

	char currentLetter = input[0];
	int counter = 1;

	for (int i = 1; i <= inputSize; i++)
	{
		if (i == inputSize || input[i] != currentLetter)
		{
			outputSize += 1;

			if (counter > 1)
			{
				char buf[10];
				sprintf(buf, "%d", counter);

				outputSize += strlen(buf);
			}

			output = (char*) realloc(output, outputSize * sizeof(char));
			 
			if (counter == 1)
			{
				output[outputSize - 1] = currentLetter;
			}
			else
			{
				char buf[10];
				sprintf(buf, "%d", counter);
				int len = strlen(buf);

				for (int i = 0; i<len; i++)
				{
					output[outputSize - 1 - len + i] = buf[i];
				}
				
				output[outputSize - 1] = currentLetter;
			}

			if (i != inputSize)
			{
				currentLetter = input[i];
			}

			counter = 0;
		}

		counter++;
	}

	output[outputSize] = 0;

	return output;
}