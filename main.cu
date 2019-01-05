#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <time.h>

#include "sequential.h"
#include "parallel.h"
#include "parallel_inplace.h"


int Tests() {
    int *chunks = EndsOfChunks(5000000, 256);

    for (int i = 0; i < 256; i++) {
        printf("%d ", chunks[i]);
    }

    return 0;
}

void usage()
{
    printf("Required 1 input parameter: file length");
}

int main(int argc, char **argv) {
    clock_t start, end;
    double cpu_time_used;

    if (argc != 2)
    {
        usage();
        exit(0);
    }


    // Read input file to compress
    int fileLen = atoi(argv[1]);
    char* filePath = new char[30];
    sprintf(filePath, "input/text_%d.txt", fileLen);

    FILE* file = fopen(filePath, "r");
    char *input = (char *) malloc(fileLen + 1);
    fread(input, fileLen, 1, file);
    fclose(file);
    input[fileLen] = 0;

    // Read compressed file to check if my algorithm works properly
    char* compressedFilePath = new char[30];
    sprintf(compressedFilePath, "input/compressed_%d.txt", fileLen);
    FILE* fileCompressed = fopen(compressedFilePath, "rb");
    fseek(fileCompressed, 0, SEEK_END);
    long fileCompressedSize = ftell(fileCompressed);
    rewind(fileCompressed);

    char* correctOutput = (char*) malloc(fileCompressedSize + 1);
    fread(correctOutput, fileCompressedSize, 1, fileCompressed);
    fclose(fileCompressed);
    correctOutput[fileCompressedSize] = 0;

    start = clock();
    char *output1 = RLE_Sequential(input, fileLen);
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("RLE_Sequential took %f seconds to execute \n", cpu_time_used);
    printf("Compressed string is OK: %d\n", strcmp(output1, correctOutput));

    start = clock();
    RLE_Parallel(input, fileLen);
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("RLE_Parallel took %f seconds to execute \n", cpu_time_used);
    printf("Compressed string is OK: %d\n", strcmp(input, correctOutput));
    //printf("%s", input);


    //start = clock();
    //RLE_Parallel_Inplace(input, fileLen);
    //end = clock();
    //cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    //printf("\nRLE_Parallel_inplace took %f seconds to execute \n", cpu_time_used);
    //printf("Compressed string is OK: %d\n", strcmp(input, correctOutput));

    return 0;
}
