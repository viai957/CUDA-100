#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <chrono>
#include <thread>

#define IMAGESIZE 1024
#define NUM_IMAGES 1024
#define QUEUE_CAPACITY 2048
#define TARGET_TASKS NUM_IMAGES
#define THREADS_PER_BLOCK 256

struct TaskQueue
{
    int *tasks;
    volatile int head;
    volatile int tail;
    int capacity;
};

__device__ float dummyKernel(const float *image)
{
    float sum = 0.0f;
    for (int i = 0; i < IMAGESIZE; i++)
    {
        sum += image[i];
    }
    return sum;
}

__global__ void persistentKernel(const float *ImageData, float *output, TaskQueue *queue, volatile bool *doneFlag)
{
    while (!(*doneFlag) || (queue->head < queue->tail))
    {
        int currentHead = atomicAdd((int *)&queue->head, 0);
        int currentTail = atomicAdd((int *)&queue->tail, 0);

        if (currentHead < currentTail)
        {
            int taskIndex = atomicAdd((int *)&queue->head, 1);
            if (taskIndex < currentTail)
            {
                int imageIndex = queue->tasks[taskIndex];
                int imageOffset = imageIndex * IMAGESIZE;
                float result = dummyKernel(&ImageData[imageOffset]);
                output[imageIndex] = result;
            }
        }
        else
        {
            __nanosleep(100);
        }
    }
}

void addTasks(TaskQueue *queue, int startTask, int numTasks)
{
    int currentTail = queue->tail;
    for (int i = 0; i < numTasks; i++)
    {
        if (currentTail < queue->capacity)
        {
            queue->tasks[currentTail] = startTask + i;
            currentTail++;
        }
        else
        {
            std::cerr << "TaskQueue is full!" << std::endl;
            break;
        }
    }
    queue->tail = currentTail;
}

int main()
{
    size_t imageDataSize = NUM_IMAGES * IMAGESIZE * sizeof(float);
    float *h_ImageData = new float[NUM_IMAGES * IMAGESIZE];
    float *h_output = new float[NUM_IMAGES];

    for (int i = 0; i < NUM_IMAGES; i++)
    {
        for (int j = 0; j < IMAGESIZE; j++)
        {
            h_ImageData[i * IMAGESIZE + j] = static_cast<float>(rand()) / RAND_MAX;
        }
    }

    TaskQueue *queue;
    cudaMallocManaged(&queue, sizeof(TaskQueue));
    cudaMallocManaged(&queue->tasks, QUEUE_CAPACITY * sizeof(int));
    queue->head = 0;
    queue->tail = 0;
    queue->capacity = QUEUE_CAPACITY;

    bool *doneFlag;
    cudaMallocManaged(&doneFlag, sizeof(bool));
    *doneFlag = false;

    float *d_ImageData, *d_output;
    cudaMalloc(&d_ImageData, imageDataSize);
    cudaMallocManaged(&d_output, NUM_IMAGES * sizeof(float));

    cudaMemcpy(d_ImageData, h_ImageData, imageDataSize, cudaMemcpyHostToDevice);

    int blocks = (NUM_IMAGES + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    persistentKernel<<<blocks, THREADS_PER_BLOCK>>>(d_ImageData, d_output, queue, doneFlag);

    int tasksAdded = 0;
    int batchSize = 128;
    int index = 0;
    while (tasksAdded < TARGET_TASKS)
    {
        int tasksToAdd = std::min(batchSize, TARGET_TASKS - tasksAdded);
        addTasks(queue, tasksAdded, tasksToAdd);
        tasksAdded += tasksToAdd;
        std::cout << "Added " << tasksAdded << " tasks so far." << std::endl;
        while (d_output[index] == 0.0f);  
        std::cout << "output : " << d_output[index++] << std::endl;

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    while (queue->head < queue->tail)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    *doneFlag = true;

    cudaDeviceSynchronize();

    for (int i = 0; i < 5; i++)
    {
        std::cout << "Output[" << i << "] = " << d_output[i] << std::endl;
    }

    cudaFree(d_ImageData);
    cudaFree(d_output);
    cudaFree(queue->tasks);
    cudaFree(queue);
    cudaFree(doneFlag);
    delete[] h_ImageData;
    delete[] h_output;

    return 0;
}