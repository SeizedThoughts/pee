#include "test_kernel.h"

__device__ void draw(uchar4 *frame, const unsigned int idx, const unsigned int x, const unsigned int y, const unsigned int width, const unsigned int height){
    float w = (float) width / 4;
    float h = (float) height / 4;
    float cr = x/w - 2.0;
    float ci = 2.0 - y/h;
    float i = cr;
    float j = ci;
    
    float temp = 0;
    for(int n = 0; n < 6; n++){
        temp = i*i - j*j + cr;
        j = (i*j)*2 + ci;
        i = temp;
    }
    temp = (i*i) + (j*j);
    if(temp < 4.0){
        frame[idx].x = (int) (255*temp) / 4;
    }else{
        frame[idx].x = 0;
    }
    for(int n = 0; n < 6; n++){
        temp = i*i - j*j + cr;
        j = (i*j)*2 + ci;
        i = temp;
    }
    temp = (i*i) + (j*j);
    if(temp < 4.0){
        frame[idx].y = (int) (255*temp) / 4;
    }else{
        frame[idx].y = 0;
    }
    for(int n = 0; n < 6; n++){
        temp = i*i - j*j + cr;
        j = (i*j)*2 + ci;
        i = temp;
    }
    temp = (i*i) + (j*j);
    if(temp < 4.0){
        frame[idx].z = (int) (255*temp) / 4;
    }else{
        frame[idx].z = 0;
    }
}

__global__ void testKernel(uchar4 *d_frames, const unsigned int buffer_count, const unsigned int buffer, const unsigned int width, const unsigned int height){
    uchar4 *frame = &d_frames[buffer * width * height];
    unsigned int idx = (blockDim.x * blockIdx.x) + threadIdx.x;

    while(idx < width * height){
        unsigned int x = idx % width;
        unsigned int y = idx / width;

        draw(frame, idx, x, y, width, height);

        idx += blockDim.x * blockDim.y;
    }
}

void test_kernel(const unsigned int blocks, const unsigned int threads_per_block, const unsigned int shared_memory_per_block, cudaStream_t stream, uchar4 *d_frames, const unsigned int buffer_count, const unsigned int buffer, const unsigned int width, const unsigned int height){
    testKernel<<<blocks, threads_per_block, shared_memory_per_block, stream>>>(d_frames, buffer_count, buffer, width, height);
}