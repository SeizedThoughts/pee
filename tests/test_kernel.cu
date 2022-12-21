#include "test_kernel.h"

__device__ void draw(uchar4 *d_frames, const unsigned int buffer_count, const unsigned int buffer, const unsigned int idx, const unsigned int x, const unsigned int y, const unsigned int width, const unsigned int height, void *user_pointer){
    uchar4 *frame = &d_frames[buffer * width * height];

    //used to test past frame buffer memory
    // if(!(d_frames[idx + (width * height * ((buffer + buffer_count - 1) % buffer_count))].x == 0 && d_frames[idx + (width * height * ((buffer + buffer_count - 1) % buffer_count))].y == 0 && d_frames[idx + (width * height * ((buffer + buffer_count - 1) % buffer_count))].z == 0)){
    //     frame[idx].x = d_frames[idx + (width * height * ((buffer + buffer_count - 1) % buffer_count))].x;
    //     frame[idx].y = d_frames[idx + (width * height * ((buffer + buffer_count - 1) % buffer_count))].y;
    //     frame[idx].z = d_frames[idx + (width * height * ((buffer + buffer_count - 1) % buffer_count))].z;
    //     frame[idx].w = d_frames[idx + (width * height * ((buffer + buffer_count - 1) % buffer_count))].w;

    //     return;
    // }
    
    if(*((int*)user_pointer) == 0){
        frame[idx].x = 255;
        frame[idx].y = 0;
        frame[idx].z = 0;
        frame[idx].w = 255;

        return;
    }

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

    frame[idx].w = 255;
}

__global__ void testKernel(uchar4 *d_frames, const unsigned int buffer_count, const unsigned int buffer, unsigned int blocks, const unsigned int width, const unsigned int height, void *user_pointer){
    unsigned int idx = (blockDim.x * blockIdx.x) + threadIdx.x;

    while(idx < width * height){
        unsigned int x = idx % width;
        unsigned int y = idx / width;

        draw(d_frames, buffer_count, buffer, idx, x, y, width, height, user_pointer);

        idx += blockDim.x * blocks;
    }
}

void test_kernel(const unsigned int blocks, const unsigned int threads_per_block, const unsigned int shared_memory_per_block, cudaStream_t stream, uchar4 *d_frames, const unsigned int buffer_count, const unsigned int buffer, const unsigned int width, const unsigned int height, void *user_pointer){
    testKernel<<<blocks, threads_per_block, shared_memory_per_block, stream>>>(d_frames, buffer_count, buffer, blocks, width, height, user_pointer);
}