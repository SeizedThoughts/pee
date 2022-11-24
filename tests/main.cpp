#include "../include/pee.h"

#include "test_kernel.h"

#include <cuda_runtime.h>

#include <iostream>

#define WINDOW_WIDTH 512
#define WINDOW_HEIGHT 512
#define WINDOW_NAME "Window"

int *mode;

int main(int argc, char** argv){
    pee::initOpenGL(argc, argv);
    pee::setDevice();
    pee::createWindow(WINDOW_WIDTH, WINDOW_HEIGHT, WINDOW_NAME);
    pee::createBuffers(2);
    pee::setKernel(test_kernel);

    cudaMalloc((void **)&mode, 1 * sizeof(int));
    int start_state = 0;
    cudaMemcpy(mode, (void *)&start_state, 1 * sizeof(int), cudaMemcpyHostToDevice);

    pee::setUserPointer((void *)mode);

    pee::setKeyboardFunction([](unsigned char key, int x, int y){
        int desired_state = -1;
        switch(key){
            case 'm':
                desired_state = 1;
                break;
            case 'n':
                desired_state = 0;
                break;
        }

        if(desired_state != -1){
            std::cout << "State requested: " << desired_state << std::endl;

            cudaMemcpy(mode, (void *)&desired_state, 1 * sizeof(int), cudaMemcpyHostToDevice);

            pee::requestRedisplay();
        }
    });

    std::cout << "Press \"n\" to display a red screen." << std::endl;
    std::cout << "Press \"m\" to display the mandelbrot set." << std::endl;

    pee::start();

    return 0;
}