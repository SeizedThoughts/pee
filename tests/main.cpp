#include "../include/pee.h"

#include "test_kernel.h"

#include <iostream>

#define WINDOW_WIDTH 512
#define WINDOW_HEIGHT 512
#define WINDOW_NAME "Window"

int main(int argc, char** argv){
    pee::initOpenGL(argc, argv);
    pee::setDevice();
    pee::createWindow(WINDOW_WIDTH, WINDOW_HEIGHT, WINDOW_NAME);
    pee::createBuffers(2);
    pee::setKernel(test_kernel);
    pee::start();

    return 0;
}