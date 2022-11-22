#pragma once

#ifdef PEE_DEBUG

#include <iostream>

#include <cuda_runtime.h>

inline cudaError_t cuda_assert(const cudaError_t code, const char* const file, const unsigned int line){
    if(code != cudaSuccess){
        std::cout << "CUDA error \"" << cudaGetErrorString(code) << "\" (" << code << ") on line " << line << " in " << file << std::endl;
        exit(code);
    }

    return code;
}

#define cuda(...) cuda_assert(cuda##__VA_ARGS__, __FILE__, __LINE__);

#else

#define cuda(...) cuda##__VA_ARGS__;

#endif