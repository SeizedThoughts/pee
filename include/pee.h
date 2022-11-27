#pragma once

#include <iostream>

#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glew.h>
#include <GL/freeglut.h>
#endif

#include <cuda_runtime.h>

#include <cuda_gl_interop.h>

namespace pee{
    #ifdef PEE_DEBUG

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

    GLuint pixel_buffer_object = 0;
    unsigned int current_buffer = 0;
    unsigned int buffer_count = 0;
    unsigned int width = 0;
    unsigned int height = 0;
    void (*close_func)(void) = NULL;
    void (*display_func)(void) = NULL;
    void (*reshape_func)(int, int) = NULL;
    int cuda_device_id = -1;
    cudaDeviceProp device_properties;
    void (*kernel_launcher)(const unsigned int blocks, const unsigned int threads_per_block, const unsigned int shared_memory_per_block, cudaStream_t stream, uchar4 *d_frames, const unsigned int buffer_count, const unsigned int buffer, const unsigned int width, const unsigned int height, void *pointer) = NULL;
    cudaStream_t stream = NULL;
    cudaGraphicsResource_t graphics_resource = NULL;
    unsigned int threads = 0;
    unsigned int threads_per_block = 0;
    unsigned int blocks = 0;
    int current_window = -1;
    void *m_user_pointer;

    inline void initOpenGL(int argc, char** argv){
        glutInit(&argc, argv);
        glutInitDisplayMode(GLUT_RGBA | GLUT_SINGLE);
    }

    inline int createWindow(const unsigned int desired_width, const unsigned int desired_height, const char *title){
        glutInitWindowSize(desired_width, desired_height);
        current_window = glutCreateWindow(title);
#ifndef __APPLE__
        glewInit();
#endif

        gluOrtho2D(0, 1, 1, 0);
        
        width = desired_width;
        height = desired_height;

        return current_window;
    }

    inline void setWindow(int window){
        glutSetWindow(window);
        current_window = window;
    }

    inline int getWindow(){
        return glutGetWindow();
    }

    /*
        https://www.opengl.org/resources/libraries/glut/spec3/node45.html
        https://openglut.sourceforge.net/group__windowcallback.html
    */
    inline void setCloseFunction(void (*func)(void)){
        close_func = func;

        glutCloseFunc([](){
            close_func();

            if(stream != NULL){
                cuda(StreamDestroy(stream));
            }

            if(graphics_resource != NULL){
                cuda(GraphicsUnregisterResource(graphics_resource));
            }
        });
    }

    inline void setDisplayFunction(void (*func)(void)){
        display_func = func;

        glutDisplayFunc([](){
            display_func();
            
            uchar4 *d_frames;
            cuda(GraphicsMapResources(1, &graphics_resource, stream));
            cuda(GraphicsResourceGetMappedPointer((void **)&d_frames, NULL, graphics_resource));
            kernel_launcher(blocks, threads_per_block, 0, stream, d_frames, buffer_count, current_buffer, width, height, m_user_pointer);
            cudaStreamSynchronize(stream);
            cuda(GraphicsUnmapResources(1, &graphics_resource, stream));

            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, buffer_count * height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
            glEnable(GL_TEXTURE_2D);
                glBegin(GL_QUADS);
                    glTexCoord2f(0, current_buffer/(float)buffer_count); glVertex2f(0, 0);
                    glTexCoord2f(0, (current_buffer + 1)/(float)buffer_count); glVertex2f(0, 1);
                    glTexCoord2f(1, (current_buffer + 1)/(float)buffer_count); glVertex2f(1, 1);
                    glTexCoord2f(1, current_buffer/(float)buffer_count); glVertex2f(1, 0);
                glEnd();
            glDisable(GL_TEXTURE_2D);

            glutSwapBuffers();

            current_buffer = (current_buffer + 1) % buffer_count;
        });
    }

    inline void setOverlayDisplayFunction(void (*func)(void)){
        glutOverlayDisplayFunc(func);
    }

    inline void setReshapeFunction(void (*func)(int, int)){
        reshape_func = func;

        glutReshapeFunc([](int new_width, int new_height){
            reshape_func(new_width, new_height);

            if(new_width <= 0 || new_height <= 0){
                return;
            }

            width = new_width;
            height = new_height;

            glViewport(0, 0, new_width, new_height);

            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pixel_buffer_object);
                glBufferData(GL_PIXEL_UNPACK_BUFFER, buffer_count * width * height * 4* sizeof(GLubyte), nullptr, GL_STREAM_READ);
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
            
            if(graphics_resource != NULL){
                cuda(GraphicsUnregisterResource(graphics_resource));
            }

            cuda(GraphicsGLRegisterBuffer(&graphics_resource, pixel_buffer_object, cudaGraphicsMapFlagsWriteDiscard));

            glutPostRedisplay();
        });
    }

    inline void setWindowStatusFunction(void (*func)(int state)){
        glutWindowStatusFunc(func);
    }

    inline void setKeyboardFunction(void (*func)(unsigned char key, int x, int y)){
        glutKeyboardFunc(func);
    }

    inline void setMouseFunction(void (*func)(int button, int state, int x, int y)){
        glutMouseFunc(func);
    }

    inline void setMotionFunction(void (*func)(int x, int y)){
        glutMotionFunc(func);
    }

    inline void setPassiveMotionFunction(void (*func)(int x, int y)){
        glutPassiveMotionFunc(func);
    }

    inline void setEntryFunction(void (*func)(int state)){
        glutEntryFunc(func);
    }

    inline void setSpecialFunction(void (*func)(int key, int x, int y)){
        glutSpecialFunc(func);
    }

    inline void setSpaceballMotionFunction(void (*func)(int x, int y, int z)){
        glutSpaceballMotionFunc(func);
    }

    inline void setSpaceballRotateFunction(void (*func)(int x, int y, int z)){
        glutSpaceballRotateFunc(func);
    }

    inline void setSpaceballButtonFunction(void (*func)(int button, int state)){
        glutSpaceballButtonFunc(func);
    }

    inline void setButtonBoxFunction(void (*func)(int button, int state)){
        glutButtonBoxFunc(func);
    }

    inline void setDialsFunction(void (*func)(int dial, int value)){
        glutDialsFunc(func);
    }

    inline void setTabletMotionFunction(void (*func)(int x, int y)){
        glutTabletMotionFunc(func);
    }

    inline void setTabletButtonFunction(void (*func)(int button, int state, int x, int y)){
        glutTabletButtonFunc(func);
    }

    inline void setMenuStatusFunction(void (*func)(int status, int x, int y)){
        glutMenuStatusFunc(func);
    }

    inline void setIdleFunction(void (*func)(void)){
        glutIdleFunc(func);
    }

    inline void setTimerFunction(unsigned int msecs, void (*func)(int value), int value){
        glutTimerFunc(msecs, func, value);
    }

    //CUDA core count given a GPU's device properties
    inline int getCudaCores(cudaDeviceProp devProp){  
        int cores = -1;
        int mp = devProp.multiProcessorCount;

        switch (devProp.major){
            case 2: // Fermi
                if (devProp.minor == 1){
                    cores = mp * 48;
                }else{
                    cores = mp * 32;
                }
                break;
            case 3: // Kepler
                cores = mp * 192;
                break;
            case 5: // Maxwell
                cores = mp * 128;
                break;
            case 6: // Pascal
                if((devProp.minor == 1) || (devProp.minor == 2)){
                    cores = mp * 128;
                }else if(devProp.minor == 0){
                    cores = mp * 64;
                }
                break;
            case 7: // Volta and Turing
                if((devProp.minor == 0) || (devProp.minor == 5)){
                    cores = mp * 64;
                }
                break;
            case 8: // Ampere
                if(devProp.minor == 0){
                    cores = mp * 64;
                }else if(devProp.minor == 6){
                    cores = mp * 128;
                }
                break;
        }

        return cores;
    }

    //Select the best device choice among available devices
    inline int selectDevice(){
        int deviceCount = 0;
        cuda(GetDeviceCount(&deviceCount));

        if(deviceCount == 0){
            return -1;
        }

        cudaDeviceProp props;
        int recommended_device_id = 0;
        int recommended_device_cores = 0;
        for(int i = 0; i < deviceCount; i++){
            cuda(GetDeviceProperties(&props, i));
            int cores = getCudaCores(props);
            if(recommended_device_cores > cores){
                recommended_device_cores = cores;
                recommended_device_id = i;
            }
        }

        return recommended_device_id;
    }

    inline int setDevice(const unsigned int id){
        cuda(SetDevice(id));

        cuda(GetDevice(&cuda_device_id));

        cuda(GetDeviceProperties(&device_properties, cuda_device_id));

        threads = getCudaCores(device_properties);

        if(threads == -1){
            std::cout << "No CUDA core count was found..." << std::endl;
            return -1;
        }

        threads_per_block = NULL;
        blocks = 0;
        for(int i = 0; i < sqrt(threads); i++){
            blocks++;

            if((device_properties.maxThreadsPerBlock >= (threads / blocks)) && (threads == ((threads / blocks) * blocks))){
                threads_per_block = threads / blocks;
                break;
            }
        }

        if(threads_per_block == NULL){
            std::cout << "No block count was found..." << std::endl;
            return -1;
        }

        std::cout << "CUDA device: " << device_properties.name << std::endl;

        std::cout << "CUDA cores: " << blocks * threads_per_block << std::endl;

        cuda(StreamCreate(&stream));

        return cuda_device_id;
    }

    inline int setDevice(){
        int best_device = selectDevice();

        if(best_device == -1){
            std::cout << "No CUDA devices were found..." << std::endl;
            return -1;
        }else{
            return setDevice(best_device);
        }
    }

    inline void setUserPointer(void *user_pointer){
        m_user_pointer = user_pointer;
    }

    inline void setKernel(void (*kernel)(const unsigned int blocks, const unsigned int threads_per_block, const unsigned int shared_memory_per_block, cudaStream_t stream, uchar4 *d_frames, const unsigned int buffer_count, const unsigned int buffer, const unsigned int width, const unsigned int height, void *user_pointer)){
        kernel_launcher = kernel;
    }

    inline void requestRedisplay(){
        glutPostRedisplay();
    }

    inline void createBuffers(const unsigned int count){
        buffer_count = count;

        glGenBuffers(1, &pixel_buffer_object);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, 0);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
    }

    inline void createBuffers(){
        createBuffers(2);
    }

    inline void start(){
        if(close_func == NULL){
            glutCloseFunc([](){
                close_func();

                if(stream != NULL){
                    cuda(StreamDestroy(stream));
                }

                if(graphics_resource != NULL){
                    cuda(GraphicsUnregisterResource(graphics_resource));
                }
            });
        }

        if(display_func == NULL){
            glutDisplayFunc([](){
                uchar4 *d_frames;
                cuda(GraphicsMapResources(1, &graphics_resource, stream));
                cuda(GraphicsResourceGetMappedPointer((void **)&d_frames, NULL, graphics_resource));
                kernel_launcher(blocks, threads_per_block, 0, stream, d_frames, buffer_count, current_buffer, width, height, m_user_pointer);
                cudaStreamSynchronize(stream);
                cuda(GraphicsUnmapResources(1, &graphics_resource, stream));

                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, buffer_count * height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
                glEnable(GL_TEXTURE_2D);
                    glBegin(GL_QUADS);
                        glTexCoord2f(0, current_buffer/(float)buffer_count); glVertex2f(0, 0);
                        glTexCoord2f(0, (current_buffer + 1)/(float)buffer_count); glVertex2f(0, 1);
                        glTexCoord2f(1, (current_buffer + 1)/(float)buffer_count); glVertex2f(1, 1);
                        glTexCoord2f(1, current_buffer/(float)buffer_count); glVertex2f(1, 0);
                    glEnd();
                glDisable(GL_TEXTURE_2D);

                glutSwapBuffers();

                current_buffer = (current_buffer + 1) % buffer_count;
            });
        }

        if(reshape_func == NULL){
            glutReshapeFunc([](int new_width, int new_height){
                if(new_width <= 0 || new_height <= 0){
                    return;
                }

                width = new_width;
                height = new_height;

                glViewport(0, 0, new_width, new_height);

                glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pixel_buffer_object);
                glBufferData(GL_PIXEL_UNPACK_BUFFER, buffer_count * width * height * 4* sizeof(GLubyte), nullptr, GL_STREAM_READ);
                
                if(graphics_resource != NULL){
                    cuda(GraphicsUnregisterResource(graphics_resource));
                }

                cuda(GraphicsGLRegisterBuffer(&graphics_resource, pixel_buffer_object, cudaGraphicsMapFlagsWriteDiscard));

                glutPostRedisplay();
            });
        }

        glutMainLoop();
    }

    inline void stop(){
        if(current_window != -1){
            glutDestroyWindow(current_window);
        }
    }
};