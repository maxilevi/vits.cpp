//
// Created by Maximiliano Levi on 14/09/2023.
//

#ifndef VITS_CPP_DEBUG_H
#define VITS_CPP_DEBUG_H

#include <iostream>
#include <fstream>

#define SHAPE(tensor) printf("Shape '%s': %d x %d x %d x %d\n", ##tensor, tensor->shape[0], tensor->shape[1], tensor->shape[2], tensor->shape[3]);

#define SAVE_LAYER(tensor, name) \
    do { \
        std::ofstream outfile("./debug/" + name + ".txt"); \
        if(!outfile) { \
            std::cerr << "Error opening file " << name << std::endl; \
            break; \
        } \
        float* data = ggml_data_f32(tensor); \
        outfile << tensor->n_dims; \
        for(int i = 0; i < tensor->n_dims; i++) { \
            outfile << " " << tensor->ne[i]; \
        } \
        outfile << std::endl; \
        for(int i = 0; i < tensor->n_dims; i++) { \
            for(int j = 0; j < tensor->ne[i]; j++) { \
                outfile << data[i * tensor->ne[i] + j] << " "; \
            } \
            outfile << std::endl; \
        } \
        outfile.close(); \
    } while(0)

#endif //VITS_CPP_DEBUG_H
