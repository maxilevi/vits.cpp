//
// Created by Maximiliano Levi on 14/09/2023.
//

#ifndef VITS_CPP_DEBUG_H
#define VITS_CPP_DEBUG_H

#include <iostream>
#include <fstream>

void print_shape(const char* tensor_name, const struct ggml_tensor* tensor) {
    printf("Shape '%s':", tensor_name);
    for (int i = 0; i < tensor->n_dims; i++) {
        printf(" %lld", tensor->ne[i]);
        if(i < tensor->n_dims - 1) {
            printf(" x");
        }
    }
    printf("\n");
}

#define SHAPE(tensor) print_shape(#tensor, tensor);

#define PRINT_TENSOR(tensor) \
    do { \
        printf("Tensor '%s':\n", #tensor); \
        void* data_void = ggml_get_data(tensor); \
        if(!data_void) { \
            fprintf(stderr, "Error fetching data for tensor %s\n", #tensor); \
            continue; \
        } \
        printf("Shape: "); \
        for(int i = 0; i < tensor->n_dims; i++) { \
            printf("%d", tensor->ne[i]); \
            if(i < tensor->n_dims - 1) printf(" x "); \
        } \
        printf("\nData:\n"); \
        int total_elements = 1; \
        for(int i = 0; i < tensor->n_dims; i++) { \
            total_elements *= tensor->ne[i]; \
        } \
        if(tensor->type == GGML_TYPE_F32) { \
            float* data = (float*)data_void; \
            for(int i = 0; i < total_elements; i++) { \
                printf("%f ", data[i]); \
                if((i + 1) % tensor->ne[0] == 0) printf("\n"); \
            } \
        } \
        else if(tensor->type == GGML_TYPE_I32) { \
            int* data = (int*)data_void; \
            for(int i = 0; i < total_elements; i++) { \
                printf("%d ", data[i]); \
                if((i + 1) % tensor->ne[0] == 0) printf("\n"); \
            } \
        } \
        /* Add additional else-if blocks for other data types as needed */ \
        else { \
            fprintf(stderr, "Unsupported data type for tensor %s\n", #tensor); \
        } \
    } while(0)


#define SAVE_LAYER(tensor, name) \
    do { \
        std::ofstream outfile("./debug/" + std::string(name) + ".txt"); \
        if(!outfile) { \
            std::cerr << "Error opening file " << name << std::endl; \
            continue; \
        } \
        float* data = ggml_get_data_f32(tensor); \
        if(!data) { \
            std::cerr << "Error fetching data for tensor " << name << std::endl; \
            continue; \
        } \
        outfile << tensor->n_dims; \
        for(int i = 0; i < tensor->n_dims; i++) { \
            outfile << " " << tensor->ne[i]; \
        } \
        outfile << std::endl; \
        int total_elements = 1; \
        for(int i = 0; i < tensor->n_dims; i++) { \
            total_elements *= tensor->ne[i]; \
        } \
        for(int i = 0; i < total_elements; i++) { \
            outfile << data[i] << " "; \
            if((i + 1) % tensor->ne[0] == 0) outfile << std::endl; \
        } \
        outfile.close(); \
    } while(0)



#define ASSERT(x, msg) \
    do { \
        if (!(x)) { \
            fprintf(stderr, "Assertion failed: %s. Message: %s\n", #x, msg); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#endif //VITS_CPP_DEBUG_H
