//
// Created by Maximiliano Levi on 14/09/2023.
//

#ifndef VITS_CPP_DEBUG_H
#define VITS_CPP_DEBUG_H

#include <iostream>
#include <fstream>

#define SHAPE(tensor) \
do { \
    printf("Shape '%s' (%d):", #tensor, tensor->type); \
    for (int i = 0; i < tensor->n_dims; i++) { \
        printf(" %lld", tensor->ne[i]); \
        if(i < tensor->n_dims - 1) { \
            printf(" x"); \
        } \
    } \
    printf("\n"); \
} while (0);

#define ASSERT(x, msg) \
    do { \
        if (!(x)) { \
            fprintf(stderr, "Assertion failed: %s. Message: %s\n", #x, msg); \
            exit(EXIT_FAILURE); \
        } \
    } while(0);


#define ASSERT_SHAPE(tensor, dim0, dim1, dim2, dim3) \
    do { \
        std::vector<int64_t> expected_shape;         \
        expected_shape.push_back(dim0);              \
        expected_shape.push_back(dim1);              \
        expected_shape.push_back(dim2);              \
        expected_shape.push_back(dim3);              \
        printf("Assert shape (");                    \
        for (int i = 0; i < tensor->n_dims; ++i) { \
            printf("%d", tensor->ne[i]); \
            if (i != tensor->n_dims - 1) printf(", "); \
        } \
        printf(") == ("); \
        for (int i = 0; i < tensor->n_dims; ++i) { \
            printf("%lld", expected_shape[i]); \
            if (i != tensor->n_dims - 1) printf(", "); \
        } \
        printf(")\n");                                   \
        ASSERT(tensor->n_dims == expected_shape.size(), "Shape len mismatch"); \
        for (int i = 0; i < tensor->n_dims; ++i) { \
            ASSERT(tensor->ne[i] == expected_shape[i], "Shape mismatch"); \
        } \
} while(0);


#define PRINT_TENSOR2(tensor)                                 \
    do {                                                           \
        printf("%s (%zu, %zu, %zu, %zu): [\n", #tensor, tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);                             \
        auto ne = tensor->ne;                                      \
        auto nb0 = tensor->nb[0];                                  \
        auto nb1 = tensor->nb[1];                                  \
        auto nb2 = tensor->nb[2];                                  \
        auto data = static_cast<float*>(tensor->data);        \
        auto indent = "    ";\
                                                                   \
        for (int64_t i = 0; i < ne[2]; ++i) {                     \
            std::cout << indent << "[\n";                   \
            for (int64_t j = 0; j < ne[1]; ++j) {             \
                std::cout << indent << indent << "["; \
                for (int64_t k = 0; k < ne[0]; ++k) {              \
                    size_t offset = (k * nb0 + j * nb1 + i * nb2)  \
                                   / sizeof(float);                \
                    std::cout << std::fixed << std::setprecision(4) << *(data + offset) << " ";                   \
                }                                                  \
                std::cout << "]\n";                                 \
            }                                                 \
            std::cout << indent << "]\n";  \
        }                                                          \
        printf("]\n");                                             \
    } while (0);

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


#define MAX_PRINT_DIM 3

#define PRINT_TENSOR_PREVIEW(tensor) do { \
    ASSERT((tensor)->type == GGML_TYPE_F32, "Type must be float32"); \
    float *output_data = (float*) ggml_get_data_f32((tensor)); \
    int output_size = ggml_nelements((tensor)); \
    printf("Tensor dimension: [%d, %d, %d]\n", (tensor)->ne[0], (tensor)->ne[1], (tensor)->ne[2]); \
    printf("Data = [", output_size); \
    int preview_size = (output_size > 12) ? 6 : output_size / 2; \
    for (int i = 0; i < preview_size; ++i) { \
        printf("%f, ", output_data[i]); \
    } \
    if (output_size > 12) { \
        printf("..., "); \
        for (int i = output_size - 6; i < output_size; ++i) { \
            printf("%f, ", output_data[i]); \
        } \
    } \
    printf("]\n"); \
} while(0);

#endif //VITS_CPP_DEBUG_H
