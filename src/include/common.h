//
// Created by Maximiliano Levi on 13/09/2023.
//

#ifndef VITS_CPP_COMMON_H
#define VITS_CPP_COMMON_H
#include <fstream>
#include <stdint.h>
#include <thread>

const int MEGABYTE = 1024 * 1024;

static uint32_t read_number(std::istream& file) {
    uint32_t number;
    file.read(reinterpret_cast<char*>(&number), sizeof(uint32_t));
    return number;
}

static int get_thread_count() {
    return std::max((int)std::thread::hardware_concurrency(), 6);
}

#define ALLOC(tensor) \
    do {              \
        if (allocr)\
            ggml_allocr_alloc(allocr, tensor); \
        else\
            ASSERT(!ggml_get_no_alloc(ctx), "Failed mem initialization") \
    } while(0);

#define DEFAULT_TENSOR_TYPE GGML_TYPE_F32
#define DEFAULT_TYPE float

#endif //VITS_CPP_COMMON_H
