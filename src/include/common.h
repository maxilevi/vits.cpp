//
// Created by Maximiliano Levi on 13/09/2023.
//

#ifndef VITS_CPP_COMMON_H
#define VITS_CPP_COMMON_H
#include <fstream>
#include <stdint.h>
#include <thread>

static uint32_t read_number(std::ifstream& file) {
    uint32_t number;
    file.read(reinterpret_cast<char*>(&number), sizeof(uint32_t));
    return number;
}

static int get_thread_count() {
    return std::max((int)std::thread::hardware_concurrency(), 6);
}

#define DEFAULT_TENSOR_TYPE GGML_TYPE_F32
#define DEFAULT_TYPE float

#endif //VITS_CPP_COMMON_H
