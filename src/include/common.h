//
// Created by Maximiliano Levi on 13/09/2023.
//

#ifndef VITS_CPP_COMMON_H
#define VITS_CPP_COMMON_H
#include <fstream>
#include <stdint.h>

static uint32_t read_number(std::ifstream& file) {
    uint32_t number;
    file.read(reinterpret_cast<char*>(&number), sizeof(uint32_t));
    return number;
}

#endif //VITS_CPP_COMMON_H
