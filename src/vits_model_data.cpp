//
// Created by Maximiliano Levi on 13/09/2023.
//

#include "include/vits_model_data.h"
#include <fstream>
#include <stdio.h>
#include <stdint.h>
#include <utility>

std::pair<std::unordered_map<std::string, ggml_tensor*>, std::unordered_map<std::string, std::string>> load_model_from_bytes(const char* filename, ggml_context* ctx) {
    std::unordered_map<std::string, ggml_tensor*> tensors;
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Error opening file!");
    }

    // Read config
    std::unordered_map<std::string, std::string> config;

    while (!file.eof()) {
        // Read tensor name
        int32_t name_len;
        file.read(reinterpret_cast<char*>(&name_len), sizeof(int32_t));
        std::vector<char> name_bytes(name_len);
        file.read(name_bytes.data(), name_len);

        // Read tensor type
        uint8_t type_byte;
        file.read(reinterpret_cast<char*>(&type_byte), sizeof(uint8_t));

        // Read tensor shape
        int32_t shape_len;
        file.read(reinterpret_cast<char*>(&shape_len), sizeof(int32_t));
        std::vector<int64_t> tensor_shape(shape_len);
        for(int i = 0; i < shape_len; ++i) {
            int32_t dim;
            file.read(reinterpret_cast<char *>(&dim), sizeof(int32_t));
            tensor_shape[i] = (int64_t) dim;
        }

        // Read tensor bytes and data
        int32_t tensor_bytes_len;
        file.read(reinterpret_cast<char*>(&tensor_bytes_len), sizeof(int32_t));
        std::vector<char> tensor_bytes(tensor_bytes_len);
        file.read(tensor_bytes.data(), tensor_bytes_len);

        // Create the ggml_tensor based on shape
        ggml_tensor* tensor = tensor = ggml_new_tensor(ctx, (ggml_type)type_byte, shape_len, tensor_shape.data());
        // tensor->fill_data(tensor_bytes.data());

        std::string tensor_name(name_bytes.begin(), name_bytes.end());
        tensors[tensor_name] = tensor;
    }

    file.close();
    return std::make_pair(tensors, config);
}

std::unique_ptr<vits_model_data> vits_model_data::from_file(const char* filename, ggml_context* ctx) {
    auto tensors_and_config = load_model_from_bytes(filename, ctx);

    auto tensor_map = tensors_and_config.first;
    auto config = tensors_and_config.second;
    return std::make_unique<vits_model_data>(tensor_map, config);
}