//
// Created by Maximiliano Levi on 13/09/2023.
//

#include "include/vits_model_data.h"
#include <fstream>
#include <stdio.h>
#include <stdint.h>
#include <utility>

static uint32_t read_number(std::ifstream& file) {
    uint32_t number;
    file.read(reinterpret_cast<char*>(&number), sizeof(uint32_t));
    return number;
}

std::pair<std::unordered_map<std::string, ggml_tensor*>, std::unordered_map<std::string, std::string>> load_model_from_bytes(const char* filename, ggml_context* ctx) {
    std::unordered_map<std::string, ggml_tensor*> tensors;
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Error opening file!");
    }

    // Read config
    std::unordered_map<std::string, std::string> config;
    uint32_t config_count = read_number(file);
    for(int i = 0; i < config_count; ++i) {
        // Read config key
        uint32_t key_len = read_number(file);

        std::vector<char> key_bytes(key_len);
        file.read(key_bytes.data(), key_len);
        std::string key(key_bytes.begin(), key_bytes.end());

        // Read config value
        uint32_t value_len = read_number(file);

        std::vector<char> value_bytes(value_len);
        file.read(value_bytes.data(), value_len);
        std::string value(value_bytes.begin(), value_bytes.end());

        config[key] = value;
    }


    uint32_t tensor_count = read_number(file);
    for(int i = 0; i < tensor_count; ++i) {
        // Read tensor name
        uint32_t name_len = read_number();

        std::vector<char> name_bytes(name_len);
        file.read(name_bytes.data(), name_len);
        std::string tensor_name(name_bytes.begin(), name_bytes.end());

        // Read tensor type
        uint32_t type_byte = read_number(file);

        // Read tensor shape
        uint32_t shape_len = read_number(file);
        std::vector<int64_t> tensor_shape(GGML_MAX_DIMS, 1);
        for(int i = 0; i < shape_len; ++i) {
            uint32_t dim = read_number(file);
            tensor_shape[i] = (int64_t) dim;
        }

        // Read tensor bytes and data
        uint32_t tensor_bytes_len = read_number(file);
        std::vector<char> tensor_bytes(tensor_bytes_len);
        file.read(tensor_bytes.data(), tensor_bytes_len);

        // Create the ggml_tensor based on shape
        tensors[tensor_name] = ggml_new_tensor(ctx, (ggml_type)type_byte, shape_len, tensor_shape.data());
        printf("[%d/%d] Loaded tensor %s (%lu x %lu x %lu x %lu)\n", i, tensor_count, tensor_name.c_str(), tensor_shape[0], tensor_shape[1], tensor_shape[2], tensor_shape[3]);
    }
    printf("Loaded %lu tensors\n", tensors.size());
    file.close();
    return std::make_pair(tensors, config);
}

std::unique_ptr<vits_model_data> vits_model_data::from_file(const char* filename, ggml_context* ctx) {
    auto tensors_and_config = load_model_from_bytes(filename, ctx);

    auto tensor_map = tensors_and_config.first;
    auto config = tensors_and_config.second;
    return std::make_unique<vits_model_data>(tensor_map, config);
}

std::string join(const std::vector<std::string>& vec, const std::string& delimiter) {
    std::string result;
    if (!vec.empty()) {
        result += vec[0];
        for (size_t i = 1; i < vec.size(); ++i) {
            result += delimiter + vec[i];
        }
    }
    return result;
}

vits_model_data::vits_model_data(std::unordered_map<std::string, ggml_tensor*> tensor_map, std::unordered_map<std::string, std::string> config) {
    this->tensor_map = std::move(tensor_map);
    this->config = std::move(config);
}

std::unique_ptr<prefix_guard> vits_model_data::use(std::string name) {
    prefixes.push_back(name);
    return std::make_unique<prefix_guard>(&prefixes);
}

struct ggml_tensor* vits_model_data::get(std::string name) const {
    auto full_prefix = join(this->prefixes, '.');
    auto name = full_prefix + "." + name;
    if (tensor_map.find(name) == tensor_map.end()) {
        throw std::runtime_error("[ERROR] tensor not found: " + name);
    }
    return tensor_map[name];
}