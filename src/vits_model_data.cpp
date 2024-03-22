//
// Created by Maximiliano Levi on 13/09/2023.
//

#include "include/vits_model_data.h"
#include "include/debug.h"
#include "include/vits_tokenizer.h"
#include "include/common.h"
#include <fstream>
#include <stdio.h>
#include <utility>
#include <unordered_set>
#include <regex>
#include <cstring>

template<class T> struct ggml_tensor* load_tensor(struct ggml_context* ctx, std::ifstream& file, const std::string& tensor_name, int shape_len, const std::vector<int64_t>& tensor_shape, uint32_t tensor_bytes_len, ggml_type tensor_type) {
    auto tensor = ggml_new_tensor(ctx, tensor_type, shape_len, tensor_shape.data());
    ggml_set_name(tensor, tensor_name.c_str());

    std::vector<T> tensor_data(ggml_nelements(tensor));
    file.read(reinterpret_cast<char*>(tensor_data.data()), tensor_bytes_len);

    memcpy(tensor->data, tensor_data.data(), ggml_nelements(tensor) * ggml_element_size(tensor));

    return tensor;
}

std::tuple<std::unordered_map<std::string, ggml_tensor*>, std::unordered_map<std::string, std::string>, std::unique_ptr<vits_tokenizer>> load_model_from_bytes(const char* filename, ggml_context* ctx) {
    std::unordered_map<std::string, ggml_tensor*> tensors;
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Error opening file!");
    }

    // Here read tokenizer
    auto tokenizer = vits_tokenizer::load(file);

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
        uint32_t name_len = read_number(file);

        std::vector<char> name_bytes(name_len);
        file.read(name_bytes.data(), name_len);
        std::string tensor_name(name_bytes.begin(), name_bytes.end());

        // Read tensor type
        auto tensor_type = (ggml_type)read_number(file);

        // Read tensor shape
        uint32_t shape_len = read_number(file);
        std::vector<int64_t> tensor_shape(GGML_MAX_DIMS, 1);
        for(int i = 0; i < shape_len; ++i) {
            uint32_t dim = read_number(file);
            tensor_shape[i] = (int64_t) dim;
        }

        // Read tensor byte length and load directly from file to tensor
        uint32_t tensor_bytes_len = read_number(file);

        struct ggml_tensor* tensor;
        if (tensor_type == GGML_TYPE_F32)
            tensor = load_tensor<float>(ctx, file, tensor_name, shape_len, tensor_shape, tensor_bytes_len, tensor_type);
        else if (tensor_type == GGML_TYPE_F16)
            tensor = load_tensor<ggml_fp16_t>(ctx, file, tensor_name, shape_len, tensor_shape, tensor_bytes_len, tensor_type);
        else
            throw std::runtime_error("Unsupported tensor type");

        printf("[%d/%d] Loaded tensor [%d] %s (%lu x %lu x %lu x %lu)\n", i, tensor_count, tensor_type, tensor_name.c_str(), tensor_shape[0], tensor_shape[1], tensor_shape[2], tensor_shape[3]);
        tensors[tensor_name] = tensor;
    }
    printf("Loaded %lu tensors\n", tensors.size());
    file.close();
    return std::make_tuple(tensors, config, std::move(tokenizer));
}

std::unique_ptr<vits_model_data> vits_model_data::from_file(const char* filename, ggml_context* ctx) {
    auto tensors_and_config = load_model_from_bytes(filename, ctx);

    auto [tensor_map, config, tokenizer] = std::move(tensors_and_config);
    return std::make_unique<vits_model_data>(tensor_map, config, std::move(tokenizer));
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

vits_model_data::vits_model_data(std::unordered_map<std::string, ggml_tensor*> tensor_map, std::unordered_map<std::string, std::string> config, std::unique_ptr<vits_tokenizer> tokenizer) {
    this->tensor_map = std::move(tensor_map);
    this->config = std::move(config);
    this->tokenizer = std::move(tokenizer);
}

std::unique_ptr<prefix_guard> vits_model_data::use(std::string name) {
    prefixes.push_back(name);
    return std::make_unique<prefix_guard>(prefixes);
}

struct ggml_tensor* vits_model_data::get(std::string name) const {
    auto full_name = this->current_prefix() + "." + name;
    if (tensor_map.find(full_name) == tensor_map.end()) {
        throw std::runtime_error("[ERROR] tensor not found: " + full_name);
    }
    return tensor_map.at(full_name);
}

std::string vits_model_data::current_prefix() const {
    return join(this->prefixes, ".");
}