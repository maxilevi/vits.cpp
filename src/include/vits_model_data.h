//
// Created by Maximiliano Levi on 13/09/2023.
//

#ifndef VITS_CPP_VITS_MODEL_DATA_H
#define VITS_CPP_VITS_MODEL_DATA_H

#include <string>
#include <ggml/ggml.h>
#include <unordered_map>
#include <vector>
#include <memory>

struct vits_model_data {
    std::unordered_map<std::string, ggml_tensor*> tensor_map;
    std::unordered_map<std::string, std::string> config;

    static std::unique_ptr<vits_model_data> from_file(const char* filename, ggml_context* ctx);
    vits_model_data(std::unordered_map<std::string, ggml_tensor*> tensor_map, std::unordered_map<std::string, std::string> config) {
        this->tensor_map = std::move(tensor_map);
        this->config = std::move(config);
    }
};


#endif //VITS_CPP_VITS_MODEL_DATA_H
