
#ifndef VITS_H
#define VITS_H

#include <ggml/ggml.h>
#include <stdlib.h>
#include <string>
#include <stdint.h>
#include <vector>
#include "vits_model_data.h"

typedef struct ggml_tensor * tensor_t;

class vits_model {
private:
    int speaking_rate;
    std::unique_ptr<vits_model_data> model;
    struct ggml_context * ctx;
    struct ggml_tensor * last_hidden_state;
    int load_number(std::string key);
    float load_float(std::string key);
    std::string load_param(std::string key);
    template<typename T>
    std::vector<T> load_vector(std::string key) {
        std::string serialized_data = this->load_param(key);
        std::vector<T> deserialized_vector;

        std::istringstream ss(serialized_data);
        std::string item;
        while (std::getline(ss, item, ',')) {  // Assuming ',' delimiter for the vector items
            if constexpr (std::is_same_v<T, int>) {
                deserialized_vector.push_back(std::stoi(item));
            } else if constexpr (std::is_same_v<T, float>) {
                deserialized_vector.push_back(std::stof(item));
            } else {
                throw std::runtime_error("Unsupported type for deserialization");
            }
        }

        return deserialized_vector;
    };

public:
    vits_model(struct ggml_context* ctx, std::unique_ptr<vits_model_data> model, int speaking_rate);
    ~vits_model();
    struct ggml_cgraph* build_graph(struct ggml_tensor * input_ids);
    struct ggml_tensor* text_encoder_graph(struct ggml_tensor* input_ids);
    struct ggml_tensor* wavenet_graph(struct ggml_tensor* input, struct ggml_tensor* speaker_embedding);
    struct ggml_tensor* flow_graph(struct ggml_context* ctx, struct ggml_tensor* inputs, struct ggml_tensor* conditioning, bool reverse);
    struct ggml_tensor* hifigan_graph(struct ggml_context* ctx, struct ggml_tensor * input_ids, struct ggml_tensor* global_conditioning);
    struct ggml_tensor* stochastic_duration_predictor_graph(struct ggml_context* ctx, struct ggml_tensor * input_ids, struct ggml_tensor* speaker_embeddings, bool reverse, float noise_scale_duration);
    struct ggml_tensor* hifigan_residual_block_graph(struct ggml_context *ctx, struct ggml_tensor *hidden_states, int kernel_size, std::vector<int> dilation, double leaky_relu_slope);
    std::vector<uint8_t> process(std::string phonemes);
};

#define VITS_API extern "C" __attribute__((visibility("default")))

typedef struct vits_result {
    uint8_t * data;
    size_t size;
} vits_result;

VITS_API vits_model * vits_model_load_from_file(const char * path);

VITS_API void vits_free_model(vits_model * model);

VITS_API void vits_free_result(vits_result result);

VITS_API vits_result vits_model_process(vits_model * model, const char * phonemes);

#endif