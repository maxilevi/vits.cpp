
#ifndef VITS_H
#define VITS_H

#include <ggml/ggml.h>
#include <stdlib.h>
#include <string>
#include <stdint.h>
#include <vector>
#include <sstream>
#include <ggml/ggml-alloc.h>
#include "vits_model_data.h"
#include "ggml-util.h"

typedef struct ggml_tensor * tensor_t;

class vits_model {
private:
    int verbose;
    std::unique_ptr<vits_model_data> model;
    struct ggml_context * weights_ctx = nullptr;
    struct ggml_tensor * debug_tensor = nullptr;
    struct ggml_tensor * waveform = nullptr;
    struct ggml_tensor * cum_duration_output = nullptr;
    struct ggml_tensor * predicted_lengths_output = nullptr;
    struct ggml_tensor * text_encoder_output = nullptr;
    struct ggml_tensor * prior_means_output = nullptr;
    struct ggml_tensor * prior_log_variances_output = nullptr;
    struct ggml_tensor * log_duration_output = nullptr;
    struct ggml_tensor * latents_output = nullptr;
    int load_number(const std::string& key);
    float load_float(const std::string& key);
    std::string load_param(const std::string& key);
    template<typename T>
    std::vector<T> load_vector(const std::string& key) {
        this->log("Loading vector %s\n", key.c_str());
        std::string serialized_data = this->load_param(key); // Assuming this->load_param is defined somewhere in your code
        return load_vector_impl<T>(serialized_data);
    };
    template<typename T>
    std::vector<T> load_vector_impl(const std::string& serialized_data);


public:
    vits_model(struct ggml_context* ctx, std::unique_ptr<vits_model_data> model);
    ~vits_model();
    void log(const char* format, ...);
    void execute_graph(struct ggml_context* ctx, struct ggml_cgraph* graph);
    struct ggml_cgraph* build_graph_part_one(struct ggml_context* ctx, struct ggml_tensor * input_ids, struct ggml_tensor* speaker_embeddings);
    struct ggml_cgraph* build_graph_part_two(struct ggml_context* ctx, struct ggml_allocr* allocr, struct ggml_tensor* input_ids, struct ggml_tensor * cum_duration, struct ggml_tensor* prior_means, struct ggml_tensor* prior_log_variances, struct ggml_tensor* speaker_embeddings, int output_length);

    struct std::tuple<ggml_tensor*, ggml_tensor*, ggml_tensor*> text_encoder_graph(struct ggml_context* ctx, struct ggml_tensor* input_ids);
    struct ggml_tensor* wavenet_graph(struct ggml_context* ctx, struct ggml_allocr* allocr, struct ggml_tensor* input, struct ggml_tensor* speaker_embedding);
    struct ggml_tensor* flow_graph(struct ggml_context* ctx, struct ggml_allocr* allocr, struct ggml_tensor* inputs, struct ggml_tensor* conditioning, bool reverse);
    std::pair<struct ggml_tensor*, struct ggml_tensor*> flow_graph_layer(struct ggml_context* ctx, struct ggml_allocr* allocr, struct ggml_tensor* inputs, struct ggml_tensor* conditioning, bool reverse);
    struct ggml_tensor* hifigan_graph(struct ggml_context* ctx, struct ggml_allocr* allocr, struct ggml_tensor * input_ids, struct ggml_tensor* global_conditioning);
    struct ggml_tensor* dilated_depth_separable_conv_graph(struct ggml_context* ctx, struct ggml_tensor * inputs, struct ggml_tensor* global_conditioning);
    struct ggml_tensor* elementwise_affine_graph(struct ggml_context* ctx, struct ggml_tensor * inputs, struct ggml_tensor* global_conditioning, bool reverse);
    struct ggml_tensor* conv_flow_graph(struct ggml_context* ctx, struct ggml_tensor * inputs, struct ggml_tensor* global_conditioning, bool reverse);
    struct ggml_tensor* stochastic_duration_predictor_graph(struct ggml_context* ctx, struct ggml_tensor * inputs, struct ggml_tensor* speaker_embeddings, bool reverse, float noise_scale_duration);
    struct ggml_tensor* hifigan_residual_block_graph(struct ggml_context *ctx, struct ggml_tensor *hidden_states, struct ggml_tensor* buffer, int kernel_size, std::vector<int> dilation, double leaky_relu_slope);
    struct ggml_tensor* unconstrained_rational_quadratic_spline(
            struct ggml_context* ctx,
            struct ggml_tensor* inputs,
            struct ggml_tensor* unnormalized_widths,
            struct ggml_tensor* unnormalized_heights,
            struct ggml_tensor* unnormalized_derivatives,
            bool reverse = false,
            float tail_bound = 5.0,
            float min_bin_width = 1e-3,
            float min_bin_height = 1e-3,
            float min_derivative = 1e-3);
    struct ggml_tensor* rational_quadratic_spline(
            struct ggml_context* ctx,
            struct ggml_tensor* inputs,
            struct ggml_tensor* unnormalized_widths,
            struct ggml_tensor* unnormalized_heights,
            struct ggml_tensor* unnormalized_derivatives,
            bool reverse = false,
            float tail_bound = 5.0,
            float min_bin_width = 1e-3,
            float min_bin_height = 1e-3,
            float min_derivative = 1e-3);
    std::vector<float> process(std::string text);
};

#define VITS_API extern "C" __attribute__((visibility("default")))

typedef struct vits_result {
    float * data;
    size_t size;
} vits_result;

VITS_API vits_model * vits_model_load_from_bytes(const char * bytes, size_t size);

VITS_API vits_model * vits_model_load_from_file(const char * path);

VITS_API void vits_free_model(vits_model * model);

VITS_API void vits_free_result(vits_result result);

VITS_API vits_result vits_model_process(vits_model * model, const char * phonemes);

#endif