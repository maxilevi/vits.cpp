
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

public:
    vits_model(struct ggml_context* ctx, std::unique_ptr<vits_model_data> model, int speaking_rate);
    ~vits_model();
    struct ggml_cgraph* build_graph(struct ggml_tensor * input_ids);
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