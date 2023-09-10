
#ifndef VITS_H
#define VITS_H

#include <ggml/ggml.h>
#include <stdlib.h>
#include <string>
#include <stdint.h>
#include <vector>

typedef struct ggml_tensor * tensor_t;

struct vits_model_weights {
    struct ggml_tensor* encoder_embeddings;
    struct ggml_tensor* encoder_embeddings;
};

class vits_model {
private:
    int speaking_rate;
    vits_model_weights * model;

public:

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