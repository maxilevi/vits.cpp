#include "include/vits.h"
#include <memory>
#include <stdlib.h>

//https://github.com/huggingface/transformers/blob/09b2de6eb74b1e5ff4f4c3d9839485f4165627c9/src/transformers/models/vits/modeling_vits.py#L1356
tensor_t vits_duration_predictor::process(tensor_t inputs) {
/*
    ggml_conv_1d(ctx, );
    inputs = self.conv_1(inputs * padding_mask)
    inputs = torch.relu(inputs)
    inputs = self.norm_1(inputs.transpose(1, -1)).transpose(1, -1)
    inputs = self.dropout(inputs)

    inputs = self.conv_2(inputs * padding_mask)
    inputs = torch.relu(inputs)
    inputs = self.norm_2(inputs.transpose(1, -1)).transpose(1, -1)
    inputs = self.dropout(inputs)

    inputs = self.proj(inputs * padding_mask)
    return inputs * padding_mask*/
}


std::vector<uint8_t> vits_model::process(std::string phonemes) {
    struct ggml_init_params params = {
            .mem_size   = 16*1024*1024,
            .mem_buffer = nullptr,
    };

    struct ggml_context * ctx = ggml_init(params);

    struct ggml_tensor* cur = nullptr;
    struct ggml_tensor* input_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);

    // Embeddings layer
    auto hidden_states = ggml_get_rows(ctx, model->tok_embeddings, input_ids);
    {
        for (int i = 0; i < model->num_hidden_layers; i++) {
            // Attention
            {

            }
            // Layer norm
            {
                ggml_mul_mat(ctx, model->ln_f[i], hidden_states);
            }
            //Feed forward
            {

            }
            // Final layer norm
            {

            }
        }
    }


    struct ggml_tensor* hidden_states = this->text_encoder->process(input_tensor);
    struct ggml_tensor* duration = this->duration_predictor->process(hidden_states);
    struct ggml_tensor* latents = this->flow->process(hidden_states);
    struct ggml_tensor* waveform = this->decoder->process(latents);

    return std::vector<uint8_t>();
}

vits_model * vits_model_load_from_file(const char * path) {
    return new vits_model();
}

void vits_free_model(vits_model * model) {
    delete model;
}

void vits_free_result(vits_result result) {
    delete result.data;
}

vits_result vits_model_process(vits_model * model, const char * phonemes) {
    std::vector<uint8_t> result = model->process(phonemes);
    vits_result r;
    r.data = new uint8_t[result.size()];
    r.size = result.size();
    memcpy(r.data, result.data(), result.size());
    return r;
}