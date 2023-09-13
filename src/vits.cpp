#include "include/vits.h"
#include <memory>
#include <stdlib.h>

vits_model::vits_model(struct ggml_context* ctx, std::unique_ptr<vits_model_data> model, int speaking_rate) {
    this->ctx = ctx;
    this->model = std::move(model);
    this->speaking_rate = speaking_rate;
}

//https://github.com/huggingface/transformers/blob/09b2de6eb74b1e5ff4f4c3d9839485f4165627c9/src/transformers/models/vits/modeling_vits.py#L1356
/*tensor_t vits_duration_predictor::process(tensor_t inputs) {

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
    return inputs * padding_mask
}

*/
std::vector<uint8_t> vits_model::process(std::string phonemes) {
/*
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
                if (config.hidden_act != "RELU") GGML_ASSERT("activation function not supported");
            }
            // Final layer norm
            {

            }
        }
    }


    //struct ggml_tensor* hidden_states = this->text_encoder->process(input_tensor);

    // Duration predictor
    {

         *        kernel_size = config.duration_predictor_kernel_size
        filter_channels = config.duration_predictor_filter_channels
         self.dropout = nn.Dropout(config.duration_predictor_dropout)
        self.conv_1 = nn.Conv1d(config.hidden_size, filter_channels, kernel_size, padding=kernel_size // 2)
        self.norm_1 = nn.LayerNorm(filter_channels, eps=config.layer_norm_eps)
        self.conv_2 = nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.norm_2 = nn.LayerNorm(filter_channels, eps=config.layer_norm_eps)
        self.proj = nn.Conv1d(filter_channels, 1, 1)

        cur = ggml_conv_1d_s1_ph();
        cur = ggml_relu(cur);
        cur = ggml_mul_mat();

        cur = ggml_conv_1d_s1_ph();
        cur = ggml_relu(cur);
        cur = ggml_mul_mat();

        cur = ggml_conv_1d_s1_ph();

    }
    //struct ggml_tensor* duration = this->duration_predictor->process(hidden_states);

    // Flow
    {
        //config.prior_encoder_num_flows)
        int flows = 1;
        for(int i = flows-1; i > -1; --i)
        {
            //inputs = torch.flip(inputs, [1])
            // VitsResidualCouplingLayer
        }
    }

    //struct ggml_tensor* latents = this->flow->process(hidden_states);
    //struct ggml_tensor* waveform = this->decoder->process(latents);
*/
    return std::vector<uint8_t>();
}

vits_model * vits_model_load_from_file(const char * path) {
    struct ggml_init_params params = {
            .mem_size   = 16*1024*1024,
            .mem_buffer = nullptr,
    };

    struct ggml_context * ctx = ggml_init(params);
    auto model_data = vits_model_data::from_file(path, ctx);
    return new vits_model(ctx, std::move(model_data), 1);
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
