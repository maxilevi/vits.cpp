#include "include/vits.h"
#include <memory>
#include <stdlib.h>

#define SHAPE(tensor) printf("Shape '%s': %d x %d x %d x %d\n", ##tensor, tensor->shape[0], tensor->shape[1], tensor->shape[2], tensor->shape[3]);

#define SAVE_LAYER(tensor, name) todo

vits_model::vits_model(struct ggml_context* ctx, std::unique_ptr<vits_model_data> model, int speaking_rate) {
    this->ctx = ctx;
    this->model = std::move(model);
    this->speaking_rate = speaking_rate;
}

vits_model::~vits_model() {
    printf("Free'ing vits model\n");
    ggml_free(ctx);
}

//https://github.com/huggingface/transformers/blob/09b2de6eb74b1e5ff4f4c3d9839485f4165627c9/src/transformers/models/vits/modeling_vits.py#L1356

struct ggml_tensor* linear_with_bias(struct ggml_context ctx, struct ggml_tensor* input, struct ggml_tensor* weight, struct ggml_tensor* bias) {
    auto cur = ggml_mul_mat(ctx, input, weight);
    cur = ggml_add(ctx, cur, bias);
    return cur;
}

struct ggml_tensor* conv1d_with_bias(struct ggml_context ctx, struct ggml_tensor* input, struct ggml_tensor* weight, struct ggml_tensor* bias) {
    auto cur = ggml_conv_1d(ctx, cur, proj_weights, 1, 0, 1);
    cur = ggml_add(ctx, cur, proj_bias);
    return cur;
}

struct ggml_cgraph vits_model::build_graph() {
    struct ggml_cgraph gf = {};
    struct ggml_tensor* cur = nullptr;

    auto config = this->model->config;
    auto hidden_size = std::stoi(config["hidden_size"]);
    // Text encoder
    {
        auto _0 = model->use("text_encoder");

        auto input_ids_in = this->graph_input_ids;
        cur = ggml_get_rows(ctx, model->get("embed_tokens.weight"), input_ids_in);

        for (int i = 0; i < model->config["num_hidden_layers"]; i++) {
            std::string base_name = "encoder.layers." + std::to_string(i);
            auto _1 = model->use("attention");
            // Attention
            {
                auto _ = model->use("attention");

                auto k = model->get("emb_rel_k");
                auto v = model->get("emb_rel_v");
                auto k_proj_w = model->get("k_proj.weight");
                auto k_proj_b = model->get("k_proj.bias");
                auto v_proj_w = model->get("v_proj.weight");
                auto v_proj_b = model->get("v_proj.bias");
                auto q_proj_w = model->get("q_proj.weight");
                auto q_proj_b = model->get("q_proj.bias");
                auto out_proj_w = model->get("out_proj.weight");
                auto out_proj_b = model->get("out_proj.bias");

                auto query = linear_with_bias(ctx, cur, q_proj_w, q_proj_b); // add scaling?
                auto key = linear_with_bias(ctx, cur, k_proj_w, k_proj_b);
                auto value = linear_with_bias(ctx, cur, v_proj_w, v_proj_b);

                cur = linear_with_bias(cur, out_proj_w, out_proj_b);
            }
            // Layer norm
            {
                auto _ = model->use("layer_norm");
                cur = linear_with_bias(ctx, cur, model->get("weight"), model->get("bias"));
            }
            //Feed forward
            {
                if (config.hidden_act != "RELU") GGML_ASSERT("activation function not supported");
            }
            // Final layer norm
            {
                auto _ = model->use("final_layer_norm");
                cur = linear_with_bias(ctx, cur, model->get("weight"), model->get("bias"));
            }
        }
        auto _ = model->use("project");
        cur = conv1d_with_bias(ctx, cur, model->get("weight"), model->get("bias"));
    }
    SAVE_LAYER(cur, "text_encoder");

    ggml_build_forward_expand(&gf, cur);

    return gf;
/*

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
}

std::vector<uint8_t> vits_model::process(std::string phonemes) {
    return std::vector<uint8_t>();
}

vits_model * vits_model_load_from_file(const char * path) {
    struct ggml_init_params params = {
            .mem_size   = 256*1024*1024,
            .mem_buffer = nullptr,
    };

    struct ggml_context * ctx = ggml_init(params);
    printf("Initialized ggml context with %d mb\n", params.mem_size / 1024 / 1024);
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
