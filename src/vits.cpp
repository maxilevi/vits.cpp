#include "include/vits.h"
#include "debug.h"
#include "ggml-util.h"
#include <memory>
#include <thread>
#include <algorithm>
#include <stdlib.h>
#define VITS_DEBUG 1

vits_model::vits_model(struct ggml_context* ctx, std::unique_ptr<vits_model_data> model, int speaking_rate) {
    this->ctx = ctx;
    this->model = std::move(model);
    this->speaking_rate = speaking_rate;
    #if VITS_DEBUG
        printf("Config:\n");
        for(auto& [key, value] : this->model->config) {
            printf("  %s: %s\n", key.c_str(), value.c_str());
        }
    #endif
}

vits_model::~vits_model() {
    printf("Free'ing vits model\n");
    ggml_free(ctx);
}

//https://github.com/huggingface/transformers/blob/09b2de6eb74b1e5ff4f4c3d9839485f4165627c9/src/transformers/models/vits/modeling_vits.py#L1356

struct ggml_tensor* layer_norm(struct ggml_context* ctx, struct ggml_tensor* input, struct ggml_tensor* weight, struct ggml_tensor* bias, float eps) {
    auto cur = ggml_norm(ctx, input, eps);
    cur = ggml_mul_mat(ctx, cur, weight);
    cur = ggml_add(ctx, cur, bias);
    return cur;
}

struct ggml_tensor* linear_with_bias(struct ggml_context* ctx, struct ggml_tensor* input, struct ggml_tensor* weight, struct ggml_tensor* bias) {
    auto cur = ggml_mul_mat(ctx, weight, input);
    cur = ggml_add(ctx, cur, bias);
    return cur;
}

struct ggml_tensor* conv1d_with_bias(struct ggml_context* ctx, struct ggml_tensor* input, struct ggml_tensor* proj_weights, struct ggml_tensor* proj_bias) {
    auto cur = ggml_conv_1d(ctx, input, proj_weights, 1, 0, 1);
    cur = ggml_add(ctx, cur, proj_bias);
    return cur;
}

struct ggml_tensor * get_relative_embeddings(struct ggml_context* ctx, struct ggml_tensor * relative_embeddings, int length, int window_size) {
    int pad_length = std::max(length - (window_size + 1), 0);
    if (pad_length > 0) {
        SHAPE(relative_embeddings);
        relative_embeddings = pad_3d(ctx, relative_embeddings, {0, 0, pad_length, pad_length, 0, 0});
    }

    int slice_start_position = std::max((window_size + 1) - length, 0);
    int slice_end_position = slice_start_position + 2 * length - 1;
    SHAPE(relative_embeddings)
    printf("slice_start_position = %d, slice_end_position = %d\n", slice_start_position, slice_end_position);
    return slice_3d(ctx, relative_embeddings, 0, -1, slice_start_position, slice_end_position, 0, -1);
}

struct ggml_tensor * relative_position_to_absolute_position(struct ggml_context* ctx, struct ggml_tensor * x) {
    auto sizes = x->ne;
    auto batch_heads = sizes[2];
    auto length = sizes[1];

    x = pad_3d(ctx, x, {0, 0, 0, 0, 0, 1});
    printf("length = %d, batch_heads = %d\n", length, batch_heads);
    auto x_flat = ggml_reshape_2d(ctx, x, length * 2 * length, batch_heads);
    SHAPE(x_flat);
    x_flat = pad_2d(ctx, x_flat, {0, 0, 0, (int)length - 1});
    SHAPE(x_flat);

    auto x_final = ggml_reshape_3d(ctx, x_flat, 2 * length - 1, length + 1, batch_heads);
    x_final = slice_3d(ctx, x_final, length - 1, -1, 0, length, 0, -1);
    return x_final;
}

struct ggml_tensor * absolute_position_to_relative_position(struct ggml_context* ctx, struct ggml_tensor * x) {
    auto sizes = x->ne;
    auto batch_heads = sizes[2];
    auto length = sizes[1];

    x = pad_3d(ctx, x, {0, (int)length - 1, 0, 0, 0, 0});
    auto x_flat = ggml_reshape_2d(ctx, x, length * length + length * (length - 1), batch_heads);
    SHAPE(x_flat);
    x_flat = pad_2d(ctx, x_flat, {0, 0, (int)length, 0});
    auto x_final = ggml_reshape_3d(ctx, x_flat, 2 * length, length, batch_heads);
    x_final = slice_3d(ctx, x_final, 0, -1, 0, -1, 1, -1);
    return x_final;
}

std::string vits_model::load_param(std::string key) {
    auto val_str = this->model->config[key];
    if (val_str.empty())
        throw std::runtime_error("Failed to find '" + key + "' in the model's config");
    return val_str;
}

int vits_model::load_number(std::string key) {
    auto value = std::stoi(this->load_param(key));
    printf("%s = %d\n", key.c_str(), value);
    return value;
}

float vits_model::load_float(std::string key) {
    auto value = std::stof(this->load_param(key));
    printf("%s = %f\n", key.c_str(), value);
    return value;
}

ggml_tensor* shape_attn(struct ggml_context* ctx, struct ggml_tensor* tensor, int head_dim, int num_heads, int seq_len) {
    // bsz is always 1
    auto cur = ggml_reshape_4d(ctx, tensor, head_dim, num_heads, seq_len, 1);
    cur = ggml_permute(ctx, cur, 0, 2, 1, 3);
    return ggml_cont(ctx, cur);
}

struct ggml_cgraph vits_model::build_graph(struct ggml_tensor * input_ids) {
    struct ggml_cgraph gf = {};
    struct ggml_tensor* cur = nullptr;

    auto config = this->model->config;

    auto hidden_size = this->load_number("hidden_size");
    auto num_heads = this->load_number("num_attention_heads");
    auto layer_norm_eps = this->load_float("layer_norm_eps");
    auto head_dim = hidden_size / num_heads;
    auto embed_dim = hidden_size;
    // Text encoder
    {
        auto _0 = model->use("text_encoder");
        auto window_size = this->load_number("window_size");
        auto layer_head_mask = false;//config["layer_head_mask"];
        auto layer_count = this->load_number("num_hidden_layers");
        auto ffn_kernel_size = this->load_number("ffn_kernel_size");

        SHAPE(input_ids);
        cur = ggml_get_rows(ctx, model->get("embed_tokens.weight"), input_ids);
        SHAPE(cur);
        cur = ggml_scale(ctx, cur, ggml_new_f32(ctx, (float)sqrt(hidden_size)));

        for (int i = 0; i < layer_count; i++) {
            std::string base_name = "encoder.layers." + std::to_string(i);
            auto _1 = model->use(base_name);
            // Attention
            {
                printf("Building '%d'\n", i);
                auto _ = model->use("attention");

                auto emb_rel_k = model->get("emb_rel_k");
                auto emb_rel_v = model->get("emb_rel_v");
                auto k_proj_w = model->get("k_proj.weight");
                auto k_proj_b = model->get("k_proj.bias");
                auto v_proj_w = model->get("v_proj.weight");
                auto v_proj_b = model->get("v_proj.bias");
                auto q_proj_w = model->get("q_proj.weight");
                auto q_proj_b = model->get("q_proj.bias");
                auto out_proj_w = model->get("out_proj.weight");
                auto out_proj_b = model->get("out_proj.bias");

                SHAPE(cur);
                SHAPE(q_proj_w);
                SHAPE(q_proj_b);

                // 1. Project the input into query, key, and value states.
                auto query = linear_with_bias(ctx, cur, q_proj_w, q_proj_b);
                auto key = linear_with_bias(ctx, cur, k_proj_w, k_proj_b);
                auto value = linear_with_bias(ctx, cur, v_proj_w, v_proj_b);

                SHAPE(query);
                SHAPE(key);
                SHAPE(value);

                int bsz = cur->ne[2];
                int tgt_len = query->ne[1];
                int src_len = key->ne[1];
                printf("bsz = %d, tgt_len = %d, src_len = %d\n", bsz, tgt_len, src_len);
                printf("num_heads = %d, head_dim = %d\n", num_heads, head_dim);

                // Scaling the query_states (Assuming `scaling` is a float or double type variable you have)
                float scaling = pow(head_dim, -0.5);
                query = ggml_scale(ctx, query, ggml_new_f32(ctx, scaling));
                auto query_states = shape_attn(ctx, query, head_dim, num_heads, tgt_len);
                auto key_states = shape_attn(ctx, key, head_dim, num_heads, value->ne[1]);
                auto value_states = shape_attn(ctx, value, head_dim, num_heads, value->ne[1]);

                SHAPE(query_states);
                SHAPE(key_states);
                SHAPE(value_states);

                ASSERT(cur->ne[2] == 1, "Batch size must be 1");

                int dim0 = bsz * num_heads;
                int dim2 = head_dim;
                int dim1 = ggml_nelements(query) / (dim0 * dim2);
                printf("dim0 = %d, dim1 = %d, dim2 = %d\n", dim0, dim1, dim2);
                ASSERT(src_len == dim1, "Shape is wrong");

                // ggml shapes are flipped. hf shape is (bsz * self.num_heads, -1, self.head_dim)
                int target_shape[3] = {dim2, dim1, dim0};
                query_states = ggml_reshape_3d(ctx, query_states, target_shape[0], target_shape[1], target_shape[2]);

                key_states = ggml_reshape_3d(ctx, key_states, target_shape[0], target_shape[1], target_shape[2]);
                value_states = ggml_reshape_3d(ctx, value_states, target_shape[0], target_shape[1], target_shape[2]);

                SHAPE(key_states);
                SHAPE(query_states);
                auto attn_weights = ggml_mul_mat(ctx, query_states, key_states);
                SHAPE(attn_weights);

                ASSERT(attn_weights->ne[2] == dim0 && attn_weights->ne[1] == src_len && attn_weights->ne[0] == src_len, "Shape is wrong");

                if (window_size > 0) {
                    auto key_relative_embeddings = get_relative_embeddings(ctx, emb_rel_k, src_len, window_size);
                    auto relative_logits = ggml_mul_mat(ctx, key_relative_embeddings, query_states);
                    auto rel_pos_bias = relative_position_to_absolute_position(ctx, relative_logits);
                    attn_weights = ggml_add(ctx, attn_weights, rel_pos_bias);
                }

                SHAPE(attn_weights);
                attn_weights = ggml_soft_max(ctx,attn_weights);
                SHAPE(attn_weights);

                // If layer head mask is defined
                if (layer_head_mask) {
                    ASSERT(false, "Not implemented");
                }

                attn_weights = ggml_permute(ctx, attn_weights, 1, 0, 2, 3);
                value_states = ggml_permute(ctx, value_states, 1, 0, 2, 3);

                auto attn_output = batched_mul_mat(ctx, value_states, attn_weights);

                ASSERT(attn_output->ne[0] == dim2 && attn_output->ne[1] == tgt_len && attn_output->ne[2] == dim0, "`attn_output` size mismatch");

                if (window_size > 0) {
                    auto value_relative_embeddings = get_relative_embeddings(ctx, emb_rel_v, src_len, window_size);
                    auto relative_weights = absolute_position_to_relative_position(ctx, attn_weights);
                    auto rel_pos_bias = ggml_mul_mat(ctx, relative_weights, value_relative_embeddings);
                    attn_output = ggml_add(ctx, attn_output, rel_pos_bias);
                }

                attn_output = ggml_reshape_4d(ctx, attn_output, bsz, num_heads, tgt_len, head_dim);
                attn_output = ggml_permute(ctx, attn_output, 0, 2, 1, 3);
                attn_output = ggml_reshape_3d(ctx, attn_output, bsz, tgt_len, embed_dim);

                cur = linear_with_bias(ctx, attn_output, out_proj_w, out_proj_b);
            }

            // Layer norm
            {
                auto _ = model->use("layer_norm");
                cur = layer_norm(ctx, cur, model->get("weight"), model->get("bias"), layer_norm_eps);
            }

            //Feed forward
            {
                auto _ = model->use("feed_forward");
                if (config["hidden_act"] != "RELU")
                    throw std::runtime_error("activation function not supported");

                cur = ggml_permute(ctx, cur, 0, 2, 1, -1);

                cur = conv1d_with_bias(ctx, cur,  model->get("conv_1.weight"), model->get("conv_1.bias"));
                cur = ggml_relu(ctx, cur);

                cur = conv1d_with_bias(ctx, cur,  model->get("conv_2.weight"), model->get("conv_2.bias"));
                cur = ggml_relu(ctx, cur);

                cur = ggml_permute(ctx, cur, 0, 2, 1, -1);
            }

            // Final layer norm
            {
                auto _ = model->use("final_layer_norm");
                cur = layer_norm(ctx, cur, model->get("weight"), model->get("bias"), layer_norm_eps);
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
    // tokenize phonemes TODO
    std::vector<int32_t> input_ids = {0, 19,  0, 39,  0, 35,  0, 35,  0, 41,  0, 27,  0, 41,  0, 43,  0, 35, 0, 29,  0};
    auto input_ids_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, input_ids.size());
    memcpy(input_ids_tensor->data, input_ids.data(), ggml_element_size(input_ids_tensor) * input_ids.size());

    auto graph = this->build_graph(input_ids_tensor);

    int threads = std::min((int)std::thread::hardware_concurrency(), 2);
    auto plan = ggml_graph_plan(&graph, threads);
    ggml_graph_compute(&graph, &plan);

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
