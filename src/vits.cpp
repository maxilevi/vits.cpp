#include "include/vits.h"
#include "include/debug.h"
#include "include/ggml-util.h"
#include <memory>
#include <thread>
#include <algorithm>
#include <stdlib.h>
#include <tuple>
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


//https://github.com/huggingface/transformers/blob/09b2de6eb74b1e5ff4f4c3d9839485f4165627c9/src/transformers/models/vits/modeling_vits.py#L1356

struct ggml_tensor* layer_norm(struct ggml_context* ctx, struct ggml_tensor* input, struct ggml_tensor* weight, struct ggml_tensor* bias, float eps) {
    auto cur = ggml_norm(ctx, input, eps);
    cur = ggml_mul(ctx, cur, weight);
    cur = ggml_add(ctx, cur, bias);
    return cur;
}

struct ggml_tensor* linear_with_bias(struct ggml_context* ctx, struct ggml_tensor* input, struct ggml_tensor* weight, struct ggml_tensor* bias) {
    auto cur = ggml_mul_mat(ctx, weight, input);
    cur = ggml_add(ctx, cur, bias);
    return cur;
}

struct ggml_tensor* conv1d(struct ggml_context* ctx, struct ggml_tensor* input, struct ggml_tensor* proj_weights, struct ggml_tensor* proj_bias, int stride = 1, int padding = 0, int dilation= 1) {
    auto proj_weights_fp16 = cast_tensor_fp32_to_fp16(ctx, proj_weights);
    auto cur = ggml_conv_1d(ctx, proj_weights_fp16, input, stride, padding, dilation);
    return cur;
}

struct ggml_tensor* conv1d_with_bias(struct ggml_context* ctx, struct ggml_tensor* input, struct ggml_tensor* proj_weights, struct ggml_tensor* proj_bias, int stride = 1, int padding = 0, int dilation= 1) {
    auto cur = conv1d(ctx, input, proj_weights, proj_bias, stride, padding, dilation);
    cur = ggml_permute(ctx, cur, 1, 0, 2, 3);
    cur = ggml_cont(ctx, cur);
    cur = ggml_add(ctx, cur, proj_bias);
    cur = ggml_permute(ctx, cur, 1, 0, 2, 3);
    cur = ggml_cont(ctx, cur);
    return cur;
}

struct ggml_tensor * get_relative_embeddings(struct ggml_context* ctx, struct ggml_tensor * relative_embeddings, int length, int window_size) {
    int pad_length = std::max(length - (window_size + 1), 0);
    struct ggml_tensor* padded_embeddings = relative_embeddings;
    if (pad_length > 0) {
        padded_embeddings = pad_3d(ctx, relative_embeddings, {0, 0, pad_length, pad_length, 0, 0});
    }

    int slice_start_position = std::max((window_size + 1) - length, 0);
    int slice_end_position = slice_start_position + 2 * length - 1;
    printf("slice_start_position = %d, slice_end_position = %d\n", slice_start_position, slice_end_position);
    return slice_3d(ctx, padded_embeddings, 0, -1, slice_start_position, slice_end_position, 0, -1);
}

struct ggml_tensor * relative_position_to_absolute_position(struct ggml_context* ctx, struct ggml_tensor * x) {
    auto sizes = x->ne;
    auto batch_heads = sizes[2];
    auto length = sizes[1];

    x = pad_3d(ctx, x, {0, 0, 0, 0, 0, 1});
    printf("length = %d, batch_heads = %d\n", length, batch_heads);

    auto x_flat = ggml_reshape_2d(ctx, x, length * 2 * length, batch_heads);
    x_flat = pad_2d(ctx, x_flat, {0, 0, 0, (int)length - 1});

    auto x_final = ggml_reshape_3d(ctx, x_flat, 2 * length - 1, length + 1, batch_heads);
    x_final = slice_3d(ctx, x_final, length - 1, -1, 0, length, 0, -1);
    return x_final;
}

struct ggml_tensor * absolute_position_to_relative_position(struct ggml_context* ctx, struct ggml_tensor * x) {
    auto sizes = x->ne;
    auto batch_heads = sizes[2];
    auto length = sizes[1];

    x = pad_3d(ctx, x, {0, 0, 0, 0, 0, (int)length - 1});
    auto x_flat = ggml_reshape_2d(ctx, x, length * length + length * (length - 1), batch_heads);
    x_flat = pad_2d(ctx, x_flat, {0, 0, (int)length, 0});

    auto x_final = ggml_reshape_3d(ctx, x_flat, 2 * length, length, batch_heads);
    x_final = slice_3d(ctx, x_final, 1, -1, 0, -1, 0, -1);

    return x_final;
}

ggml_tensor* shape_attn(struct ggml_context* ctx, struct ggml_tensor* tensor, int head_dim, int num_heads, int seq_len) {
    // bsz is always 1
    auto cur = ggml_reshape_4d(ctx, tensor, head_dim, num_heads, seq_len, 1);
    cur = ggml_permute(ctx, cur, 0, 2, 1, 3);
    return ggml_cont(ctx, cur);
}

/*
 class VitsDilatedDepthSeparableConv(nn.Module):
    def __init__(self, config: VitsConfig, dropout_rate=0.0):
        super().__init__()
        kernel_size = config.duration_predictor_kernel_size
        channels = config.hidden_size
        self.num_layers = config.depth_separable_num_layers

        self.dropout = nn.Dropout(dropout_rate)
        self.convs_dilated = nn.ModuleList()
        self.convs_pointwise = nn.ModuleList()
        self.norms_1 = nn.ModuleList()
        self.norms_2 = nn.ModuleList()
        for i in range(self.num_layers):
            dilation = kernel_size**i
            padding = (kernel_size * dilation - dilation) // 2
            self.convs_dilated.append(
                nn.Conv1d(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=kernel_size,
                    groups=channels,
                    dilation=dilation,
                    padding=padding,
                )
            )
            self.convs_pointwise.append(nn.Conv1d(channels, channels, 1))
            self.norms_1.append(nn.LayerNorm(channels))
            self.norms_2.append(nn.LayerNorm(channels))

    def forward(self, inputs, padding_mask, global_conditioning=None):
        if global_conditioning is not None:
            inputs = inputs + global_conditioning

        for i in range(self.num_layers):
            hidden_states = self.convs_dilated[i](inputs * padding_mask)
            hidden_states = self.norms_1[i](hidden_states.transpose(1, -1)).transpose(1, -1)
            hidden_states = nn.functional.gelu(hidden_states)
            hidden_states = self.convs_pointwise[i](hidden_states)
            hidden_states = self.norms_2[i](hidden_states.transpose(1, -1)).transpose(1, -1)
            hidden_states = nn.functional.gelu(hidden_states)
            hidden_states = self.dropout(hidden_states)
            inputs = inputs + hidden_states

        return inputs * padding_mask
 * */

struct std::tuple<ggml_tensor*, ggml_tensor*, ggml_tensor*> vits_model::text_encoder_graph(struct ggml_tensor* input_ids) {
    auto config = this->model->config;
    auto act_func = this->load_param("hidden_act");
    auto hidden_size = this->load_number("hidden_size");
    auto window_size = this->load_number("window_size");
    auto flow_size = this->load_number("flow_size");
    auto layer_head_mask = false;//config["layer_head_mask"];
    auto layer_count = this->load_number("num_hidden_layers");
    auto ffn_kernel_size = this->load_number("ffn_kernel_size");
    auto num_heads = this->load_number("num_attention_heads");
    auto layer_norm_eps = this->load_float("layer_norm_eps");
    auto head_dim = hidden_size / num_heads;
    auto embed_dim = hidden_size;

    struct ggml_tensor* cur = nullptr;

    auto _0 = model->use("text_encoder");

    SHAPE(input_ids);
    cur = ggml_get_rows(ctx, model->get("embed_tokens.weight"), input_ids);
    cur = ggml_scale(ctx, cur, ggml_new_f32(ctx, (float)sqrt(hidden_size)));

    for (int i = 0; i < layer_count; i++) {
        std::string base_name = "encoder.layers." + std::to_string(i);
        auto _1 = model->use(base_name);
        auto residual = cur;
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

            // 1. Project the input into query, key, and value states.
            auto query = linear_with_bias(ctx, cur, q_proj_w, q_proj_b);
            auto key = linear_with_bias(ctx, cur, k_proj_w, k_proj_b);
            auto value = linear_with_bias(ctx, cur, v_proj_w, v_proj_b);

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

            auto attn_weights = ggml_mul_mat(ctx, key_states, query_states);

            ASSERT(attn_weights->ne[2] == dim0 && attn_weights->ne[1] == src_len && attn_weights->ne[0] == src_len, "Shape is wrong");

            if (window_size > 0) {
                auto key_relative_embeddings = get_relative_embeddings(ctx, emb_rel_k, src_len, window_size);
                auto relative_logits = ggml_mul_mat(ctx, key_relative_embeddings, query_states);
                auto rel_pos_bias = relative_position_to_absolute_position(ctx, relative_logits);

                attn_weights = ggml_add(ctx, attn_weights, rel_pos_bias);
            }

            attn_weights = ggml_soft_max(ctx,attn_weights);

            // If layer head mask is defined
            if (layer_head_mask) {
                ASSERT(false, "Not implemented");
            }

            value_states = ggml_permute(ctx, value_states, 1, 0, 2, 3);
            value_states = ggml_cont(ctx, value_states);

            auto attn_output = ggml_mul_mat(ctx, value_states, attn_weights);

            ASSERT(attn_output->ne[0] == dim2 && attn_output->ne[1] == tgt_len && attn_output->ne[2] == dim0, "`attn_output` size mismatch");

            if (window_size > 0) {
                auto value_relative_embeddings = get_relative_embeddings(ctx, emb_rel_v, src_len, window_size);
                auto relative_weights = absolute_position_to_relative_position(ctx, attn_weights);
                value_relative_embeddings = ggml_permute(ctx, value_relative_embeddings, 1, 0, 2, 3);
                value_relative_embeddings = ggml_cont(ctx, value_relative_embeddings);
                auto rel_pos_bias = ggml_mul_mat(ctx, value_relative_embeddings, relative_weights);
                attn_output = ggml_add(ctx, attn_output, rel_pos_bias);
            }


            attn_output = ggml_reshape_4d(ctx, attn_output, head_dim, tgt_len, num_heads, 1);
            attn_output = ggml_permute(ctx, attn_output, 0, 2, 1, 3);
            attn_output = ggml_cont(ctx, attn_output);
            attn_output = ggml_reshape_3d(ctx, attn_output, embed_dim, tgt_len, 1);

            cur = linear_with_bias(ctx, attn_output, out_proj_w, out_proj_b);
        }

        printf("Layer norm for layer %d\n", i);

        // Layer norm
        {
            auto _ = model->use("layer_norm");
            cur = ggml_add(ctx, cur, residual);
            cur = layer_norm(ctx, cur, model->get("weight"), model->get("bias"), layer_norm_eps);

        }

        residual = cur;
        printf("FF for layer %d\n", i);
        //Feed forward
        {
            auto _ = model->use("feed_forward");
            if (act_func != "relu")
                throw std::runtime_error("activation function not supported " + act_func);

            cur = ggml_permute(ctx, cur, 1, 0, 2, 3);

            int pad_left = 0;
            int pad_right = 0;

            if (ffn_kernel_size > 1) {
                pad_left = (ffn_kernel_size - 1) / 2;
                pad_right = ffn_kernel_size / 2;
            } else {
                throw std::runtime_error("ffn_kernel_size == 1 not supported ");
            }

            cur = pad_3d(ctx, cur, {0, 0, 0, 0, pad_left, pad_right});

            cur = conv1d_with_bias(ctx, cur,  model->get("conv_1.weight"), model->get("conv_1.bias"));
            cur = ggml_relu(ctx, cur);

            cur = ggml_reshape_3d(ctx, cur, cur->ne[0], cur->ne[1], 1);
            cur = pad_3d(ctx, cur, {0, 0, 0, 0, pad_left, pad_right});

            cur = conv1d_with_bias(ctx, cur,  model->get("conv_2.weight"), model->get("conv_2.bias"));
            cur = ggml_reshape_3d(ctx, cur, cur->ne[0], cur->ne[1], 1);

            cur = ggml_permute(ctx, cur, 1, 0, 2, 3);
        }


        printf("Final layer norm for layer %d\n", i);
        // Final layer norm
        {
            auto _ = model->use("final_layer_norm");
            cur = ggml_cont(ctx, cur);
            cur = ggml_add(ctx, cur, residual);
            cur = layer_norm(ctx, cur, model->get("weight"), model->get("bias"), layer_norm_eps);
        }

    }
    printf("Calculating prior and variances\n");
    // In the future add support for returning this
    printf("Final proj for text encoder\n");
    auto text_encoder_output = cur;

    cur = ggml_permute(ctx, text_encoder_output, 1, 0, 2, 3);
    auto stats = conv1d_with_bias(ctx, cur, model->get("project.weight"), model->get("project.bias"));
    stats = ggml_permute(ctx, stats, 1, 0, 2, 3);
    stats = ggml_view_3d(ctx, stats, stats->ne[0], stats->ne[1], stats->ne[2], stats->ne[1], stats->ne[2], 0);

    auto [prior_means, prior_log_variances] = split_3d(ctx, stats, flow_size, flow_size, 0);
    printf("Finished text encoder\n");
    return std::make_tuple(text_encoder_output, prior_means, prior_log_variances);
}

struct ggml_tensor* add_tanh_sigmoid_multiply(struct ggml_context* ctx, struct ggml_tensor* a, struct ggml_tensor* b, int num_channels) {
    auto in_act = ggml_add(ctx, a, b);
    auto in_act_slice = slice_3d(ctx, in_act, 0, -1, 0, num_channels, 0, -1);
    auto tanh = ggml_tanh(ctx, in_act_slice);
    auto sigmoid = ggml_sigmoid(ctx, in_act_slice);
    auto out = ggml_mul(ctx, tanh, sigmoid);
    return out;
}

struct ggml_tensor* vits_model::wavenet_graph(struct ggml_tensor* inputs, struct ggml_tensor* global_conditioning) {
    auto num_layers = this->load_number("prior_encoder_num_wavenet_layers");
    auto hidden_size = this->load_number("hidden_size");
    auto wavenet_dilation_rate = this->load_number("wavenet_dilation_rate");
    auto wavenet_kernel_size = this->load_number("wavenet_kernel_size");
    auto speaker_embedding_size = this->load_number("speaker_embedding_size");

    auto _ = model->use("wavenet");
    auto outputs = ggml_new_tensor(ctx, inputs->type, inputs->n_dims, inputs->ne);
    ASSERT(global_conditioning == nullptr, "Not implemented");

    for (int i = 0; i < num_layers; ++i) {
        struct ggml_tensor* global_states = nullptr;
        {
            auto dilation = (int) pow(wavenet_dilation_rate, i);
            auto padding = (int) ((wavenet_kernel_size * dilation - dilation) / 2);
            auto _0 = model->use("in_layers." + std::to_string(i));
            auto hidden_states = conv1d_with_bias(ctx, inputs, this->model->get("weight"), this->model->get("bias"));

            if (global_conditioning != nullptr) {
                ASSERT(false, "Global conditioning not implemented");
            } else {
                global_states = ggml_new_tensor(ctx, inputs->type, inputs->n_dims, inputs->ne);
            }
            auto acts = add_tanh_sigmoid_multiply(ctx, hidden_states, global_states, hidden_size);

            _0 = model->use("res_skip_layers." + std::to_string(i));

            auto res_skip_acts = conv1d_with_bias(ctx, acts, this->model->get("weight"), this->model->get("bias"));
            if (i < num_layers -1) {
                auto res_skip_acts_slice = slice_3d(ctx, res_skip_acts, 0, -1, 0, hidden_size, 0, -1);
                inputs = ggml_add(ctx, inputs, res_skip_acts_slice);
                outputs = ggml_add(ctx, outputs, res_skip_acts_slice);
            } else {
                outputs = ggml_add(ctx, outputs, res_skip_acts);
            }
        }
    }
    return outputs;
}

struct ggml_tensor* vits_model::flow_graph(struct ggml_context* ctx, struct ggml_tensor* inputs, struct ggml_tensor* conditioning, bool reverse) {
    ASSERT(reverse, "Non reverse not supported");

    auto _0 = model->use("flow");
    auto num_flows = this->load_number("prior_encoder_num_flows");
    struct ggml_tensor* cur = inputs;

    if (!reverse) {
        ASSERT(false, "Non reverse not supported");
    } else {

        // flow(inputs, padding_mask, global_conditioning, reverse=True)
        auto half_channels = this->load_number("flow_size") / 2;
        for(int i = num_flows-1; i > -1; --i) {
            auto _1 = model->use("flows." + std::to_string(i));
            {

                auto [first_half, second_half] = split_3d(ctx, inputs, half_channels, half_channels, 1);
                auto hidden_states = conv1d_with_bias(ctx, first_half, this->model->get("conv_pre.weight"),
                                                      this->model->get("conv_pre.bias"));
                hidden_states = this->wavenet_graph(hidden_states, conditioning);
                auto mean = conv1d_with_bias(ctx, hidden_states, this->model->get("conv_post.weight"),
                                             this->model->get("conv_post.bias"));

                if (!reverse) {
                    ASSERT(false, "Non reverse not supported");
                } else {
                    second_half = ggml_sub(ctx, second_half, mean);
                    cur = concat_3d(ctx, first_half, second_half, 1);
                }
            }
            cur = flip_3d(ctx, cur, 1);
        }
    }
    return cur;
}


int get_padding_hifigan_residual_block(int kernel_size, int dilation=1) {
    return (int) ((kernel_size * dilation - dilation) / 2);
}

struct ggml_tensor* vits_model::hifigan_residual_block_graph(struct ggml_context *ctx, struct ggml_tensor *hidden_states, int kernel_size, std::vector<int> dilation, double leaky_relu_slope) {
    for (int i = 0; i < dilation.size(); i++) {
        auto residual = hidden_states;
        auto cur = hidden_states;
        auto _0 = model->use("convs1." + std::to_string(i));
        cur = leaky_relu(ctx, cur, leaky_relu_slope);
        cur = conv1d_with_bias(ctx, cur,
                               this->model->get("weight"),
                               this->model->get("bias"),
                               1,
                               get_padding_hifigan_residual_block(kernel_size, dilation[i]),
                               dilation[i]
        );
        _0 = model->use("convs2." + std::to_string(i));
        cur = leaky_relu(ctx, cur, leaky_relu_slope);
        cur = conv1d_with_bias(ctx, cur,
                               this->model->get("weight"),
                               this->model->get("bias"),
                               1,
                               get_padding_hifigan_residual_block(kernel_size, 1),
                               1
        );
        hidden_states = ggml_add(ctx, cur, residual);
    }
    return hidden_states;
}

struct ggml_tensor* vits_model::hifigan_graph(struct ggml_context* ctx, struct ggml_tensor * spectogram, struct ggml_tensor* global_conditioning) {
    auto _ = model->use("decoder");
    auto upsample_rates = this->load_vector<int>("upsample_rates");
    auto upsample_kernel_sizes = this->load_vector<int>("upsample_kernel_sizes");
    auto num_upsamples = upsample_rates.size();
    auto kernel_sizes = this->load_vector<int>("resblock_kernel_sizes");
    auto num_kernels = this->load_number("resblock_kernel_sizes");
    auto dilations = this->load_vector<std::vector<int>>("resblock_dilation_sizes");
    auto leaky_relu_slope = this->load_float("leaky_relu_slope");
    std::vector<std::tuple<int, std::vector<int>, double>> all_params;

    for (int i = 0; i < num_upsamples; ++i) {
        auto channels = this->load_number("upsample_initial_channel") / (int) pow(2, i + 1);
        for (int j = 0; j < kernel_sizes.size(); ++j) {
            all_params.push_back(std::make_tuple(kernel_sizes[i], dilations[i], leaky_relu_slope));
        }
    }


    auto hidden_states = conv1d_with_bias(ctx, spectogram, this->model->get("conv_pre.weight"), this->model->get("conv_pre.bias"), 1, 3);

    if (global_conditioning != nullptr) {
        ASSERT(false, "Not implemented");
    }

    for(int i = 0; i < num_upsamples; ++i)
    {
        hidden_states = leaky_relu(ctx, hidden_states, leaky_relu_slope);
        hidden_states = ggml_conv_transpose_1d(ctx, hidden_states, this->model->get("upsample_conv.weight"), upsample_rates[i], (int) ((upsample_kernel_sizes[i] - upsample_rates[i]) / 2), 1);
        hidden_states = ggml_add(ctx, hidden_states, this->model->get("upsample_conv.bias"));

        auto idx = i * num_kernels;
        const auto [kernel_size, dilation, slope] = all_params[idx];
        auto _0 = model->use("resblock_conv." + std::to_string(idx));
        auto res_state = this->hifigan_residual_block_graph(ctx, hidden_states, kernel_size, dilation, slope);

        for(auto j = 1; j < num_kernels; ++j) {
            idx = i * num_kernels + j;
            _0 = model->use("resblock_conv." + std::to_string(idx));
            auto [kernel_size, dilation, slope] = all_params[idx];
            auto block_res = this->hifigan_residual_block_graph(ctx, hidden_states, kernel_size, dilation, slope);
            res_state = ggml_add(ctx, res_state, block_res);
        }
        hidden_states = ggml_scale(ctx, res_state, ggml_new_f32(ctx, (float) (1.0 / num_kernels)));
    }

    hidden_states = leaky_relu(ctx, hidden_states, leaky_relu_slope);
    hidden_states = conv1d(ctx, hidden_states, this->model->get("conv_post.weight"), this->model->get("conv_post.bias"), 1, 3);
    auto waveform = ggml_tanh(ctx, hidden_states);
    return waveform;
}

struct ggml_tensor* vits_model::dilated_depth_separable_conv_graph(struct ggml_context* ctx, struct ggml_tensor * inputs, struct ggml_tensor* global_conditioning) {
    auto kernel_size = this->load_number("duration_predictor_kernel_size");
    auto num_layers = this->load_number("depth_separable_num_layers");

    if (global_conditioning != nullptr)
        inputs = ggml_add(ctx, inputs, global_conditioning);

    for(int i = 0; i < num_layers; ++i) {
        auto conv_dilated_i_weight = this->model->get("convs_dilated." + std::to_string(i) + ".weight");
        auto conv_dilated_i_bias = this->model->get("convs_dilated." + std::to_string(i) + ".bias");
        auto hidden_states = conv1d_with_bias(ctx, inputs, conv_dilated_i_weight, conv_dilated_i_bias);

        auto norm1_i_weight = this->model->get("norm1." + std::to_string(i) + ".weight");
        auto norm1_i_bias = this->model->get("norm1." + std::to_string(i) + ".bias");
        hidden_states = ggml_permute(ctx, hidden_states, 2, 0, 1, 3);
        hidden_states = layer_norm(ctx, hidden_states, norm1_i_weight, norm1_i_bias);
        hidden_states = ggml_permute(ctx, hidden_states, 2, 0, 1, 3);

        hidden_states = ggml_gelu(ctx, hidden_states);

        auto dilation = pow(kernel_size, i);
        auto padding = (int) ((kernel_size * dilation - dilation) / 2);
        auto conv_pointwise_i_weight = this->model->get("convs_pointwise." + std::to_string(i) + ".weight");
        auto conv_pointwise_i_bias = this->model->get("convs_pointwise." + std::to_string(i) + ".bias");
        hidden_states = conv1d_with_bias(ctx, hidden_states, conv_pointwise_i_weight, conv_pointwise_i_bias, 1, padding, dilation);

        auto norm2_i_weight = this->model->get("norm2." + std::to_string(i) + ".weight");
        auto norm2_i_bias = this->model->get("norm2." + std::to_string(i) + ".bias");
        hidden_states = ggml_permute(ctx, hidden_states, 2, 0, 1, 3);
        hidden_states = layer_norm(ctx, hidden_states, norm2_i_weight, norm2_i_bias);
        hidden_states = ggml_permute(ctx, hidden_states, 2, 0, 1, 3);

        hidden_states = ggml_gelu(ctx, hidden_states);
        inputs = ggml_add(ctx, inputs, hidden_states);
    }
    return inputs
}

    ASSERT(reverse, "Non reverse not supported");
}


struct ggml_cgraph* vits_model::build_graph(struct ggml_tensor * input_ids) {
    auto config = this->model->config;
    auto noise_scale_duration = this->load_float("noise_scale_duration");
    auto noise_scale = this->load_float("noise_scale");
    struct ggml_tensor* speaker_embeddings = nullptr;

    auto [text_encoder_output, prior_means, prior_log_variances] = this->text_encoder_graph(input_ids);
    ASSERT_SHAPE(text_encoder_output, 192, input_ids->ne[0], 1, 0);
    ASSERT_SHAPE(prior_means, 192, input_ids->ne[0], 1, 0);
    ASSERT_SHAPE(prior_log_variances, 192, input_ids->ne[0], 1, 0);
    this->text_encoder_output = text_encoder_output;
    this->prior_means_output = prior_means;
    this->prior_log_variances_output = prior_log_variances;

    auto hidden_states = text_encoder_output;
    hidden_states = ggml_permute(ctx, hidden_states, 1, 0, 2, 3);

    ASSERT(config["use_stochastic_duration_prediction"] == "True", "Only stochastic duration prediction is supported");
    // Duration predictor
    auto log_duration = this->stochastic_duration_predictor_graph(ctx, hidden_states, speaker_embeddings, true, noise_scale_duration);
    auto length_scale = ggml_new_f32(ctx, 1.0 / this->speaking_rate);
    auto duration = ggml_ceil(ctx, ggml_scale(ctx, ggml_exp(ctx, log_duration), length_scale));
    this->log_duration_output = log_duration;
    ASSERT_SHAPE(log_duration, 1, 1, input_ids->ne[0], 0);
    SHAPE(duration);
    auto predicted_lengths = clamp_min(ctx, ggml_sum(ctx, duration), 1);

    auto output_length = tensor_max(ctx, predicted_lengths);
    auto cum_duration = cumsum(ctx, duration);
    auto indices = ggml_arange(ctx, output_length);
    auto valid_indices = ggml_compare_less_than(indices, cum_duration);
    SHAPE(valid_indices);
    auto minus = slice_3d(ctx, pad_3d(ctx, valid_indices, {0, 0, 1, 0, 0, 0}), 0, -1, 0, valid_indices->ne[1]-1, 0, -1);
    auto padded_indices = ggml_sub(ctx, valid_indices, minus);
    SHAPE(padded_indices);
    struct ggml_tensor* attn = ggml_permute(ctx, padded_indices, 1, 0, 1, 2);
    ASSERT_SHAPE(attn, 1, 1, 93, input_ids->ne[0]);

    prior_means = ggml_mul_mat(ctx, attn, prior_means);
    prior_means = ggml_permute(ctx, prior_means, 0, 2, 1, 3);
    prior_log_variances = ggml_mul_mat(ctx, attn, prior_log_variances);
    prior_log_variances = ggml_permute(ctx, prior_log_variances, 0, 2, 1, 3);

    auto noise = randn_like(ctx, prior_means);
    noise = ggml_mul(ctx, noise, ggml_exp(ctx, prior_log_variances));
    noise = ggml_scale(ctx, noise, ggml_new_f32(ctx, noise_scale));

    auto prior_latents = ggml_add(ctx, prior_means, noise);
    auto latents = this->flow_graph(ctx, prior_latents, speaker_embeddings, true);
    this->latents_output = latents;
    ASSERT_SHAPE(this->latents_output, 1, 192, 93, 0);

    this->waveform = this->hifigan_graph(ctx, latents, speaker_embeddings);
    ASSERT_SHAPE(this->waveform, 1, 1, 23808, 0);

    auto gf = ggml_build_forward_ctx(ctx, this->waveform);

    printf("Finished building graph\n");

    return gf;
}

std::vector<uint8_t> vits_model::process(std::string phonemes) {
    // tokenize phonemes TODO
    auto debug_mode = true;
    std::vector<int32_t> input_ids = {0, 19,  0, 39,  0, 35,  0, 35,  0, 41,  0, 27,  0, 41,  0, 43,  0, 35, 0, 29,  0};
    auto input_ids_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, input_ids.size());
    memcpy(input_ids_tensor->data, input_ids.data(), ggml_element_size(input_ids_tensor) * input_ids.size());

    auto graph = this->build_graph(input_ids_tensor);

    printf("Allocating memory for work computation...\n");
    int threads = std::min((int)std::thread::hardware_concurrency(), 2);
    auto plan = ggml_graph_plan(graph, threads);
    if (plan.work_size > 0) {
        plan.work_data = (uint8_t*) malloc(plan.work_size);
    }

    printf("Computing...\n");
    auto start = std::chrono::high_resolution_clock::now();
    ggml_graph_compute(graph, &plan);
    auto end = std::chrono::high_resolution_clock::now();
    auto delta = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    free(plan.work_data);
    printf("Computation took %lld milliseconds\n", delta);

    if (debug_mode) {
        ASSERT_STARTS_WITH(this->text_encoder_output, 0.1938, 0.2144, 0.1059);
        ASSERT_STARTS_WITH(this->prior_means, 0.4238,  0.1439,  0.1764);
        ASSERT_STARTS_WITH(this->prior_log_variances, -0.2889, -0.0325, -0.2308);
        ASSERT_STARTS_WITH(this->latents, 0.9742,  2.0036,  1.5632);
        ASSERT_STARTS_WITH(this->log_duration, 3.1618, -0.1879,  0.7810);
        ASSERT_STARTS_WITH(this->waveform, -3.2723e-05, -1.2340e-05,  2.3337e-05);
    }

    //ASSERT(this->last_hidden_state->ne[2] == 1, "Batch size must be 1");
    //ASSERT(this->last_hidden_state->type == GGML_TYPE_F32, "Type must be float32");

    PRINT_TENSOR2(this->last_hidden_state);

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
