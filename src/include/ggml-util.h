//
// Created by Maximiliano Levi on 08/10/2023.
//

#ifndef VITS_CPP_GGML_UTIL_H
#define VITS_CPP_GGML_UTIL_H

#include <ggml/ggml.h>
#include "debug.h"
#include <random>

struct ggml_tensor* pad_3d(struct ggml_context* ctx, struct ggml_tensor* tensor, std::vector<int> pads) {
    ASSERT(tensor->n_dims == 3, "Input tensor should be 3D");
    ASSERT(pads.size() == 6, "Invalid pad count");
    if (!ggml_is_contiguous(tensor))
        tensor = ggml_cont(ctx, tensor);

    int64_t new_shape[GGML_MAX_DIMS];
    for(int i = 0; i < tensor->n_dims; i++) {
        int reverse_index = (tensor->n_dims - i - 1) * 2;
        new_shape[i] = tensor->ne[i] + pads[reverse_index] + pads[reverse_index + 1];
    }

    auto cur = ggml_new_tensor(ctx, tensor->type, tensor->n_dims, new_shape);

    size_t nb0 = tensor->nb[0];
    size_t nb1 = nb0 * cur->ne[0];
    size_t nb2 = nb1 * cur->ne[1];
    size_t offset = (pads[0] * cur->ne[1] * cur->ne[0] + pads[2] * cur->ne[0] + pads[4]) * nb0;

    auto un_padded_view = ggml_view_3d(ctx, cur, tensor->ne[0], tensor->ne[1],tensor->ne[2], nb1, nb2, offset);
    auto cpy = ggml_cpy(ctx, tensor, un_padded_view);
    auto to_return = ggml_cpy(ctx, cur, cur);
    to_return->src[1] = cpy;

    return to_return;
}

// FIX TODO THIS IS CAN BE SIMPLIFIED ON A GENERIC PAD
struct ggml_tensor* pad_2d(struct ggml_context* ctx, struct ggml_tensor* tensor, std::vector<int> pads) {
    // Assure the correct number of pads
    ASSERT(tensor->n_dims == pads.size() / 2, "Invalid pad count");
    if (!ggml_is_contiguous(tensor))
        tensor = ggml_cont(ctx, tensor);

    int64_t new_shape[GGML_MAX_DIMS];
    for(int i = 0; i < tensor->n_dims; i++) {
        int reverse_index = (tensor->n_dims - i - 1) * 2;
        new_shape[i] = tensor->ne[i] + pads[reverse_index] + pads[reverse_index + 1];
    }

    auto cur = ggml_new_tensor(ctx, tensor->type, tensor->n_dims, new_shape);

    size_t nb0 = tensor->nb[0];
    size_t nb1 = nb0 * cur->ne[0];
    size_t offset = (pads[0] * cur->ne[0] + pads[2]) * nb0;

    // Create a 2D view on the padded tensor
    auto un_padded_view = ggml_view_2d(ctx, cur, tensor->ne[0], tensor->ne[1], nb1, offset);

    auto cpy = ggml_cpy(ctx, tensor, un_padded_view);
    auto to_return = ggml_cpy(ctx, cur, cur);
    to_return->src[1] = cpy;

    return to_return;
}

struct ggml_tensor* slice_3d(struct ggml_context* ctx, struct ggml_tensor* tensor,
                             int start0, int end0,
                             int start1, int end1,
                             int start2, int end2) {
    ASSERT(tensor->n_dims == 3, "Input tensor should be 3D");
    printf("shape %lld %lld %lld\n", tensor->ne[0], tensor->ne[1], tensor->ne[2]);
    if (!ggml_is_contiguous(tensor))
        tensor = ggml_cont(ctx, tensor);

    // Handle negative index -1 as the last index
    if (end0 == -1) end0 = tensor->ne[0];
    if (end1 == -1) end1 = tensor->ne[1];
    if (end2 == -1) end2 = tensor->ne[2];


    ASSERT(start0 >= 0 && start0 < tensor->ne[0], "Invalid start0 index");
    ASSERT(end0 > start0 && end0 <= tensor->ne[0], "Invalid end0 index");
    ASSERT(start1 >= 0 && start1 < tensor->ne[1], "Invalid start1 index");
    ASSERT(end1 > start1 && end1 <= tensor->ne[1], "Invalid end1 index");
    ASSERT(start2 >= 0 && start2 < tensor->ne[2], "Invalid start2 index");
    ASSERT(end2 > start2 && end2 <= tensor->ne[2], "Invalid end2 index");

    int64_t new_shape[3] = { end0 - start0, end1 - start1, end2 - start2 };
    printf("new_shape: (%lld, %lld, %lld)\n", new_shape[0], new_shape[1], new_shape[2]);
    size_t nb0 = tensor->nb[0];
    size_t nb1 = nb0 * new_shape[0];
    size_t nb2 = nb1 * new_shape[1];
    size_t offset = (start2 * tensor->ne[1] * tensor->ne[0] + start1 * tensor->ne[0] + start0) * nb0;

    // Create a 3D view on the original tensor
    auto sliced_view = ggml_view_3d(ctx, tensor, new_shape[0], new_shape[1], new_shape[2], nb1, nb2, offset);
    auto cur = ggml_new_tensor(ctx, tensor->type, tensor->n_dims, new_shape);
    cur = ggml_cpy(ctx, sliced_view, cur);

    return cur;
}

struct ggml_tensor* slice_2d(struct ggml_context* ctx, struct ggml_tensor* tensor, int start0, int end0, int start1, int end1) {
    return slice_3d(ctx, tensor, start0, end0, start1, end1, 0, -1);
}

struct ggml_tensor* cast_tensor_fp32_to_fp16(struct ggml_context* ctx, struct ggml_tensor* tensor) {
    ASSERT(tensor->type == GGML_TYPE_F32, "Input tensor needs to be fp32");
    struct ggml_tensor* target = ggml_new_tensor(ctx, GGML_TYPE_F16, tensor->n_dims, tensor->ne);
    auto output = ggml_cpy(ctx, tensor, target);
    ASSERT(output->type == GGML_TYPE_F16, "Output tensor needs to be fp16");
    return output;
}

struct ggml_tensor* cast_tensor_fp16_to_fp32(struct ggml_context* ctx, struct ggml_tensor* tensor) {
    ASSERT(tensor->type == GGML_TYPE_F16, "Input tensor needs to be fp16");
    struct ggml_tensor* target = ggml_new_tensor(ctx, GGML_TYPE_F32, tensor->n_dims, tensor->ne);
    auto output = ggml_cpy(ctx, tensor, target);
    ASSERT(output->type == GGML_TYPE_F32, "Output tensor needs to be fp32");
    return output;
}


std::pair<struct ggml_tensor*, struct ggml_tensor*> split_3d(struct ggml_context* ctx, struct ggml_tensor* tensor, int left, int right, int dim) {
    ASSERT(tensor->n_dims == 3, "Input tensor should be 3D");
    ASSERT(left + right == tensor->ne[dim], "Left and right should sum to the dimension size");

    if (!ggml_is_contiguous(tensor))
        tensor = ggml_cont(ctx, tensor);

    struct ggml_tensor* left_tensor;
    struct ggml_tensor* right_tensor;
    if (dim == 0) {
        left_tensor = slice_3d(ctx, tensor, 0, left, 0, -1, 0, -1);
        right_tensor = slice_3d(ctx, tensor, left, -1, 0, -1, 0, -1);
    } else if (dim == 1) {
        left_tensor = slice_3d(ctx, tensor, 0, -1, 0, left, 0, -1);
        right_tensor = slice_3d(ctx, tensor, 0, -1, left, -1, 0, -1);
    } else {
        left_tensor = slice_3d(ctx, tensor, 0, -1, 0, -1, 0, left);
        right_tensor = slice_3d(ctx, tensor, 0, -1, 0, -1, left, -1);
    }
    ASSERT(left_tensor->ne[dim] == left, "Left tensor should have the correct size");
    ASSERT(right_tensor->ne[dim] == right, "Right tensor should have the correct size");

    return std::make_pair(left_tensor, right_tensor);
}

void flip_3d_custom_op(struct ggml_tensor* dst, const struct ggml_tensor* src, int ith, int nth, void* userdata) {
    ASSERT(src->type == GGML_TYPE_F32, "Input tensor should be fp32");
    ASSERT(dst->type == GGML_TYPE_F32, "Output tensor should be fp32");
    auto* dst_ptr = (float*)dst->data;
    auto* src_ptr = (float*)src->data;
    /*
     * {1, 2, 3}
     * {4, 5, 6}
     * 0 -> 2
     * 1 -> 1
     * 3 -> 5
     * {3, 2, 1}
     * {6, 5, 4}
     * */


    size_t size = ggml_element_size(src);
    int dim_to_flip = *((int*)userdata);
    for (int i = 0; i < src->ne[0]; ++i) {
        for (int row = 0; row < src->ne[1]; ++row) {
            for (int mat = 0; mat < src->ne[2]; ++mat) {

                auto idx = (i * src->nb[0] + row * src->nb[1] + mat * src->nb[2]) / size;
                auto flipped = (src->ne[0] - i - 1);
                auto dst_idx = ((flipped) * src->nb[0] + row * src->nb[1] + mat * src->nb[2]) / size;

                printf("idx: %d flipped: %d  dst_idx: %d\n", idx, flipped, dst_idx);

                dst_ptr[idx] = src_ptr[dst_idx];
            }
        }
    }
}

struct ggml_tensor* flip_3d(struct ggml_context* ctx, struct ggml_tensor* tensor, int along) {
    ASSERT(tensor->n_dims == 3, "Input tensor should be 3D");
    ASSERT(along == 1, "Only supported on dim 1")
    return ggml_map_custom1(ctx, tensor, flip_3d_custom_op, 1, &along);
}

void sigmoid_custom_op(struct ggml_tensor* dst, const struct ggml_tensor* src, int ith, int nth, void* userdata) {
    ASSERT(src->type == GGML_TYPE_F32, "Input tensor should be fp32");
    ASSERT(dst->type == GGML_TYPE_F32, "Output tensor should be fp32");
    auto* dst_ptr = (float*)dst->data;
    auto* src_ptr = (float*)src->data;

    float value = src_ptr[ith];
    dst_ptr[ith] = 1.0f / (1.0f + std::exp(-value));
}

struct ggml_tensor* ggml_sigmoid(struct ggml_context* ctx, struct ggml_tensor* tensor) {
    return ggml_map_custom1(ctx, tensor, sigmoid_custom_op, GGML_N_TASKS_MAX, nullptr);
}


void exp_custom_op(struct ggml_tensor* dst, const struct ggml_tensor* src, int ith, int nth, void* userdata) {
    ASSERT(src->type == GGML_TYPE_F32, "Input tensor should be fp32");
    ASSERT(dst->type == GGML_TYPE_F32, "Output tensor should be fp32");
    auto* dst_ptr = (float*)dst->data;
    auto* src_ptr = (float*)src->data;

    dst_ptr[ith] = std::exp(src_ptr[nth]);
}

struct ggml_tensor* ggml_exp(struct ggml_context* ctx, struct ggml_tensor* tensor) {
    return ggml_map_custom1(ctx, tensor, exp_custom_op, GGML_N_TASKS_MAX, nullptr);
}

// 4. leaky_relu
struct LeakyReLUParams {
    double slope;
};

void leaky_relu_custom_op(struct ggml_tensor* dst, const struct ggml_tensor* src, int ith, int nth, void* userdata) {
    ASSERT(src->type == GGML_TYPE_F32, "Input tensor should be fp32");
    ASSERT(dst->type == GGML_TYPE_F32, "Output tensor should be fp32");
    auto* dst_ptr = (float*)dst->data;
    auto* src_ptr = (float*)src->data;

    auto params = (LeakyReLUParams*)userdata;
    if (src_ptr[ith] > 0) {
        dst_ptr[ith] = src_ptr[ith];
    } else {
        dst_ptr[ith] = params->slope * src_ptr[ith];
    }
}

struct ggml_tensor* leaky_relu(struct ggml_context* ctx, struct ggml_tensor* tensor, double slope) {
    LeakyReLUParams params = { slope };
    return ggml_map_custom1(ctx, tensor, leaky_relu_custom_op, tensor->ne[1], &params);
}

struct ggml_tensor* concat_3d(struct ggml_context* ctx, struct ggml_tensor* a, struct ggml_tensor* b, int dim) {
    ASSERT(a->n_dims == 3, "Input B tensor should be 3D");
    ASSERT(b->n_dims == 3, "Input A tensor should be 3D");
    ASSERT(a->ne[0] == a->ne[0], "a and b should have the same dimensions");
    ASSERT(b->ne[2] == b->ne[2], "a and b should have the same dimensions");
    ASSERT(dim == 1, "Only concat on second dimension is supported");

    int64_t new_shape[3] = { a->ne[0], a->ne[1] + b->ne[1], a->ne[2] };
    struct ggml_tensor* result = ggml_new_tensor(ctx, a->type, a->n_dims, new_shape);

    // Copy a's data
    auto view_a = ggml_view_3d(ctx, result, a->ne[0], a->ne[1], a->ne[2], a->nb[0], a->nb[1] * a->ne[0], 0);
    auto copy_a = ggml_cpy(ctx, a, view_a);

    // Copy b's data
    auto offset = ggml_element_size(a) * ggml_nelements(a);
    auto view_b = ggml_view_3d(ctx, result, b->ne[0], b->ne[1], b->ne[2], b->nb[0], b->nb[1] * b->ne[0], offset);
    auto copy_b = ggml_cpy(ctx, b, view_b);

    result = ggml_cpy(ctx, result, result);
    result->src[1] = copy_a;
    result->src[2] = copy_b;

    return result;
}

struct ggml_tensor* randn_like(struct ggml_context* ctx, struct ggml_tensor* other) {
    auto tensor = ggml_new_tensor(ctx, other->type, other->n_dims, other->ne);
    auto data = static_cast<float*>(tensor->data);
    auto size = ggml_nelements(tensor) ;
    std::mt19937 rng;
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < size; ++i) {
        data[i] = dist(rng);
    }
    return tensor;
}

struct ggml_tensor* ones_like(struct ggml_context* ctx, struct ggml_tensor* other) {
    auto tensor = ggml_new_tensor(ctx, other->type, other->n_dims, other->ne);
    auto data = static_cast<float*>(tensor->data);
    auto size = ggml_nelements(tensor) ;
    for (int i = 0; i < size; ++i) {
        data[i] = 1.0f;
    }
    return tensor;
}


#endif //VITS_CPP_GGML_UTIL_H
