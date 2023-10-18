//
// Created by Maximiliano Levi on 08/10/2023.
//

#ifndef VITS_CPP_GGML_UTIL_H
#define VITS_CPP_GGML_UTIL_H

#include <ggml/ggml.h>
#include "debug.h"

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

    size_t nb0 = tensor->nb[0];
    size_t nb1 = nb0 * tensor->ne[0];
    size_t nb2 = nb1 * tensor->ne[1];
    size_t offset = (start2 * tensor->ne[1] * tensor->ne[0] + start1 * tensor->ne[0] + start0) * nb0;

    // Create a 3D view on the original tensor
    auto sliced_view = ggml_view_3d(ctx, tensor, new_shape[0], new_shape[1], new_shape[2], nb1, nb2, offset);

    return sliced_view;
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
    ASSERT(dim == 1, "Only split on second dimension is supported");
    ASSERT(left + right == tensor->ne[dim], "Left and right should sum to the dimension size");

    if (!ggml_is_contiguous(tensor))
        tensor = ggml_cont(ctx, tensor);

    auto left_tensor = slice_3d(ctx, tensor, 0, -1, 0, left, 0, -1);
    auto right_tensor = slice_3d(ctx, tensor, 0, -1, left, -1, 0, -1);
    return std::make_pair(left_tensor, right_tensor);
}

struct ggml_tensor* flip_3d(struct ggml_context* ctx, struct ggml_tensor* tensor, int along) {
    ggml_map_custom1(ctx, tensor);

    for (int64_t i = 0; i < src->ne[0]; ++i) {
        void* src_ptr = (uint8_t*)src->data + i * src->nb[0];
        void* dst_ptr = (uint8_t*)dst->data + (src->ne[0] - 1 - i) * src->nb[0];
        memcpy(dst_ptr, src_ptr, src->nb[0]);
    }
}

struct ggml_tensor* concat_3d(struct ggml_context* ctx, struct ggml_tensor* a, struct ggml_tensor* b, int dim) {

}
#endif //VITS_CPP_GGML_UTIL_H
