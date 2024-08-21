//
// Created by Maximiliano Levi on 08/10/2023.
//

#ifndef VITS_CPP_GGML_UTIL_H
#define VITS_CPP_GGML_UTIL_H

#include "common.h"
#include <ggml/ggml.h>
#include "debug.h"
#include <limits>
#include <ggml/ggml-alloc.h>
#include <random>
#include "custom-ops.h"

struct ggml_tensor* pad_3d(struct ggml_context* ctx, struct ggml_tensor* tensor, std::vector<int> pads) {
    ASSERT(tensor->n_dims == 3, "pad_3d: Input tensor should be 3D");
    ASSERT(pads.size() == 6, "Invalid pad count");
    if (!ggml_is_contiguous(tensor))
        tensor = ggml_cont(ctx, tensor);

    int64_t new_shape[GGML_MAX_DIMS];
    for(int i = 0; i < tensor->n_dims; i++) {
        int reverse_index = (tensor->n_dims - i - 1) * 2;
        new_shape[i] = tensor->ne[i] + pads[reverse_index] + pads[reverse_index + 1];
    }

    auto cur = ggml_new_tensor(ctx, tensor->type, tensor->n_dims, new_shape);
    cur = tensor_set_zero(ctx, cur);

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
    cur = tensor_set_zero(ctx, cur);

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
                             int start2, int end2, bool view = false) {
    ASSERT(tensor->n_dims == 3, "slice_3d: Input tensor should be 3D");
    if (!ggml_is_contiguous(tensor))
        tensor = ggml_cont(ctx, tensor);

    // Handle negative index -1 as the last index
    if (end0 < 0) end0 = tensor->ne[0] + (end0 + 1);
    if (end1 < 0) end1 = tensor->ne[1] + (end1 + 1);
    if (end2 < 0) end2 = tensor->ne[2] + (end2 + 1);


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

    size_t offset = start2 * tensor->ne[1] * tensor->ne[0] * nb0
                    + start1 * tensor->ne[0] * nb0
                    + start0 * nb0;

    auto sliced_view = ggml_view_3d(ctx, tensor, new_shape[0], new_shape[1], new_shape[2], nb1, nb2, offset);
    ggml_format_name(sliced_view, "%s_sliced_[%d-%d, %d-%d, %d-%d]", tensor->name, start0, end0, start1, end1, start2, end2);
    if (view)
        return sliced_view;

    auto cur = ggml_new_tensor(ctx, tensor->type, tensor->n_dims, new_shape);
    //cur = tensor_set_zero(ctx, cur);
    cur = ggml_cpy(ctx, sliced_view, cur);
    ggml_format_name(cur, "%s_sliced_copy_[%d:%d, %d:%d, %d:%d]", tensor->name, start0, end0, start1, end1, start2, end2);
    return cur;
}

struct ggml_tensor* slice_2d(struct ggml_context* ctx, struct ggml_tensor* tensor, int start0, int end0, int start1, int end1) {
    tensor = ggml_reshape_3d(ctx, tensor, tensor->ne[0], tensor->ne[1], 1);
    tensor = slice_3d(ctx, tensor, start0, end0, start1, end1, 0, -1);
    tensor = ggml_reshape_2d(ctx, tensor, tensor->ne[0], tensor->ne[1]);
    return tensor;
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
    ASSERT(tensor->n_dims == 3, "split3d: Input tensor should be 3D");
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

struct ggml_tensor* concat_3d(struct ggml_context* ctx, struct ggml_tensor* a, struct ggml_tensor* b, int dim) {
    ASSERT(a->n_dims == 3, "Input B tensor should be 3D");
    ASSERT(b->n_dims == 3, "Input A tensor should be 3D");
    for (int i = 0; i < 3; ++i) {
        if (i != dim)
            ASSERT(a->ne[i] == b->ne[i], "a and b should have the same dimension on all dimensions except the concat dimension");
    }
    ASSERT(b->ne[2] == b->ne[2], "a and b should have the same dimensions");
    ASSERT(dim == 1 || dim == 0, "Only concat on first and second dimension is supported");

    int64_t new_shape[3] = {
            dim == 0 ? a->ne[0] + b->ne[0] : a->ne[0],
            dim == 1 ? a->ne[1] + b->ne[1] : a->ne[1],
            a->ne[2]
    };
    struct ggml_tensor* result = ggml_new_tensor(ctx, a->type, a->n_dims, new_shape);
    //result = tensor_set_zero(ctx, result);
    auto a_set = tensor_set_inplace(ctx, result, a, 0, 0, 0);
    auto b_set = tensor_set_inplace(ctx, a_set, b, dim == 0 ? a->ne[0] : 0, dim == 1 ? a->ne[1] : 0, 0);
    return b_set;
}

extern std::default_random_engine rng;

struct ggml_tensor* tensor_randn(struct ggml_context* ctx, struct ggml_allocr* allocr, std::vector<int64_t> dims) {
    auto tensor = ggml_new_tensor(ctx, DEFAULT_TENSOR_TYPE, dims.size(), dims.data());
    ALLOC(tensor)
    auto data = static_cast<float*>(tensor->data);
    auto size = ggml_nelements(tensor) ;
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < size; ++i) {
        data[i] = dist(rng);
    }
    return tensor;
}

struct ggml_tensor* tensor_randn_like(struct ggml_context* ctx, struct ggml_allocr* allocr, struct ggml_tensor* other) {
    std::vector<int64_t> dims;
    for (int i = 0; i < other->n_dims; ++i) {
        dims.push_back(other->ne[i]);
    }
    return tensor_randn(ctx, allocr, dims);
}

struct ggml_tensor* tensor_like(struct ggml_context* ctx, struct ggml_allocr* allocr, struct ggml_tensor* other, float value) {
    std::vector<int64_t> shape;
    for (auto i = 0; i < other->n_dims; ++i) {
        shape.push_back(other->ne[i]);
    }
    return tensor_shaped_like(ctx, allocr, other->type, shape, value);
}

struct ggml_tensor* ones_like(struct ggml_context* ctx, struct ggml_allocr* allocr, struct ggml_tensor* other) {
    return tensor_like(ctx, allocr, other, 1.0f);
}

struct ggml_tensor* zeros_like(struct ggml_context* ctx, struct ggml_allocr* allocr, struct ggml_tensor* other) {
    return tensor_like(ctx, allocr, other, 0.0f);
}

struct ggml_tensor* tensor_detach(struct ggml_context* ctx, struct ggml_tensor* tensor) {
    auto detached = ggml_new_tensor(ctx, tensor->type, tensor->n_dims, tensor->ne);
    ASSERT(!ggml_get_no_alloc(ctx), "Cannot detach tensor when no alloc is set");
    memcpy(detached->data, tensor->data, ggml_nelements(tensor) * ggml_element_size(tensor));
    return detached;
}

struct ggml_tensor* index_put_last_dim(struct ggml_context* ctx, struct ggml_allocr* allocr, struct ggml_tensor* tensor, int index, float value) {
    // our index is actually 0
    ASSERT(tensor->n_dims == 3, "Only support 3d tensors");
    auto offset = tensor->nb[0] * index;
    auto view = ggml_view_3d(ctx, tensor, 1, tensor->ne[1], tensor->ne[2], tensor->nb[1], tensor->nb[2], offset);
    auto new_values = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, tensor->ne[1] * tensor->ne[2]);
    ALLOC(new_values)
    for (size_t i = 0; i < ggml_nelements(new_values); ++i) {
        ((float*)new_values->data)[i] = value;
    }

    auto cpy = ggml_cpy(ctx, new_values, view);
    auto to_return = ggml_cpy(ctx, tensor, tensor);
    to_return->src[1] = cpy;

    return to_return;
}

struct ggml_tensor* index_add_last_dim(struct ggml_context* ctx, struct ggml_allocr* allocr, struct ggml_tensor* tensor, int index, float value) {
    ASSERT(tensor->n_dims == 3, "Only support 3d tensors");
    auto offset = tensor->nb[0] * index;
    auto view = ggml_view_3d(ctx, tensor, 1, tensor->ne[1], tensor->ne[2], tensor->nb[1], tensor->nb[2], offset);
    auto new_values = ggml_new_tensor_3d(ctx, DEFAULT_TENSOR_TYPE, 1, tensor->ne[1], tensor->ne[2]);
    ALLOC(new_values)
    for (size_t i = 0; i < ggml_nelements(new_values); ++i) {
        ((float*)new_values->data)[i] = value;
    }

    auto add = ggml_add_inplace(ctx, view, new_values);
    auto to_return = ggml_cpy(ctx, tensor, tensor);
    to_return->src[1] = add;

    return to_return;
}


struct ggml_tensor* tensor_arange(struct ggml_context* ctx, struct ggml_allocr* allocr, int end) {
    auto tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, end);
    ALLOC(tensor)
    auto data = static_cast<float*>(tensor->data);
    for (int i = 0; i < end; ++i) {
        data[i] = (float)i;
    }
    return tensor;
}

struct ggml_tensor* reshape_3d(struct ggml_context* ctx, struct ggml_tensor* tensor, size_t dim0, size_t dim1, size_t dim2) {
    ASSERT(dim0 * dim1 * dim2 == ggml_nelements(tensor), "Invalid reshape");

    return ggml_reshape_3d(ctx, tensor, dim0, dim1, dim2);
    //return ggml_view_3d(ctx, tensor, dim0, dim1, dim2, tensor->nb[1], tensor->nb[2], 0);
}

struct ggml_tensor* reshape_4d(struct ggml_context* ctx, struct ggml_tensor* tensor, size_t dim0, size_t dim1, size_t dim2, size_t dim3) {
    ASSERT(dim0 * dim1 * dim2 * dim3 == ggml_nelements(tensor), "Invalid reshape");

    return ggml_reshape_4d(ctx, tensor, dim0, dim1, dim2, dim3);
    //return ggml_view_4d(ctx, tensor, dim0, dim1, dim2, dim3, tensor->nb[1], tensor->nb[2], tensor->nb[3], 0);
}

struct ggml_tensor* cast_tensor(struct ggml_context* ctx, struct ggml_tensor* tensor, ggml_type to) {
    if (tensor->type == to) return tensor;

    struct ggml_tensor* target = ggml_new_tensor(ctx, to, tensor->n_dims, tensor->ne);
    //target = tensor_set_zero(ctx, target);
    return ggml_cpy(ctx, tensor, target);
}

struct ggml_tensor* unsqueeze(struct ggml_context* ctx, struct ggml_tensor* tensor, int dim) {
    if (tensor->n_dims == 1) {
        if (dim == 0)
            tensor = ggml_view_2d(ctx, tensor, 1, tensor->ne[0], tensor->nb[1], 0);
        else if (dim == 1)
            tensor = ggml_view_2d(ctx, tensor, tensor->ne[0], 1, tensor->nb[1], 0);
        else
            ASSERT(false, "Invalid dim");
    } else if (tensor->n_dims == 2) {
        if (dim == 0)
            tensor = ggml_view_3d(ctx, tensor, 1, tensor->ne[0], tensor->ne[1], tensor->nb[1], tensor->nb[2], 0);
        else if (dim == 2)
            tensor = ggml_view_3d(ctx, tensor, tensor->ne[0], tensor->ne[1], 1, tensor->nb[1], tensor->nb[2], 0);
        else
            ASSERT(false, "Invalid dim");
    } else if (tensor->n_dims == 3) {
        if (dim == 0)
            tensor = ggml_view_4d(ctx, tensor, 1, tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->nb[1], tensor->nb[2], tensor->nb[3], 0);
        else if (dim == 1)
            tensor = ggml_view_4d(ctx, tensor, tensor->ne[0], 1, tensor->ne[1], tensor->ne[2], tensor->nb[1], tensor->nb[2], tensor->nb[3], 0);
        else if (dim == 2)
            tensor = ggml_view_4d(ctx, tensor, tensor->ne[0], tensor->ne[1], 1, tensor->ne[2], tensor->nb[1], tensor->nb[2], tensor->nb[3], 0);
        else if (dim == 3)
            tensor = ggml_view_4d(ctx, tensor, tensor->ne[0], tensor->ne[1], tensor->ne[2], 1, tensor->nb[1], tensor->nb[2], tensor->nb[3], 0);
        else
            ASSERT(false, "Invalid dim");
    } else {
        ASSERT(false, "Invalid tensor dimension");
    }
    return tensor;
}

struct ggml_tensor* squeeze(struct ggml_context* ctx, struct ggml_tensor* tensor, int dim) {
    if (tensor->n_dims == 2) {
        if ((dim == 0 && tensor->ne[0] == 1) || (dim == 1 && tensor->ne[1] == 1)) {
            tensor = ggml_view_1d(ctx, tensor, tensor->ne[1], 0);
        } else {
            ASSERT(false, "Invalid squeeze dimension or non-squeezable dimension");
        }
    } else if (tensor->n_dims == 3) {
        if (dim == 0 && tensor->ne[0] == 1) {
            tensor = ggml_view_2d(ctx, tensor, tensor->ne[1], tensor->ne[2], tensor->nb[2], 0);
        } else if (dim == 1 && tensor->ne[1] == 1) {
            tensor = ggml_view_2d(ctx, tensor, tensor->ne[0], tensor->ne[2], tensor->nb[2], 0);
        } else if (dim == 2 && tensor->ne[2] == 1) {
            tensor = ggml_view_2d(ctx, tensor, tensor->ne[0], tensor->ne[1], tensor->nb[1], 0);
        } else {
            ASSERT(false, "Invalid squeeze dimension or non-squeezable dimension");
        }
    } else if (tensor->n_dims == 4) {
        if (dim == 0 && tensor->ne[0] == 1) {
            tensor = ggml_view_3d(ctx, tensor, tensor->ne[1], tensor->ne[2], tensor->ne[3], tensor->nb[2],
                                  tensor->nb[3], 0);
        } else if (dim == 1 && tensor->ne[1] == 1) {
            tensor = ggml_view_3d(ctx, tensor, tensor->ne[0], tensor->ne[2], tensor->ne[3], tensor->nb[2],
                                  tensor->nb[3], 0);
        } else if (dim == 2 && tensor->ne[2] == 1) {
            tensor = ggml_view_3d(ctx, tensor, tensor->ne[0], tensor->ne[1], tensor->ne[3], tensor->nb[1],
                                  tensor->nb[3], 0);
        } else if (dim == 3 && tensor->ne[3] == 1) {
            tensor = ggml_view_3d(ctx, tensor, tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->nb[1],
                                  tensor->nb[2], 0);
        } else {
            ASSERT(false, "Invalid squeeze dimension or non-squeezable dimension");
        }
    } else {
        ASSERT(false, "Invalid tensor dimension for squeeze");
    }

    return tensor;
}
#endif //VITS_CPP_GGML_UTIL_H
