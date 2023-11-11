//
// Created by Maximiliano Levi on 08/10/2023.
//

#ifndef VITS_CPP_GGML_UTIL_H
#define VITS_CPP_GGML_UTIL_H

#include <ggml/ggml.h>
#include "debug.h"
#include <limits>
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
    memset(cur->data, 0, ggml_nelements(cur) * ggml_element_size(cur));

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
                             int start2, int end2, bool view = false) {
    ASSERT(tensor->n_dims == 3, "Input tensor should be 3D");
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

    // {1 [0], 2 [4], 3 [8] }
    // {4 [12], 5 [16], 6 [20]}
    //
    // 0, 4
    // 12, 16
    // Create a 3D view on the original tensor
    auto sliced_view = ggml_view_3d(ctx, tensor, new_shape[0], new_shape[1], new_shape[2], nb1, nb2, offset);
    if (view)
        return sliced_view;

    auto cur = ggml_new_tensor(ctx, tensor->type, tensor->n_dims, new_shape);
    cur = ggml_cpy(ctx, sliced_view, cur);
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

    size_t size = ggml_element_size(src);
    // Flipping along dimension 1
    for (int i = 0; i < src->ne[0]; ++i) {
        for (int row = 0; row < src->ne[1]; ++row) {
            for (int mat = 0; mat < src->ne[2]; ++mat) {

                auto idx = (i * src->nb[0] + row * src->nb[1] + mat * src->nb[2]) / size;
                auto flipped_row = (src->ne[1] - row - 1); // Flipping the row index
                auto dst_idx = (i * src->nb[0] + flipped_row * src->nb[1] + mat * src->nb[2]) / size;
                dst_ptr[dst_idx] = src_ptr[idx];
            }
        }
    }
    //delete (int*)userdata;
}

struct ggml_tensor* flip_3d(struct ggml_context* ctx, struct ggml_tensor* tensor, int along) {
    ASSERT(tensor->n_dims == 3, "Input tensor should be 3D");
    ASSERT(along == 1, "Only supported on dim 1")
    //int* alloc_along = new int(along);
    return ggml_map_custom1(ctx, tensor, flip_3d_custom_op, 1, nullptr);
}

void custom_op2(struct ggml_tensor* dst, const struct ggml_tensor* src0, const struct ggml_tensor* src1, int ith, int nth, void* userdata, std::function<float(float, float)> op) {
    ASSERT(src0->type == GGML_TYPE_F32, "Input tensor should be fp32");
    ASSERT(src1->type == GGML_TYPE_F32, "Input tensor should be fp32");
    ASSERT(dst->type == GGML_TYPE_F32, "Output tensor should be fp32");
    auto* dst_ptr = (float*)dst->data;
    auto* src0_ptr = (float*)src0->data;
    auto* src1_ptr = (float*)src1->data;

    ASSERT(ggml_nelements(src0) == ggml_nelements(src1), "Input tensors should have the same size");
    ASSERT(ggml_nelements(src0) == ggml_nelements(dst), "Input tensors should have the same size");

    size_t size = ggml_element_size(src0);
    for (int i = 0; i < src0->ne[0]; ++i) {
        for (int j = 0; j < src0->ne[1]; ++j) {
            for (int k = 0; k < src0->ne[2]; ++k) {
                for (int w = 0; w < src0->ne[3]; ++w) {

                    auto idx0 = (i * src0->nb[0] + j * src0->nb[1] + k * src0->nb[2] + w * src0->nb[3]) / size;
                    auto idx1 = (i * src1->nb[0] + j * src1->nb[1] + k * src1->nb[2] + w * src1->nb[3]) / size;
                    auto dst_idx = (i * dst->nb[0] + j * dst->nb[1] + k * dst->nb[2] + w * dst->nb[3]) / size;

                    dst_ptr[dst_idx] = op(src0_ptr[idx0], src1_ptr[idx1]);
                }
            }
        }
    }
}

void custom_op3(struct ggml_tensor* dst, const struct ggml_tensor* src0, const struct ggml_tensor* src1, const struct ggml_tensor* src2, int ith, int nth, void* userdata, std::function<float(float, float, float)> op) {
    ASSERT(src0->type == GGML_TYPE_F32, "Input tensor should be fp32");
    ASSERT(src1->type == GGML_TYPE_F32, "Input tensor should be fp32");
    ASSERT(src2->type == GGML_TYPE_F32, "Input tensor should be fp32");
    ASSERT(dst->type == GGML_TYPE_F32, "Output tensor should be fp32");
    auto* dst_ptr = (float*)dst->data;
    auto* src0_ptr = (float*)src0->data;
    auto* src1_ptr = (float*)src1->data;
    auto* src2_ptr = (float*)src2->data;

    ASSERT(ggml_nelements(src0) == ggml_nelements(src1), "Input tensors should have the same size");
    ASSERT(ggml_nelements(src0) == ggml_nelements(src2), "Input tensors should have the same size");
    ASSERT(ggml_nelements(src0) == ggml_nelements(dst), "Input tensors should have the same size");

    size_t size = ggml_element_size(src0);
    for (int i = 0; i < src0->ne[0]; ++i) {
        for (int j = 0; j < src0->ne[1]; ++j) {
            for (int k = 0; k < src0->ne[2]; ++k) {
                for (int w = 0; w < src0->ne[3]; ++w) {

                    auto idx0 = (i * src0->nb[0] + j * src0->nb[1] + k * src0->nb[2] + w * src0->nb[3]) / size;
                    auto idx1 = (i * src1->nb[0] + j * src1->nb[1] + k * src1->nb[2] + w * src1->nb[3]) / size;
                    auto idx2 = (i * src2->nb[0] + j * src2->nb[1] + k * src2->nb[2] + w * src2->nb[3]) / size;
                    auto dst_idx = (i * dst->nb[0] + j * dst->nb[1] + k * dst->nb[2] + w * dst->nb[3]) / size;

                    dst_ptr[dst_idx] = op(src0_ptr[idx0], src1_ptr[idx1], src2_ptr[idx2]);
                }
            }
        }
    }
}

void custom_op_with_data(struct ggml_tensor* dst, const struct ggml_tensor* src, int ith, int nth, void* userdata, std::function<float(float, void*)> op) {
    ASSERT(src->type == GGML_TYPE_F32, "Input tensor should be fp32");
    ASSERT(dst->type == GGML_TYPE_F32, "Output tensor should be fp32");
    auto* dst_ptr = (float*)dst->data;
    auto* src_ptr = (float*)src->data;

    ASSERT(ggml_nelements(src) == ggml_nelements(dst), "Input tensors should have the same size");

    size_t size = ggml_element_size(src);
    for (int w = 0; w < src->ne[3]; ++w) {
        for (int k = 0; k < src->ne[2]; ++k) {
            for (int j = 0; j < src->ne[1]; ++j) {
                for (int i = 0; i < src->ne[0]; ++i) {
                    auto idx = (i * src->nb[0] + j * src->nb[1] + k * src->nb[2] + w * src->nb[3]) / size;
                    auto dst_idx = (i * dst->nb[0] + j * dst->nb[1] + k * dst->nb[2] + w * dst->nb[3]) / size;

                    dst_ptr[dst_idx] = op(src_ptr[idx], userdata);
                }
            }
        }
    }
}

void custom_op(struct ggml_tensor* dst, const struct ggml_tensor* src, int ith, int nth, void* userdata, float (*op)(float)) {
    custom_op_with_data(dst, src, ith, nth, userdata, [op](float src, void* userdata) {
        return op(src);
    });
}

struct ggml_tensor* ggml_sigmoid(struct ggml_context* ctx, struct ggml_tensor* tensor) {
    ggml_custom1_op_t func = [](struct ggml_tensor * dst, const struct ggml_tensor * a, int ith, int nth, void * userdata) {
        auto sigmoid_custom_op = [](float src) {
            return 1.0f / (1.0f + std::exp(-src));
        };
        return custom_op(dst, a, ith, nth, nullptr, sigmoid_custom_op);
    };

    return ggml_map_custom1(
            ctx,
            tensor,
            func,
            1,
            nullptr
    );
}


struct ggml_tensor* ggml_exp(struct ggml_context* ctx, struct ggml_tensor* tensor) {
    ggml_custom1_op_t func = [](auto* dst, auto a, auto ith, auto nth, auto userdata) {
        custom_op(dst, a, ith, nth, nullptr, std::exp);
    };

    return ggml_map_custom1(
            ctx,
            tensor,
            func,
            1,
            nullptr
    );
}

struct ggml_tensor* softplus(struct ggml_context* ctx, struct ggml_tensor* tensor) {
    ggml_custom1_op_t func = [](auto* dst, auto a, auto ith, auto nth, auto userdata) {
        custom_op(dst, a, ith, nth, nullptr, [](float x) {
            const float beta = 1.0;
            const float threshold = 20;
            if (x > threshold)
                return beta * x;
            return (float) (1.0f / beta * std::log(1.0 + std::exp(beta * x)));
        });
    };

    return ggml_map_custom1(
            ctx,
            tensor,
            func,
            1,
            nullptr
    );
}

struct ggml_tensor* tensor_pow(struct ggml_context* ctx, struct ggml_tensor* tensor, double to) {
    auto storage = new double(to);
    ggml_custom1_op_t func = [](struct ggml_tensor * dst, const struct ggml_tensor * a, int ith, int nth, void * userdata) {
        auto op = [](float src, void* userdata) {
            auto to = *(double*)userdata;
            return (float)std::pow(src, to);
        };
        custom_op_with_data(dst, a, ith, nth, userdata, op);
        delete (double*)userdata;
    };

    return ggml_map_custom1(
            ctx,
            tensor,
            func,
            1,
            storage
    );
}


struct ggml_tensor* ggml_ceil(struct ggml_context* ctx, struct ggml_tensor* tensor) {
    ggml_custom1_op_t func = [](auto* dst, auto a, auto ith, auto nth, auto userdata) {
        custom_op(dst, a, ith, nth, nullptr, std::ceil);
    };

    return ggml_map_custom1(
            ctx,
            tensor,
            func,
            1,
            nullptr
    );
}

struct ggml_tensor* leaky_relu(struct ggml_context* ctx, struct ggml_tensor* tensor, double slope) {
    ggml_custom1_op_t func = [](struct ggml_tensor * dst, const struct ggml_tensor * a, int ith, int nth, void * userdata) {
        auto leaky_relu_custom_op = [](float src, void* userdata) {
            auto slope = *(double*)userdata;
            return (float)((src > 0) ? src : src * slope);
        };
        custom_op_with_data(dst, a, ith, nth, userdata, leaky_relu_custom_op);
    };

    return ggml_map_custom1(
            ctx,
            tensor,
            func,
            1,
            &slope
    );
}

struct ggml_tensor* per_row_cumsum(struct ggml_context* ctx, struct ggml_tensor* tensor) {
    ASSERT(tensor->ne[2] == 1, "Input tensor should be 2D");
    ASSERT(tensor->n_dims == 3, "Input tensor should be 3D");
    ggml_custom1_op_t func = [](struct ggml_tensor * dst, const struct ggml_tensor * a, int ith, int nth, void * userdata) {
        float cum = 0;
        int idx = 0;
        int elements_per_row = a->ne[0];
        auto op = [&cum, &idx, elements_per_row](float src, void* userdata) {
            if (idx == elements_per_row) {
                cum = 0;
                idx = 0;
            }
            cum += src;
            idx += 1;
            return cum;
        };
        custom_op_with_data(dst, a, ith, nth, userdata, op);
    };

    return ggml_map_custom1(
            ctx,
            tensor,
            func,
            1,
            nullptr
    );
}

struct ggml_tensor* tensor_max(struct ggml_context* ctx, struct ggml_tensor* tensor) {
    ggml_custom1_op_t func = [](struct ggml_tensor * dst, const struct ggml_tensor * a, int ith, int nth, void * userdata) {
        float current_max = std::numeric_limits<float>::min();
        auto op = [&current_max](float src, void* userdata) {
            current_max = std::max(current_max, src);
            return current_max;
        };
        custom_op_with_data(dst, a, ith, nth, userdata, op);
    };

    auto max = ggml_map_custom1(
            ctx,
            tensor,
            func,
            1,
            nullptr
    );
    // Can be improved
    max = ggml_cont(ctx, max);
    return ggml_view_1d(ctx, max, 1, (ggml_nelements(max)-1) * ggml_element_size(max));
}


struct ggml_tensor* tensor_compare(struct ggml_context* ctx, struct ggml_tensor* a, struct ggml_tensor* b, std::function<bool(float, float)> compare_op) {
    ASSERT(a->n_dims == b->n_dims, "Input tensors should have the same number of dimensions");
    ASSERT(a->ne[a->n_dims-1] == b->ne[b->n_dims-1], "Input tensors should have the same size on the first dimension");

    if (a->ne[0] != b->ne[0] && a->n_dims == 2)
        a = ggml_repeat(ctx, a, b);

    auto compare_op_heap = new std::function<bool(float, float)>(compare_op);
    ggml_custom2_op_t func = [](struct ggml_tensor * dst, const struct ggml_tensor * a, const struct ggml_tensor * b, int ith, int nth, void * userdata) {
        auto& func = *((std::function<bool(float, float)>*) userdata);
        auto op = [&func](float src0, float src1) {
            return func(src0, src1) ? 1.0f : 0.0f;
        };
        custom_op2(dst, a, b, ith, nth, userdata, op);

        delete (std::function<bool(float, float)>*) userdata;
    };

    return ggml_map_custom2(
            ctx,
            a,
            b,
            func,
            1,
            compare_op_heap
    );
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

    // Copy a's data
    auto view_a = slice_3d(
            ctx,
            result,
            0,dim == 0 ? a->ne[0] : -1,
            0, dim == 1 ? a->ne[1] : -1,
            0, -1,
            true
    );

    auto view_b = slice_3d(
            ctx,
            result,
            dim == 0 ? a->ne[0] : 0, -1,
            dim == 1 ? a->ne[1] : 0, -1,
            0, -1,
            true
    );

    auto copy_a = ggml_cpy(ctx, a, view_a);
    auto copy_b = ggml_cpy(ctx, b, view_b);
    result = ggml_cpy(ctx, result, result);
    result->src[1] = copy_a;
    result->src[2] = copy_b;

    return result;
}

struct ggml_tensor* concat_2d(struct ggml_context* ctx, struct ggml_tensor* a, struct ggml_tensor* b, int dim) {
    a = ggml_reshape_3d(ctx, a, a->ne[0], a->ne[1], 1);
    b = ggml_reshape_3d(ctx, b, b->ne[0], b->ne[1], 1);
    auto tensor = concat_3d(ctx, a, b, dim);
    tensor = ggml_reshape_2d(ctx, tensor, tensor->ne[0], tensor->ne[1]);
    return tensor;
}

struct ggml_tensor* tensor_randn(struct ggml_context* ctx, std::vector<int64_t> dims) {
    auto tensor = ggml_new_tensor(ctx, GGML_TYPE_F32, dims.size(), dims.data());
    auto data = static_cast<float*>(tensor->data);
    auto size = ggml_nelements(tensor) ;
    std::mt19937 rng;
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < size; ++i) {
        data[i] = dist(rng);
    }
    return tensor;
}

struct ggml_tensor* tensor_zeros(struct ggml_context* ctx, std::vector<int64_t> dims) {
    auto tensor = ggml_new_tensor(ctx, GGML_TYPE_F32, dims.size(), dims.data());
    auto data = static_cast<float*>(tensor->data);
    auto size = ggml_nelements(tensor) ;
    for (int i = 0; i < size; ++i) {
        data[i] = 0;
    }
    return tensor;
}

struct ggml_tensor* tensor_randn_like(struct ggml_context* ctx, struct ggml_tensor* other) {
    std::vector<int64_t> dims;
    for (int i = 0; i < other->n_dims; ++i) {
        dims.push_back(other->ne[i]);
    }
    return tensor_randn(ctx, dims);
}

struct ggml_tensor* tensor_like(struct ggml_context* ctx, struct ggml_tensor* other, float value) {
    auto tensor = ggml_new_tensor(ctx, other->type, other->n_dims, other->ne);
    auto data = static_cast<float*>(tensor->data);
    auto size = ggml_nelements(tensor) ;
    for (int i = 0; i < size; ++i) {
        data[i] = value;
    }
    return tensor;
}

struct ggml_tensor* ones_like(struct ggml_context* ctx, struct ggml_tensor* other) {
    return tensor_like(ctx, other, 1.0f);
}


struct ggml_tensor* zeros_like(struct ggml_context* ctx, struct ggml_tensor* other) {
    return tensor_like(ctx, other, 0.0f);
}

struct ggml_tensor* index_put_last_dim(struct ggml_context* ctx, struct ggml_tensor* tensor, int index, float value) {
    // our index is actually 0
    ASSERT(tensor->n_dims == 3, "Only support 3d tensors");
    auto offset = tensor->nb[0] * index;
    auto view = ggml_view_3d(ctx, tensor, 1, tensor->ne[1], tensor->ne[2], tensor->nb[1], tensor->nb[2], offset);
    auto new_values = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, tensor->ne[1] * tensor->ne[2]);
    for (size_t i = 0; i < ggml_nelements(new_values); ++i) {
        ((float*)new_values->data)[i] = value;
    }

    auto cpy = ggml_cpy(ctx, new_values, view);
    auto to_return = ggml_cpy(ctx, tensor, tensor);
    to_return->src[1] = cpy;

    return to_return;
}

struct ggml_tensor* index_add_last_dim(struct ggml_context* ctx, struct ggml_tensor* tensor, int index, float value) {
    ASSERT(tensor->n_dims == 3, "Only support 3d tensors");
    auto offset = tensor->nb[0] * index;
    auto view = ggml_view_3d(ctx, tensor, 1, tensor->ne[1], tensor->ne[2], tensor->nb[1], tensor->nb[2], offset);
    auto new_values = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, tensor->ne[1], tensor->ne[2]);
    for (size_t i = 0; i < ggml_nelements(new_values); ++i) {
        ((float*)new_values->data)[i] = value;
    }

    auto add = ggml_add_inplace(ctx, view, new_values);
    auto to_return = ggml_cpy(ctx, tensor, tensor);
    to_return->src[1] = add;

    return to_return;
}

struct ggml_tensor* broadcast_if_possible(struct ggml_context* ctx, struct ggml_tensor* tensor, struct ggml_tensor* mask) {
    if (mask->ne[0] == tensor->ne[1] && mask->ne[1] == 1) {
        mask = ggml_permute(ctx, mask, 1, 0, 2, 3);
        mask = ggml_cont(ctx, mask);
        mask = ggml_repeat(ctx, mask, tensor);
    }
    return mask;
}


struct ggml_tensor* tensor_masked_get(struct ggml_context* ctx, struct ggml_tensor* tensor, struct ggml_tensor* mask) {
    ASSERT(tensor->n_dims == 3, "Only support 3d tensors");
    ASSERT(mask->n_dims == 3, "Only support 3d tensors");
    mask = broadcast_if_possible(ctx, tensor, mask);

    ggml_custom2_op_t func = [](struct ggml_tensor * dst, const struct ggml_tensor * a, const struct ggml_tensor * b, int ith, int nth, void * userdata) {
        auto op = [](float src0, float src1) {
            ASSERT(abs(src1) < 1e-5f || abs(src1 - 1) < 1e-5f, "Mask tensor should be 0 or 1");
            return ((int)src1) == 1 ? src0 : 0;
        };
        custom_op2(dst, a, b, ith, nth, userdata, op);
    };
    return ggml_map_custom2(
            ctx,
            tensor,
            mask,
            func,
            1,
            nullptr
    );
}

struct ggml_tensor* tensor_gather(struct ggml_context* ctx, struct ggml_tensor* tensor, int dim, struct ggml_tensor* index) {
    ASSERT(tensor->n_dims == 3, "Only support 3d tensors");
    ASSERT(index->n_dims == 1, "Only support 1d tensors for the index");
    ASSERT(dim == 0, "Only dim == 0 supported");

    ggml_custom2_op_t func = [](struct ggml_tensor * dst, const struct ggml_tensor * index_tensor, const struct ggml_tensor * values_tensor, int ith, int nth, void * userdata) {
        auto values_ptr = ggml_get_data_f32(values_tensor);
        ASSERT(dst->n_dims == 1, "Only support 1d tensors");
        ASSERT(dst->ne[0] == index_tensor->ne[0], "Index and output mismatch");

        for(int i = 0; i < index_tensor->ne[0]; ++i)
        {
            int j = (int) ((float*) index_tensor->data)[i];
            int index = j + i * values_tensor->ne[0];
            ASSERT(index < ggml_nelements(values_tensor) && index > 0, "Index should be smaller than the number of elements in the value tensor");
            ((float*)dst->data)[i] = values_ptr[index];
        }
    };

    return ggml_map_custom2(
            ctx,
            index,
            tensor,
            func,
            1,
            nullptr
    );
}

struct ggml_tensor* tensor_not(struct ggml_context* ctx, struct ggml_tensor* tensor) {
    ggml_custom1_op_t func = [](auto* dst, auto a, auto ith, auto nth, auto userdata) {
        custom_op(dst, a, ith, nth, nullptr, [](float x) {
            ASSERT(abs(x) < 1e-5f || abs(x - 1) < 1e-5f, "Input tensor should be 0 or 1");
            return ((int)x) == 0 ? 1.0f : 0.0f;
        });
    };

    return ggml_map_custom1(
            ctx,
            tensor,
            func,
            1,
            nullptr
    );
}

struct ggml_tensor* tensor_masked_set(struct ggml_context* ctx, struct ggml_tensor* tensor, struct ggml_tensor* mask, struct ggml_tensor* value) {
    ASSERT(tensor->n_dims == 3, "Only support 3d tensors");
    ASSERT(mask->n_dims == 3, "Only support 3d tensors");
    mask = broadcast_if_possible(ctx, tensor, mask);
    // tensor = tensor * ~mask + value * mask
    auto cur = zeros_like(ctx, tensor);
    cur = ggml_add(ctx, cur, ggml_mul(ctx, tensor, tensor_not(ctx, mask)));
    cur = ggml_add(ctx, cur, ggml_mul(ctx, value, mask));
    return cur;
    /*
    ggml_custom2_op_t func = [](struct ggml_tensor * dst, const struct ggml_tensor * a, const struct ggml_tensor * b, int ith, int nth, void * userdata) {
        int index = 0;
        auto value_tensor = (struct ggml_tensor*) userdata;
        float* values = ggml_get_data_f32(value_tensor);
        auto op = [&index, &values, value_tensor](float src0, float src1) {
            ASSERT(abs(src1) < 1e-5f || abs(src1 - 1) < 1e-5f, "Mask tensor should be 0 or 1");
            ASSERT(index < ggml_nelements(value_tensor), "Index should be smaller than the number of elements in the value tensor");
            return ((int)src1) == 1 ? values[index++] : src0;
        };
        custom_op2(dst, a, b, ith, nth, userdata, op);
    };
    return ggml_map_custom3(
            ctx,
            tensor,
            mask,
            func,
            1,
            value
    );*/
}

struct ggml_tensor* tensor_arange(struct ggml_context* ctx, int end) {
    auto tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, end);
    auto data = static_cast<float*>(tensor->data);
    for (int i = 0; i < end; ++i) {
        data[i] = (float)i;
    }
    return tensor;
}

struct ggml_tensor* tensor_conv_transpose_1d_get_output(struct ggml_context* ctx, struct ggml_tensor* input, struct ggml_tensor* weights, int stride) {
    auto batch_size = input->ne[2];
    auto input_channels = input->ne[1];
    auto input_length = input->ne[0];

    auto input_channels_weights = weights->ne[2];
    auto output_channels = weights->ne[1];
    auto kernel_size = weights->ne[0];

    ASSERT(input_channels == input_channels_weights, "Input channels mismatch!");

    auto output_length = (input_length - 1) * stride + kernel_size;
    auto output_tensor = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, output_length, output_channels, batch_size);
    return output_tensor;
}

struct ggml_tensor* tensor_conv_transpose_1d(struct ggml_context* ctx, struct ggml_tensor* input, struct ggml_tensor* weights, int stride, int padding, int dilation) {
    ASSERT(input->n_dims == 3, "Input tensor should be 3D");

    auto output_tensor = tensor_conv_transpose_1d_get_output(ctx, input, weights, stride);

    struct conv_data_t {
        int stride;
        int padding;
        int dilation;
    };

    auto conv = new conv_data_t{stride, padding, dilation};

    ggml_map_custom3_inplace(ctx, output_tensor, input, weights, [](struct ggml_tensor * dst, const struct ggml_tensor * _, const struct ggml_tensor * input, const struct ggml_tensor * weights, int ith, int nth, void * userdata) {
        auto conv_data = ((struct conv_data_t*)userdata);

        auto batch_size = input->ne[2];
        auto input_channels = input->ne[1];
        auto input_length = input->ne[0];

        auto input_channels_weights = weights->ne[2];
        auto output_channels = weights->ne[1];
        auto kernel_size = weights->ne[0];

        ASSERT(input_channels == input_channels_weights, "Input channels mismatch!");

        auto* dst_ptr = (float*)dst->data;
        auto* input_ptr = (float*)input->data;
        auto* weights_ptr = (float*)weights->data;

        for (int b = 0; b < batch_size; ++b) {
            for (int i = 0; i < output_channels; ++i) {
                for (int j = 0; j < input_channels; ++j) {
                    for (int k = 0; k < input_length; ++k) {
                        for (int l = 0; l < kernel_size; ++l) {
                            auto start = k * conv_data->stride;
                            auto end = start + kernel_size;

                            //output_tensor[b, i, start:end] += input_tensor[b, j, k] * weights[j, i]
                            dst_ptr[b * dst->nb[0] * dst->nb[1] + i * dst->nb[0] + start + l] += input_ptr[b * input->nb[0] * input->nb[1] + j * input->nb[0] + k] * weights_ptr[i * weights->nb[0] * weights->nb[1] + j * weights->nb[0] + l];
                        }
                    }
                }
            }
        }

        delete (struct conv_data_t*)userdata;
    }, 1, conv);
}

#endif //VITS_CPP_GGML_UTIL_H
