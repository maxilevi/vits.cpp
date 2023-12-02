//
// Created by Maximiliano Levi on 11/19/23.
//

#ifndef VITS_CUSTOM_OPS_H
#define VITS_CUSTOM_OPS_H

#include <ggml/ggml.h>
#include "debug.h"
#include "common.h"

#define START_BENCH() /*auto start = std::chrono::high_resolution_clock::now();*/

#define PRINT_BENCH(name) /*auto end = std::chrono::high_resolution_clock::now(); \
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start); \
    if (duration.count() > 1)                                                                        \
        printf("Time taken by function %s: %lld ms\n",  name , duration.count());*/

struct ggml_tensor* tensor_shaped_like(struct ggml_context* ctx, struct ggml_allocr* allocr, ggml_type type, std::vector<int64_t> shape, float value) {
    auto tensor = ggml_new_tensor(ctx, type, shape.size(), shape.data());
    ALLOC(tensor)
    auto data_fp32 = (type == GGML_TYPE_F32)
            ? static_cast<float*>(tensor->data)
            : nullptr;
    auto data_fp16 = (type == GGML_TYPE_F16)
                ? static_cast<ggml_fp16_t*>(tensor->data)
                : nullptr;
    auto size = ggml_nelements(tensor);
    for (int i = 0; i < size; ++i) {
        if (type == GGML_TYPE_F16) {
            data_fp16[i] = ggml_fp16_t(value);
        } else if(type == GGML_TYPE_F32) {
            data_fp32[i] = value;
        } else {
            ASSERT(false, "Not supported type");
        }
    }
    return tensor;
}

struct ggml_tensor* tensor_zeros(struct ggml_context* ctx, struct ggml_allocr* allocr, std::vector<int64_t> shape) {
    return tensor_shaped_like(ctx, allocr, DEFAULT_TENSOR_TYPE, std::move(shape), 0);
}

void* allocate_temp_i32_array(const std::vector<int>& arr) {
    void* ptr = malloc(arr.size() * sizeof(int32_t));
    memcpy(ptr, arr.data(), arr.size() * sizeof(int));
    return ptr;
}

void* allocate_temp_f32_array(const std::vector<float>& arr) {
    void* ptr = malloc(arr.size() * sizeof(float));
    memcpy(ptr, arr.data(), arr.size() * sizeof(float));
    return ptr;
}

struct ggml_tensor* cleanup(struct ggml_context* ctx, struct ggml_tensor* result, void* userdata) {
    return ggml_map_custom1_inplace(ctx, result, [](struct ggml_tensor* dst, const struct ggml_tensor* src, int ith, int nth, void* userdata) {
        free(userdata);
    }, 1, userdata);
}

template<class T> T* get_temp_data(void* userdata) {
    return (T*) userdata;
}

inline int compute_index(const ggml_tensor* tensor, int i, int j, int k) {
    return (i * tensor->nb[0] + j * tensor->nb[1] + k * tensor->nb[2]) / ggml_element_size(tensor);
}

inline void compute_indices(const struct ggml_tensor* tensor, size_t w, int& i, int& j, int& k) {
    size_t temp = w * ggml_element_size(tensor);
    auto nb0 = tensor->nb[0];
    auto nb1 = tensor->nb[1];
    auto nb2 = tensor->nb[2];
    k = temp / tensor->nb[2];
    temp -= k * nb2;
    if (temp == 0) return;
    j = temp / tensor->nb[1];
    temp -= j * nb1;
    if (temp == 0) return;
    i = temp / nb0;
}


void for_each_element_threaded(const struct ggml_tensor* dst, int ith, int nth, std::function<void(int i, int j, int k)> callback) {
    auto total_elements = ggml_nelements(dst);
    auto part_size = total_elements / nth;
    auto offset = part_size * ith;
    for (size_t w = offset; w < offset + part_size; w++) {
        int i = 0, j = 0, k = 0;
        compute_indices(dst, w, i, j, k);

        ASSERT(i < dst->ne[0] && j < dst->ne[1] && k < dst->ne[2], "Invalid index");

        callback(i, j, k);
    }
}


template<class T> void custom_op2(struct ggml_tensor* dst, const struct ggml_tensor* src0, const struct ggml_tensor* src1, int ith, int nth, void* userdata, std::function<T(T, T)> op) {
    auto* dst_ptr = (T*)dst->data;
    auto* src0_ptr = (T*)src0->data;
    auto* src1_ptr = (T*)src1->data;

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

template<class T> void custom_op_with_data(struct ggml_tensor* dst, const struct ggml_tensor* src, int ith, int nth, void* userdata, std::function<T(T, void*)> op) {
    auto* dst_ptr = reinterpret_cast<T*>(dst->data);
    auto* src_ptr = reinterpret_cast<T*>(src->data);

    ASSERT(ggml_nelements(src) == ggml_nelements(dst), "Input tensors should have the same size");

    for_each_element_threaded(dst, ith, nth, [&] (int i, int j, int k) {
        int src_idx = compute_index(src, i, j, k);
        int dst_idx = compute_index(dst, i, j, k);
        dst_ptr[dst_idx] = op(src_ptr[src_idx], userdata);
    });
}

template<class T> void custom_op(struct ggml_tensor* dst, const struct ggml_tensor* src, int ith, int nth, std::function<T(T)> op) {
    custom_op_with_data<T>(dst, src, ith, nth, nullptr, [op](T src, void* userdata) {
        return op(src);
    });
}

#define TENSOR_OP_IMPL(OP_NAME, TENSOR, ...) \
    do { \
        switch ((TENSOR)->type) { \
            case GGML_TYPE_F16: \
                return OP_NAME##_impl<ggml_fp16_t>(__VA_ARGS__); \
            case GGML_TYPE_F32: \
                return OP_NAME##_impl<float>(__VA_ARGS__); \
            default: \
                ASSERT(false, "Not supported type"); \
        } \
    } while (0);

#define UNARY_INPLACE_OR_COPY_PARAM(name, suffix, has_param) \
template<class T> \
struct ggml_tensor* name##suffix##_impl(struct ggml_context* ctx, struct ggml_tensor* tensor, double value = 0.0) { \
    auto func = [] (struct ggml_tensor* dst, const struct ggml_tensor* src, int ith, int nth, void* userdata) { \
        auto* dst_ptr = (T*)dst->data;\
        auto* src_ptr = (T*)src->data;                       \
        auto data = *(get_temp_data<float>(userdata));   \
        size_t size = ggml_element_size(src);                \
        auto part_size = ggml_nelements(src) / nth;                                                     \
        auto offset = ith * part_size;                                                     \
        for(int w = offset; w < offset + part_size; ++w){\
            if constexpr (has_param)\
                dst_ptr[w] = name<T>(src_ptr[w], data);                         \
            else                                              \
                dst_ptr[w] = name<T>(src_ptr[w]);\
        };\
    };                                                       \
    auto userdata = allocate_temp_f32_array({(float) value});                                                         \
    auto result = ggml_map_custom1##suffix(ctx, tensor, func, ggml_nelements(tensor) >= get_thread_count() ? GGML_N_TASKS_MAX : 1, userdata); \
    return cleanup(ctx, result, userdata);                                                             \
}                                           \
\
struct ggml_tensor* tensor_##name##suffix(struct ggml_context* ctx, struct ggml_tensor* tensor, double _extra = 0.0) {\
    TENSOR_OP_IMPL(name##suffix, tensor, ctx, tensor, _extra)                                                                    \
}

#define UNARY_INPLACE_OR_COPY(name, suffix) UNARY_INPLACE_OR_COPY_PARAM(name, suffix, false)
#define UNARY_INPLACE(name) UNARY_INPLACE_OR_COPY(name, _inplace)
#define UNARY_COPY(name) UNARY_INPLACE_OR_COPY(name, )

#define UNARY_INPLACE_PARAM(name) UNARY_INPLACE_OR_COPY_PARAM(name, _inplace, true)
#define UNARY_COPY_PARAM(name) UNARY_INPLACE_OR_COPY_PARAM(name, , true)

#define UNARY_OP_PARAM(name) \
    UNARY_INPLACE_PARAM(name)\
    UNARY_COPY_PARAM(name)

#define UNARY_OP(name) \
    UNARY_INPLACE(name)\
    UNARY_COPY(name)

template<class T> struct ggml_tensor* flip_3d_impl(struct ggml_context* ctx, struct ggml_tensor* tensor, int along) {
    ASSERT(tensor->n_dims == 3, "flip3d: Input tensor should be 3D");

    auto func = [] (struct ggml_tensor* dst, const struct ggml_tensor* src, int ith, int nth, void* userdata) {
        auto* dst_ptr = (T*)dst->data;
        auto* src_ptr = (T*)src->data;

        size_t size = ggml_element_size(src);
        int dim = *get_temp_data<int>(userdata);
        for_each_element_threaded(dst, ith, nth, [&] (int i, int j, int k) {
            // Calculate indices based on the flipping dimension
            int flipped_i = dim == 0 ? (src->ne[0] - i - 1) : i;
            int flipped_j = dim == 1 ? (src->ne[1] - j - 1) : j;
            int flipped_k = dim == 2 ? (src->ne[2] - k - 1) : k;

            // Compute source and destination indices
            auto src_idx = (i * src->nb[0] + j * src->nb[1] + k * src->nb[2]) / size;
            auto dst_idx = (flipped_i * src->nb[0] + flipped_j * src->nb[1] + flipped_k * src->nb[2]) / size;

            // Perform the flip
            dst_ptr[dst_idx] = src_ptr[src_idx];
        });
    };

    auto userdata = allocate_temp_i32_array({along});
    auto result = ggml_map_custom1(ctx, tensor, func, GGML_N_TASKS_MAX, userdata);
    return cleanup(ctx, result, userdata);
}

template<class T> struct ggml_tensor* per_row_cumsum_impl(struct ggml_context* ctx, struct ggml_tensor* tensor) {
    ASSERT(tensor->ne[2] == 1, "Dim 2 == 1");
    ASSERT(tensor->n_dims == 3, "per_row_cumsum: Input tensor should be 3D");
    ggml_custom1_op_t func = [](struct ggml_tensor * dst, const struct ggml_tensor * a, int ith, int nth, void * userdata) {
        T cum = 0;
        int idx = 0;
        int elements_per_row = a->ne[0];
        auto op = [&cum, &idx, elements_per_row](T src) {
            if (idx == elements_per_row) {
                cum = 0;
                idx = 0;
            }
            cum += src;
            idx += 1;
            return cum;
        };
        custom_op<T>(dst, a, ith, nth, op);
    };

    return ggml_map_custom1(
            ctx,
            tensor,
            func,
            1,
            nullptr
    );
}

template <class T> struct ggml_tensor* max_element_impl(struct ggml_context* ctx, struct ggml_tensor* tensor) {
    ggml_custom1_op_t func = [](struct ggml_tensor * dst, const struct ggml_tensor * src, int ith, int nth, void * userdata) {
        T current_max = (T) std::numeric_limits<float>::min();
        auto op = [&current_max](T x) {
            return x > current_max ? x : current_max;
        };
        custom_op<T>(dst, src, ith, nth, op);
    };

    auto max = ggml_map_custom1(
            ctx,
            tensor,
            func,
            1,
            nullptr
    );

    max = ggml_cont(ctx, max);
    return ggml_view_1d(ctx, max, 1, (ggml_nelements(max)-1) * ggml_element_size(max));
}

template<class T> struct ggml_tensor* repeat_impl(struct ggml_context* ctx, struct ggml_allocr* allocr, struct ggml_tensor* tensor, int64_t new_dim_size, int across) {
    ASSERT(tensor->n_dims == 1, "Only 1d tensors supported");
    ASSERT(across == 0 || across == 1, "Only across == 0 || 1 supported");
    std::vector<int64_t> shape = {across == 0 ? new_dim_size : tensor->ne[0], across == 1 ? new_dim_size : tensor->ne[0]};
    auto new_tensor = tensor_zeros(ctx, allocr, shape);
    ggml_custom2_op_t func = [](struct ggml_tensor * dst, const struct ggml_tensor * _, const struct ggml_tensor * src1, int ith, int nth, void * userdata) {
        START_BENCH()
        auto across = *get_temp_data<int>(userdata);
        auto* dst_ptr = (T*)dst->data;
        auto* src1_ptr = (T*)src1->data;

        size_t size = ggml_element_size(dst);
        for_each_element_threaded(dst, ith, nth, [&] (int i, int j, int k) {
            auto idx1 = ((across == 0 ? j : i) * src1->nb[0]) / size;
            auto dst_idx = (i * dst->nb[0] + j * dst->nb[1]) / size;

            dst_ptr[dst_idx] = src1_ptr[idx1];
        });
        PRINT_BENCH("repeat")
    };
    auto userdata = allocate_temp_i32_array({across});
    // inplace breaks?
    auto result = ggml_map_custom2(
            ctx,
            new_tensor,
            tensor,
            func,
            GGML_N_TASKS_MAX,
            userdata
    );
    return cleanup(ctx, result, userdata);
}

template<class T> struct ggml_tensor* compare_impl(struct ggml_context* ctx, struct ggml_tensor* a, struct ggml_tensor* b, std::function<bool(float, float)> compare_op) {
    ASSERT(a->n_dims == b->n_dims, "Input tensors should have the same number of dimensions");
    ASSERT(a->ne[a->n_dims-1] == b->ne[b->n_dims-1], "Input tensors should have the same size on the first dimension");

    if (a->ne[0] != b->ne[0] && a->n_dims == 2)
        a = ggml_repeat(ctx, a, b);

    auto compare_op_heap = new std::function<bool(T, T)>(compare_op);
    ggml_custom2_op_t func = [](struct ggml_tensor * dst, const struct ggml_tensor * a, const struct ggml_tensor * b, int ith, int nth, void * userdata) {
        START_BENCH()
        auto& func = *((std::function<bool(T, T)>*) userdata);
        auto op = [&func](T src0, T src1) {
            return (T) (func((float)src0, (float)src1) ? 1.0f : 0.0f);
        };
        custom_op2<T>(dst, a, b, ith, nth, userdata, op);

        delete (std::function<bool(T, T)>*) userdata;
        PRINT_BENCH("compare")
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

template<class T> struct ggml_tensor* set_inplace_impl(struct ggml_context* ctx, struct ggml_tensor* tensor, struct ggml_tensor* values, int start0, int start1, int start2) {
    ASSERT(tensor->n_dims == 3, "Only support 3d tensors");
    ASSERT(values->n_dims == 3, "Only support 3d tensors");
    ASSERT(start0 + values->ne[0] <= tensor->ne[0], "Invalid start0 index");
    ASSERT(start1 + values->ne[1] <= tensor->ne[1], "Invalid start1 index");
    ASSERT(start2 + values->ne[2] <= tensor->ne[2], "Invalid start2 index");
    ASSERT(ggml_is_contiguous(tensor), "Input tensor should be contiguous");
    ASSERT(ggml_is_contiguous(values), "Values tensor should be contiguous");

    //USE MEMCPY HERE
    ggml_custom2_op_t func = [](struct ggml_tensor * dst, const struct ggml_tensor * src0, const struct ggml_tensor * src1, int ith, int nth, void * userdata) {
        START_BENCH()
        auto starts = get_temp_data<int>(userdata);
        auto* dst_ptr = (T*)dst->data;
        auto* src1_ptr = (T*)src1->data;

        size_t size = ggml_element_size(src0);
        auto total_elements = ggml_nelements(dst);
        auto idx_offset_b = (starts[0] * src1->nb[0] + starts[1] * src1->nb[1] + starts[2] * src1->nb[2]);
        for_each_element_threaded(src1, ith, nth, [&](int i, int j, int k) {
            auto idx1 = (i * src1->nb[0] + j * src1->nb[1] + k * src1->nb[2]) / size;
            auto dst_idx = (idx_offset_b + i * dst->nb[0] + j * dst->nb[1] + k * dst->nb[2]) / size;
            dst_ptr[dst_idx] = src1_ptr[idx1];
        });
        /*
         auto total_elements = ggml_nelements(dst);
        auto part_size = total_elements / nth;
        auto offset = part_size * ith;
        for (size_t w = offset; w < offset + part_size; w++) {
            int i = 0, j = 0, k = 0;
            compute_indices(dst, w, i, j, k);

            ASSERT(i < dst->ne[0] && j < dst->ne[1] && k < dst->ne[2], "Invalid index");

            callback(i, j, k);
        }
         * */
        PRINT_BENCH("set_inplace")
    };
    auto userdata = allocate_temp_i32_array({start0, start1, start2});
    auto result = ggml_map_custom2_inplace(
            ctx,
            tensor,
            values,
            func,
            1,
            userdata
    );
    return cleanup(ctx, result, userdata);
}

template<class T> struct ggml_tensor* add_bias_inplace_impl(struct ggml_context* ctx, struct ggml_tensor* tensor, struct ggml_tensor* bias) {
    ASSERT(tensor->n_dims == 3, "Only support 3d tensors");
    ASSERT(tensor->ne[1] == bias->ne[0], "Can only broadcast add across second dimension");
    ASSERT(bias->n_dims == 1, "Only support 1d bias tensors");
    ASSERT(ggml_is_contiguous(tensor), "Input tensor should be contiguous");
    ggml_custom2_op_t func = [](struct ggml_tensor * dst, const struct ggml_tensor * src0, const struct ggml_tensor * src1, int ith, int nth, void * userdata) {
        START_BENCH()
        auto* dst_ptr = (T*)dst->data;
        auto* src0_ptr = (T *)src0->data;
        auto* bias_ptr = (T*)src1->data;
        auto size = ggml_element_size(dst);
        auto bias_size = ggml_element_size(src1);
        ASSERT(src0_ptr == dst_ptr, "src0 and dst should be same tensors");

        auto total_elements = ggml_nelements(dst);
        auto part_size = total_elements / nth;
        auto offset = part_size * ith;
        auto ne0 = src0->ne[0];
        for (size_t w = offset; w < offset + part_size; w++) {
            int j = w / ne0;
            dst_ptr[w] = src0_ptr[w] + *(bias_ptr + j);
        };
        PRINT_BENCH("add_bias")
    };
    return ggml_map_custom2_inplace(
            ctx,
            tensor,
            bias,
            func,
            GGML_N_TASKS_MAX,
            nullptr
    );
}

void execute_conv1d_fp16(struct ggml_tensor * dst, const struct ggml_tensor * inputs, const struct ggml_tensor * weights, int ith, int nth, void * userdata) {
    auto* dst_ptr = (ggml_fp16_t*)dst->data;
    const auto* src0_ptr = (ggml_fp16_t*)inputs->data;
    const auto* weights_ptr = (ggml_fp16_t*)weights->data;

    auto input_size = ggml_element_size(inputs);
    auto weights_size = ggml_element_size(weights);
    auto work_size = ggml_element_size(dst);

    auto batch_size = inputs->ne[2];

    auto storage_ptr = (int64_t*) userdata;
    auto stride = storage_ptr[0];
    auto padding = storage_ptr[1];
    auto dilation = storage_ptr[2];
    auto output_size = storage_ptr[3];
    auto* work_data_ptr = (ggml_fp16_t*)storage_ptr[4];

    auto kernel_size = weights->ne[0];
    auto in_length = inputs->ne[0];
    auto channel_count = inputs->ne[1];

    ASSERT(dst_ptr == src0_ptr, "dst and src0 should be same tensors");
    ASSERT(batch_size == 1, "Only batch size 1 supported");
    ASSERT(stride == 1, "Only stride 1 supported");

    auto traits = ggml_internal_get_type_traits(inputs->type);

    const int max_kernel_size = 16;
    const int max_channel_count = 768;
    const int max_buffer_size = max_channel_count * max_kernel_size;
    ASSERT(kernel_size <= max_kernel_size, "Kernel size too big");
    ASSERT(channel_count <= max_buffer_size, "Channel count too big");

    ggml_fp16_t input_buffer[max_buffer_size];
    ggml_fp16_t kernel_buffer[max_buffer_size];
    auto start = std::chrono::high_resolution_clock::now();
//    printf("conv1d stride: %d channels: %d, kernel_size: %d, in_+length %d\n", stride, channel_count, kernel_size, in_length);
    size_t w = 0;
    for (int co = 0; co < channel_count; ++co) {
        for (int i = 0; i < output_size; i++) {
            float sum = 0;
            int n = 0;
            for (int ci = 0; ci < channel_count; ++ci) {
                for (int j = 0; j < kernel_size; j += dilation) {
                    auto input_index = i - padding + j;

                    if (0 <= input_index && input_index < in_length) {
                        auto input_tensor_idx = (input_index * inputs->nb[0] + ci * inputs->nb[1]) / input_size;
                        auto weights_idx = (j * weights->nb[0] + ci * weights->nb[1] + co * weights->nb[2]) / weights_size;

                        sum += src0_ptr[input_tensor_idx] * weights_ptr[weights_idx];
                        input_buffer[n] = src0_ptr[input_tensor_idx];
                        kernel_buffer[n] = weights_ptr[weights_idx];
                        n++;
                        w++;
                    }
                }
            }
            traits.vec_dot(n, &sum, input_buffer, kernel_buffer);
            auto output_idx = (i * dst->nb[0] + co * dst->nb[1]) / work_size;
            work_data_ptr[output_idx] = (ggml_fp16_t) sum;
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto delta = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    printf("Done total: %f million elements. Took %lld ms\n", (w / 1e6), delta);
};

struct ggml_tensor* conv_1d_inplace_impl_fp16(struct ggml_context* ctx, struct ggml_tensor* inputs, struct ggml_tensor* weights, int stride = 1, int padding = 0, int dilation= 1) {
    ASSERT(weights->n_dims == 3, "Only support 3d tensors");
    ASSERT(weights->type == GGML_TYPE_F16, "Only support f16 tensors");
    ASSERT(inputs->type == GGML_TYPE_F16, "Only support f16 tensors");
    ASSERT(ggml_is_contiguous(inputs), "Input tensor should be contiguous");
    ASSERT(ggml_is_contiguous(weights), "Input tensor should be contiguous");
    auto kernel_size = weights->ne[0];
    auto w_in_channels = weights->ne[1];
    auto out_channels = weights->ne[2];

    auto in_length = inputs->ne[0];
    auto i_in_channels = inputs->ne[1];
    auto batch_size = inputs->ne[2];

    ASSERT(w_in_channels == i_in_channels, "Input channels should match filters");
    ASSERT(w_in_channels == out_channels, "Only support same number of channels for conv1d");

    // output_size = ((input_size - dilation * (kernel_size - 1) - 1) // stride) + 1
    auto output_size = ((in_length - dilation * (kernel_size - 1) - 1) / stride) + 1;
    //printf("output_size: %d in_length: %d\n", output_size, in_length);
    ASSERT(output_size <= in_length, "Invalid output size");
    auto work_data = new ggml_fp16_t[ggml_nelements(inputs)];
    memset(work_data, 0, ggml_nelements(inputs) * sizeof(ggml_fp16_t));
    auto storage = new int64_t[5] {stride, padding, dilation, output_size, (int64_t)work_data};

    auto conv1d_2_work_data = ggml_map_custom2_inplace(
            ctx,
            inputs,
            weights,
            execute_conv1d_fp16,
            1,
            storage
    );

    ggml_custom1_op_t copy_func = [](struct ggml_tensor * dst, const struct ggml_tensor * src, int ith, int nth, void * userdata) {
        auto* dst_ptr = (ggml_fp16_t*)dst->data;
        auto* src_ptr = (ggml_fp16_t*)src->data;
        ASSERT(src_ptr == dst_ptr, "src and dst should be same tensors");
        auto work_data = (ggml_fp16_t*) userdata;
        auto work_size = ggml_element_size(dst);
        for_each_element_threaded(dst, ith, nth, [&](int i, int j, int k) {
            auto dst_idx = (i * dst->nb[0] + j * dst->nb[1] + k * dst->nb[2]) / work_size;
            dst_ptr[dst_idx] = work_data[dst_idx];
        });
    };

    auto result = ggml_map_custom1_inplace(ctx, conv1d_2_work_data, copy_func, 1, work_data);

    ggml_custom1_op_t cleanup_func = [](struct ggml_tensor * dst, const struct ggml_tensor * src, int ith, int nth, void * userdata) {
        auto storage = (int64_t*) userdata;
        auto work_data = (ggml_fp16_t*) storage[4];
        delete[] storage;
        delete[] work_data;
    };

    return ggml_map_custom1_inplace(ctx, result, cleanup_func, 1, storage);
}
/*
<3, 1 256>
<3, 3, 256>
<3, 5, 256>

<7, 1, 256>
<7, 1, 256>
<7, 1, 256>

<11, 1, 256>
<11, 1, 256>
<11, 1, 256>

<3, 1 128>
<3, 3, 128>
<3, 5, 128>

<7, 1, 128>
<7, 3, 128>
<7, 5, 128>

<11, 1, 128>
<11, 3, 128>
<11, 5, 128>
*/

#include <arm_neon.h>


void im2col_multi_channel(float * dst_data, const float* src_data, int num_channels, int input_length, int output_length, int kernel_size, int stride, int padding, int dilation, int ith, int nth) {
    // Precompute constants that are invariant across the inner loops
    int stride_times_dilation = stride * dilation;
    int input_length_times_num_channels = input_length * num_channels;

    for (int c = 0; c < num_channels; ++c) {
        int channel_base_index = c * input_length;
        int channel_end_index = channel_base_index + input_length;

        for (int j = 0; j < kernel_size; ++j) {
            int dilation_offset = j * dilation - padding;

            for (int i = 0; i < output_length; ++i) {
                int src_index = channel_base_index + i * stride_times_dilation + dilation_offset;
                int dst_index = (c * kernel_size + j) * output_length + i;

                // Check bounds only once per loop iteration
                if (src_index >= channel_base_index && src_index < channel_end_index) {
                    dst_data[dst_index] = src_data[src_index];
                } else {
                    dst_data[dst_index] = 0;
                }
            }
        }
    }
}

struct ggml_tensor* im2col_impl(struct ggml_context* ctx, struct ggml_tensor* weights, struct ggml_tensor* inputs, int stride = 1, int padding = 0, int dilation= 1) {
    int32_t kernel_size = weights->ne[0];
    int32_t w_in_channels = weights->ne[1];
    int32_t out_channels = weights->ne[2];

    int32_t in_length = inputs->ne[0];
    int32_t i_in_channels = inputs->ne[1];
    int32_t batch_size = inputs->ne[2];

    int32_t output_columns = ((in_length + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;
    auto dst = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, output_columns, i_in_channels * kernel_size,  1, 1);
    //memset(dst->data, 0, ggml_nbytes(dst));

    //printf("Conv1d with kernel_size = %d, dilation = %d, padding = %d, channels = %d \n", kernel_size, dilation, padding, i_in_channels);

    auto func = [](struct ggml_tensor* dst, const struct ggml_tensor* _, const struct ggml_tensor* inputs, int ith, int nth, void* userdata) {
        auto dst_ptr = (float*)dst->data;
        auto inputs_ptr = (float*)inputs->data;
        auto storage = get_temp_data<int32_t>(userdata);
        auto stride = storage[0];
        auto padding = storage[1];
        auto dilation = storage[2];
        auto output_columns = storage[3];
        auto kernel_size = storage[4];
        auto in_length = storage[5];
        auto channel_count = storage[6];

        im2col_multi_channel(dst_ptr, inputs_ptr, channel_count, in_length, output_columns, kernel_size, stride, padding, dilation, ith, nth);
    };

    auto userdata = allocate_temp_i32_array({stride, padding, dilation, output_columns, kernel_size, in_length, w_in_channels});
    auto result = ggml_map_custom2_inplace(
            ctx,
            dst,
            inputs,
            func,
            GGML_N_TASKS_MAX,
            userdata
    );
    result = cleanup(ctx, result, userdata);
    result = ggml_permute(ctx, result, 1, 0, 2, 3);
    result = ggml_cont(ctx, result);
    return result;
}

struct ggml_tensor* conv1d_impl(struct ggml_context* ctx, struct ggml_tensor* weights, struct ggml_tensor* inputs, int stride = 1, int padding = 0, int dilation= 1) {
    ASSERT(weights->type == GGML_TYPE_F32, "conv1d: only support f16 tensors");
    ASSERT(inputs->type == GGML_TYPE_F32, "conv1d: only support f32 tensors");

    struct ggml_tensor * im2col = ggml_im2col_1d(ctx, weights, inputs, stride, padding, dilation);
    auto ic_by_k = (weights->ne[0] * weights->ne[1]);

    struct ggml_tensor * result =
            ggml_mul_mat(ctx,
                         ggml_view_2d(ctx, im2col, im2col->ne[0], (im2col->ne[2] * im2col->ne[1]), ggml_element_size(im2col) * im2col->ne[0], 0), // [N, OL, IC * K] => [N*OL, IC * K]
                         ggml_view_2d(ctx, weights, ic_by_k, weights->ne[2], ic_by_k * ggml_element_size(im2col), 0)); // [OC，IC, K] => [OC, IC * K]

    // ggml_element_size(im2col) * im2col->ne[1], ggml_element_size(im2col) * im2col->ne[1] * weights->ne[2], 0
    return ggml_reshape_3d(ctx, result, im2col->ne[1], weights->ne[2], im2col->ne[2]); // [N, OC, OL]
}

void add_fast(float * dst_data, const float* src0_data, const float* src1_data, size_t n, int ith, int nth) {
    int part_size = n / nth;
    int offset = ith * part_size;
    #pragma clang loop vectorize(enable)
    for (int i = offset; i < offset + part_size; i++) {
        dst_data[i] = src0_data[i] + src1_data[i];
    }
}

struct ggml_tensor* add_impl(struct ggml_context* ctx, struct ggml_tensor* inputs, struct ggml_tensor* bias) {
    ASSERT(bias->type == GGML_TYPE_F32, "add: only support f32 tensors");
    ASSERT(inputs->type == GGML_TYPE_F32, "add: only support f32 tensors");
    ASSERT(ggml_nelements(inputs) == ggml_nelements(bias), "add: should have same elements");

    auto func = [](struct ggml_tensor* dst, const struct ggml_tensor* src0, const struct ggml_tensor* src1, int ith, int nth, void* userdata) {
        auto dst_ptr = (float*)dst->data;
        auto src0_ptr = (float*)src0->data;
        auto src1_ptr = (float*)src1->data;
        add_fast(dst_ptr, src0_ptr, src1_ptr, ggml_nelements(dst), ith, nth);
    };

    return ggml_map_custom2(
            ctx,
            inputs,
            bias,
            func,
            GGML_N_TASKS_MAX,
            nullptr
    );
}


struct ggml_tensor* broadcast_if_possible(struct ggml_context* ctx, struct ggml_tensor* tensor, struct ggml_tensor* mask) {
    if (mask->ne[0] == tensor->ne[1] && mask->ne[1] == 1) {
        mask = ggml_permute(ctx, mask, 1, 0, 2, 3);
        mask = ggml_cont(ctx, mask);
        mask = ggml_repeat(ctx, mask, tensor);
    }
    return mask;
}

template<class T> struct ggml_tensor* masked_get_impl(struct ggml_context* ctx, struct ggml_tensor* tensor, struct ggml_tensor* mask) {
    ASSERT(tensor->n_dims == 3, "Only support 3d tensors");
    ASSERT(mask->n_dims == 3, "Only support 3d tensors");
    mask = broadcast_if_possible(ctx, tensor, mask);
    ASSERT(ggml_nelements(tensor) == ggml_nelements(mask), "masked_get: should have same elements");

    ggml_custom2_op_t func = [](struct ggml_tensor * dst, const struct ggml_tensor * a, const struct ggml_tensor * b, int ith, int nth, void * userdata) {
        auto op = [](T src0, T src1) {
            ASSERT(fabs(src1) < 1e-5f || fabs(src1 - 1) < 1e-5f, "Mask tensor should be 0 or 1");
            return ((int)src1) == 1 ? src0 : 0;
        };
        START_BENCH()
        custom_op2<T>(dst, a, b, ith, nth, userdata, op);
        PRINT_BENCH("masked_get")
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

template<class T> struct ggml_tensor* gather_impl(struct ggml_context* ctx, struct ggml_tensor* tensor, int dim, struct ggml_tensor* index) {
    ASSERT(tensor->n_dims == 3, "Only support 3d tensors");
    ASSERT(index->n_dims == 1, "Only support 1d tensors for the index");
    ASSERT(dim == 0, "Only dim == 0 supported");

    ggml_custom2_op_t func = [](struct ggml_tensor * dst, const struct ggml_tensor * index_tensor, const struct ggml_tensor * values_tensor, int ith, int nth, void * userdata) {
        auto values_ptr = (T*) values_tensor->data;
        ASSERT(dst->n_dims == 1, "Only support 1d tensors");
        ASSERT(dst->ne[0] == index_tensor->ne[0], "Index and output mismatch");
        START_BENCH()
        for(int i = 0; i < index_tensor->ne[0]; ++i)
        {
            int j = (int) ((T*) index_tensor->data)[i];
            int index = j + i * values_tensor->ne[0];
            ASSERT(index < ggml_nelements(values_tensor) && index > 0, "Index should be smaller than the number of elements in the value tensor");
            ((T*)dst->data)[i] = values_ptr[index];
        }
        PRINT_BENCH("gather")
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

template<class T> struct ggml_tensor* masked_set_impl(struct ggml_context* ctx, struct ggml_tensor* tensor, struct ggml_tensor* mask, struct ggml_tensor* value) {
    ASSERT(tensor->n_dims == 3, "Only support 3d tensors");
    ASSERT(mask->n_dims == 3, "Only support 3d tensors");
    mask = broadcast_if_possible(ctx, tensor, mask);
    ASSERT(ggml_nelements(tensor) == ggml_nelements(mask) , "masked_set: tensor and mask should have same elements");
    ASSERT(ggml_nelements(mask) == ggml_nelements(value), "masked_set: mask and values should have same elements");

    ggml_custom3_op_t func = [](struct ggml_tensor * dst, const struct ggml_tensor * a, const struct ggml_tensor * b, const struct ggml_tensor * c, int ith, int nth, void * userdata) {
        int index = 0;
        auto value_tensor = c;
        float* values = ggml_get_data_f32(value_tensor);
        auto op = [&index, &values, value_tensor](T src0, T src1) {
            ASSERT(abs(src1) < 1e-5f || abs(src1 - 1) < 1e-5f, "Mask tensor should be 0 or 1");
            ASSERT(index < ggml_nelements(value_tensor), "Index should be smaller than the number of elements in the value tensor");
            return ((int)src1) == 1 ? values[index++] : src0;
        };
        custom_op2<T>(dst, a, b, ith, nth, userdata, op);
    };
    return ggml_map_custom3(
            ctx,
            tensor,
            mask,
            value,
            func,
            1,
            nullptr
    );
}

template<class T> T sigmoid(T src) {
    return (T) (1.0f / (1.0f + std::exp(-(float)src)));
}

template<class T> T exp(T src) {
    return (T) std::exp((float)src);
}

template<class T> T softplus(T x) {
    const float beta = 1.0;
    const float threshold = 20;
    const auto xf = (float)x;
    if (x > threshold)
        return (T) (beta * x);
    return (T) (1.0f / beta * std::log(1.0 + std::exp(beta * xf)));
}

template<class T> T ceil(T src) {
    return (T) std::ceil((float)src);
}

template<class T> T binary_not(T x) {
    ASSERT(abs(x) < 1e-5f || abs(x - 1) < 1e-5f, "Input tensor should be 0 or 1");
    return ((int)x) == 0 ? 1.0f : 0.0f;
}

template<class T> T pow(T src, float to) {
    return (T)std::pow((float)src, to);
}

template<class T> T leaky_relu(T src, float slope) {
    return (T)((src > 0) ? src : src * slope);
}

UNARY_OP(sigmoid)

UNARY_OP(exp)

UNARY_OP(softplus)

UNARY_OP(ceil)

UNARY_OP(binary_not)

UNARY_OP_PARAM(pow)

UNARY_OP_PARAM(leaky_relu)


struct ggml_tensor* flip_3d(struct ggml_context* ctx, struct ggml_tensor* tensor, int along) {
    TENSOR_OP_IMPL(flip_3d, tensor, ctx, tensor, along);
}

struct ggml_tensor* tensor_max(struct ggml_context* ctx, struct ggml_tensor* tensor) {
    TENSOR_OP_IMPL(max_element, tensor, ctx, tensor);
}

struct ggml_tensor* tensor_per_row_cumsum(struct ggml_context* ctx, struct ggml_tensor* tensor) {
    TENSOR_OP_IMPL(per_row_cumsum, tensor, ctx, tensor);
}

struct ggml_tensor* tensor_repeat(struct ggml_context* ctx, struct ggml_allocr* allocr, struct ggml_tensor* tensor, size_t new_dim_size, int across) {
    TENSOR_OP_IMPL(repeat, tensor, ctx, allocr, tensor, new_dim_size, across);
}

struct ggml_tensor* tensor_compare(struct ggml_context* ctx, struct ggml_tensor* a, struct ggml_tensor* b, std::function<bool(float, float)> compare_op) {
    TENSOR_OP_IMPL(compare, a, ctx, a, b, compare_op);
}

struct ggml_tensor* tensor_masked_get(struct ggml_context* ctx, struct ggml_tensor* tensor, struct ggml_tensor* mask) {
    TENSOR_OP_IMPL(masked_get, tensor, ctx, tensor, mask);
}

struct ggml_tensor* tensor_masked_set(struct ggml_context* ctx, struct ggml_tensor* tensor, struct ggml_tensor* mask, struct ggml_tensor* value) {
    TENSOR_OP_IMPL(masked_set, tensor, ctx, tensor, mask, value);
}

struct ggml_tensor* tensor_gather(struct ggml_context* ctx, struct ggml_tensor* tensor, int dim, struct ggml_tensor* index) {
    TENSOR_OP_IMPL(gather, tensor, ctx, tensor, dim, index);
}

struct ggml_tensor* tensor_set_inplace(struct ggml_context* ctx, struct ggml_tensor* tensor, struct ggml_tensor* values, int start0, int start1, int start2) {
    TENSOR_OP_IMPL(set_inplace, tensor, ctx, tensor, values, start0, start1, start2);
}

struct ggml_tensor* tensor_add_bias_inplace(struct ggml_context* ctx, struct ggml_tensor* tensor, struct ggml_tensor* bias) {
    TENSOR_OP_IMPL(add_bias_inplace, tensor, ctx, tensor, bias);
}

struct ggml_tensor* tensor_conv_1d_inplace(struct ggml_context* ctx,  struct ggml_tensor* inputs, struct ggml_tensor* weights, int stride = 1, int padding = 0, int dilation= 1) {
    return conv_1d_inplace_impl_fp16(ctx, inputs, weights, stride, padding, dilation);
}

struct ggml_tensor* tensor_conv_1d(struct ggml_context* ctx, struct ggml_tensor* inputs, struct ggml_tensor* weights,  int stride = 1, int padding = 0, int dilation= 1) {
    return conv1d_impl(ctx, weights, inputs, stride, padding, dilation);
}

struct ggml_tensor* tensor_add_fast(struct ggml_context* ctx, struct ggml_tensor* inputs, struct ggml_tensor* bias) {
    return add_impl(ctx, inputs, bias);
}

struct ggml_tensor* tensor_expand(struct ggml_context* ctx, struct ggml_tensor* tensor, int stride, int dim) {
    ASSERT(tensor->n_dims == 3, "Only support 3d tensors");
    ASSERT(dim == 0, "Only support dim 0");
    auto new_tensor = ggml_new_tensor_3d(ctx, tensor->type, tensor->ne[0] * stride, tensor->ne[1], tensor->ne[2]);
    auto storage = new int[3]{stride, 1, 1};
    ggml_custom2_op_t func = [](struct ggml_tensor * dst, const struct ggml_tensor * src0, const struct ggml_tensor * src1, int ith, int nth, void * userdata) {
        START_BENCH()
        auto strides = ((int*)userdata);
        ASSERT(src0->type == GGML_TYPE_F32, "Input tensor should be fp32");
        ASSERT(src1->type == GGML_TYPE_F32, "Input tensor should be fp32");
        ASSERT(dst->type == GGML_TYPE_F32, "Output tensor should be fp32");
        auto* dst_ptr = (float*)dst->data;
        auto* src1_ptr = (float*)src1->data;

        size_t size = ggml_element_size(src0);
        for (int i = 0; i < src1->ne[0]; ++i) {
            for (int j = 0; j < src1->ne[1]; ++j) {
                for (int k = 0; k < src1->ne[2]; ++k) {
                    auto idx1 = (i * src1->nb[0] + j * src1->nb[1] + k * src1->nb[2]) / size;
                    auto dst_idx = ((strides[0] * i) * dst->nb[0] + (strides[1] * j) * dst->nb[1] + (strides[2] * k) * dst->nb[2]) / size;

                    dst_ptr[dst_idx] = src1_ptr[idx1];
                }
            }
        }
        delete[] (int*)userdata;
        PRINT_BENCH("expand")
    };
    return ggml_map_custom2_inplace(
            ctx,
            new_tensor,
            tensor,
            func,
            1,
            storage
    );
}

#endif //VITS_CUSTOM_OPS_H
