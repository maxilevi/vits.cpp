//
// Created by Maximiliano Levi on 08/10/2023.
//

#ifndef VITS_CPP_GGML_UTIL_H
#define VITS_CPP_GGML_UTIL_H


struct ggml_tensor* pad_3d(struct ggml_context* ctx, struct ggml_tensor* tensor, std::vector<int> pads) {
    ASSERT(tensor->n_dims == 3, "Input tensor should be 3D");
    ASSERT(pads.size() == 6, "Invalid pad count");

    // Compute the new shape
    int64_t new_shape[GGML_MAX_DIMS];
    for(int i = 0; i < 3; i++) {
        int reverse_index = (tensor->n_dims - i - 1) * 2;
        new_shape[i] = tensor->ne[i] + pads[reverse_index] + pads[reverse_index+ 1];
    }

    // Create a new tensor with the computed shape
    auto cur = ggml_new_tensor(ctx, tensor->type, tensor->n_dims, new_shape);

    // Compute the byte offset and strides for the view based on element size
    size_t element_size = ggml_element_size(tensor);  // Assuming such a function exists
    size_t offset = pads[0] * tensor->ne[1] * tensor->ne[2] * element_size + pads[2] * tensor->ne[2] * element_size + pads[4] * element_size;
    size_t row_stride = (pads[4] + tensor->ne[2] + pads[5]) * element_size;
    size_t slice_stride = (pads[2] + tensor->ne[1] + pads[3]) * row_stride;

    // Create a 3D view on the padded tensor
    auto unpadded_view = ggml_view_3d(ctx, cur, tensor->ne[0], tensor->ne[1], tensor->ne[2], row_stride, slice_stride, offset);

    // Copy the data from the original tensor into the view
    ggml_cpy_inplace(ctx, unpadded_view, tensor);


    return cur;
}

// FIX TODO THIS IS CAN BE SIMPLIFIED ON A GENERIC PAD
struct ggml_tensor* pad_2d(struct ggml_context* ctx, struct ggml_tensor* tensor, std::vector<int> pads) {
    // Assure the correct number of pads
    ASSERT(tensor->n_dims == pads.size() / 2, "Invalid pad count");

    // Compute the new shape
    int64_t new_shape[GGML_MAX_DIMS];
    for(int i = 0; i < tensor->n_dims; i++) {
        int reverse_index = (tensor->n_dims - i - 1) * 2;
        new_shape[i] = tensor->ne[i] + pads[reverse_index] + pads[reverse_index + 1];
    }

    // Create a new tensor with the computed shape
    auto cur = ggml_new_tensor(ctx, tensor->type, tensor->n_dims, new_shape);

    // Compute the byte offset and stride for the view based on element size
    size_t element_size = ggml_element_size(tensor);
    size_t offset = pads[0] * element_size; // assuming row-major order
    size_t row_stride = (pads[2] + tensor->ne[1] + pads[3]) * element_size;

    // Create a 2D view on the padded tensor
    auto unpadded_view = ggml_view_2d(ctx, cur, tensor->ne[0], tensor->ne[1], row_stride, offset);

    // Copy the data from the original tensor into the view
    ggml_cpy_inplace(ctx, unpadded_view, tensor);


    return cur;
}

struct ggml_tensor* slice_3d(struct ggml_context* ctx, struct ggml_tensor* tensor,
                             int start0, int end0,
                             int start1, int end1,
                             int start2, int end2) {
    ASSERT(tensor->n_dims == 3, "Input tensor should be 3D");

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

    // Compute the new shape
    int64_t new_shape[3] = { end0 - start0, end1 - start1, end2 - start2 };

    // Compute the byte offset and strides for the view based on element size
    size_t element_size = ggml_element_size(tensor);  // Assuming such a function exists
    size_t offset = start0 * tensor->ne[1] * tensor->ne[2] * element_size + start1 * tensor->ne[2] * element_size + start2 * element_size;
    size_t row_stride = tensor->ne[2] * element_size;
    size_t slice_stride = tensor->ne[1] * row_stride;

    // Create a 3D view on the original tensor
    auto sliced_view = ggml_view_3d(ctx, tensor, new_shape[0], new_shape[1], new_shape[2], row_stride, slice_stride, offset);

    return sliced_view;
}

struct ggml_tensor* slice_2d(struct ggml_context* ctx, struct ggml_tensor* tensor, int start0, int end0, int start1, int end1) {
    return slice_3d(ctx, tensor, start0, end0, start1, end1, 0, -1);
}


struct ggml_tensor* batched_mul_mat(struct ggml_context* ctx, struct ggml_tensor* a, struct ggml_tensor* b) {
    ASSERT(a->n_dims == 3 && b->n_dims == 3, "Both tensors must be 3D");
    ASSERT(a->ne[2] == b->ne[2], "The batch size (last dimension) should be the same for both tensors");

    int64_t batch_size = a->ne[2];
    struct ggml_tensor* result = nullptr;

    for (int64_t i = 0; i < batch_size; ++i) {
        // Creating a view for the i-th matrices from a and b
        struct ggml_tensor* a_i = ggml_view_2d(ctx, a, a->ne[0], a->ne[1], a->ne[1] * ggml_element_size(a), i * a->ne[0] * a->ne[1] * ggml_element_size(a));
        struct ggml_tensor* b_i = ggml_view_2d(ctx, b, b->ne[0], b->ne[1], b->ne[1] * ggml_element_size(b), i * b->ne[0] * b->ne[1] * ggml_element_size(b));

        // Multiplying the i-th matrices
        struct ggml_tensor* r_i = ggml_mul_mat(ctx, a_i, b_i);

        // Concatenating result along the last dimension
        if (result == nullptr) {
            result = r_i;
        } else {
            struct ggml_tensor* new_result = ggml_concat(ctx, result, r_i);
            result = new_result;
        }
    }
    return result;
}



#endif //VITS_CPP_GGML_UTIL_H
