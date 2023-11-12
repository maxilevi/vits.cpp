#include <ggml/ggml.h>
#include <ggml-util.h>
#include <vector>
#include <thread>
#include <debug.h>
#include <stdint.h>

void print_tensor(struct ggml_tensor* tensor, std::string name) {
    printf("%s: [", name.c_str());
    auto ne = tensor->ne;
    auto nb0 = tensor->nb[0];
    auto nb1 = tensor->nb[1];
    auto nb2 = tensor->nb[2];
    auto data = static_cast<float*>(tensor->data);

    for (int64_t i = 0; i < ne[2]; ++i) {
        std::cout << "Slice " << i << ":\n";
        for (int64_t j = 0; j < ne[1]; ++j) {
            for (int64_t k = 0; k < ne[0]; ++k) {
                // Calculate offset and print element
                size_t offset = (k * nb0 + j * nb1 + i * nb2) / sizeof(float);
                std::cout << *(data + offset) << " ";//<< " (" << offset << ") ";
                //std::cout << *(data + offset) << " ";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }
    printf("]\nData read:\n");
    std::cout << "{";
    for (int64_t i = 0; i < ne[2] * ne[1] * ne[0]; ++i)
        std::cout << " " << ((float*)tensor->data)[i];
    std::cout << "}\n";
}

struct ggml_tensor* execute_tensor(
        struct ggml_context* ctx,
        struct ggml_tensor* tensor
) {
    struct ggml_cgraph graph = {};

    ggml_build_forward_expand(&graph, tensor);
    int threads = std::min((int)std::thread::hardware_concurrency(), 2);
    auto plan = ggml_graph_plan(&graph, threads);
    if (plan.work_size > 0) {
        plan.work_data = (uint8_t*) malloc(plan.work_size);
    }

    ggml_graph_compute(&graph, &plan);

    return tensor;
}

void assert_tensor_matches_expected(
        struct ggml_tensor* tensor,
        const std::vector<int32_t>& expected,
        const std::vector<int32_t>& expected_shape,
        const std::string& op_name
) {
    auto total_elements = ggml_nelements(tensor);
    ASSERT(expected.size() == total_elements, ("Expected shape should match expected data " + std::to_string(expected.size()) + " vs " + std::to_string(total_elements)).c_str());
    for (int i = 0; i < tensor->n_dims; ++i) {
        ASSERT(tensor->ne[i] == expected_shape[i], (op_name + " should have the correct shape").c_str());
    }

    auto data = (float*) ggml_get_data(tensor);
    bool equal = true;
    printf("Data: [");
    auto index = 0;
    for (int64_t i = 0; i < tensor->ne[2]; ++i) {
        for (int64_t j = 0; j < tensor->ne[1]; ++j) {
            for (int64_t k = 0; k < tensor->ne[0]; ++k) {
                size_t offset = (k * tensor->nb[0] + j * tensor->nb[1] + i * tensor->nb[2]) / sizeof(float);
                std::cout << *(data + offset) << " ";//<< " (" << offset << ") [" << expected[index] << "] ";
                auto val = *(data + offset);
                equal &= (val == expected[index++]);
            }
        }
    }
    printf("]\n");
    printf("Expected: [");
    for (int i = 0; i < total_elements; ++i) {
        printf("%d ", expected[i]);
    }
    printf("]\n");

    ASSERT(equal, (op_name + " should be correct").c_str());
}

// Wrapper for padding
void assert_padding_is_correct(
        struct ggml_context* ctx,
        struct ggml_tensor* input_ids_tensor,
        const std::vector<int>& pads,
        const std::vector<int32_t>& expected,
        const std::vector<int32_t>& expected_shape
) {
    std::cout << "Testing Padding" << std::endl;

    struct ggml_tensor* result_tensor = execute_tensor(ctx, pad_3d(ctx, input_ids_tensor, pads));

    assert_tensor_matches_expected(result_tensor, expected, expected_shape, "Padding");

    std::cout << "Padding is correct\n";
}

void assert_slice_is_correct(
        struct ggml_context* ctx,
        struct ggml_tensor* input_ids_tensor,
        const std::vector<int>& slices,
        const std::vector<int32_t>& expected,
        const std::vector<int32_t>& expected_shape
) {
    std::cout << "Testing Slice" << std::endl;

    struct ggml_tensor* result_tensor = execute_tensor(ctx, slice_3d(ctx, input_ids_tensor, slices[0], slices[1], slices[2], slices[3], slices[4], slices[5]));

    assert_tensor_matches_expected(result_tensor, expected, expected_shape, "Slice");

    std::cout << "Slice is correct\n";
}

void assert_flip_is_correct(
        struct ggml_context* ctx,
        struct ggml_tensor* input_ids_tensor,
        int along,
        const std::vector<int32_t>& expected,
        const std::vector<int32_t>& expected_shape
) {
    std::cout << "Testing Flip" << std::endl;

    struct ggml_tensor* result_tensor = execute_tensor(ctx, flip_3d(ctx, input_ids_tensor, along));

    assert_tensor_matches_expected(result_tensor, expected, expected_shape, "Flip");

    std::cout << "Flip is correct\n";
}


void assert_split_is_correct(
        struct ggml_context* ctx,
        struct ggml_tensor* input_ids_tensor,
        int left,
        int right,
        int dim,
        const std::vector<int32_t>& expected_left,
        const std::vector<int32_t>& expected_shape_left,
        const std::vector<int32_t>& expected_right,
        const std::vector<int32_t>& expected_shape_right
) {
    std::cout << "Testing Split" << std::endl;

    auto [left_tensor, right_tensor] = split_3d(ctx, input_ids_tensor, left, right, dim);
    struct ggml_tensor* left_result_tensor = execute_tensor(ctx, left_tensor);
    struct ggml_tensor* right_result_tensor = execute_tensor(ctx, right_tensor);

    assert_tensor_matches_expected(left_result_tensor, expected_left, expected_shape_left, "Split Left");
    assert_tensor_matches_expected(right_result_tensor, expected_right, expected_shape_right, "Split Right");

    std::cout << "Split is correct\n";
}

struct ggml_tensor* assert_concat_is_correct(
        struct ggml_context* ctx,
        struct ggml_tensor* tensor_a,
        struct ggml_tensor* tensor_b,
        int dim,
        const std::vector<int32_t>& expected,
        const std::vector<int32_t>& expected_shape
) {
    std::cout << "Testing Concat" << std::endl;

    struct ggml_tensor* result_tensor = execute_tensor(ctx, concat_3d(ctx, tensor_a, tensor_b, dim));

    assert_tensor_matches_expected(result_tensor, expected, expected_shape, "Concat");

    std::cout << "Concat is correct\n";
    return result_tensor;
}

void assert_cumsum_is_correct(
        struct ggml_context* ctx,
        struct ggml_tensor* tensor,
        const std::vector<int32_t>& expected,
        const std::vector<int32_t>& expected_shape
) {
    std::cout << "Testing Cumsum" << std::endl;

    struct ggml_tensor* result_tensor = execute_tensor(ctx, per_row_cumsum(ctx, tensor));

    assert_tensor_matches_expected(result_tensor, expected, expected_shape, "Cumsum");

    std::cout << "Cumsum is correct\n";
}

void assert_max_is_correct(
        struct ggml_context* ctx,
        struct ggml_tensor* tensor,
        const std::vector<int32_t>& expected,
        const std::vector<int32_t>& expected_shape
) {
    std::cout << "Testing Max" << std::endl;

    struct ggml_tensor* result_tensor = execute_tensor(ctx, tensor_max(ctx, tensor));

    assert_tensor_matches_expected(result_tensor, expected, expected_shape, "Max");

    std::cout << "Max is correct\n";
}

void assert_not_is_correct(
        struct ggml_context* ctx,
        struct ggml_tensor* tensor,
        const std::vector<int32_t>& expected,
        const std::vector<int32_t>& expected_shape
) {
    std::cout << "Testing Not" << std::endl;

    struct ggml_tensor* result_tensor = execute_tensor(ctx, tensor_not(ctx, tensor));

    assert_tensor_matches_expected(result_tensor, expected, expected_shape, "Not");

    std::cout << "Not is correct\n";
}

void assert_compare_is_correct(
        struct ggml_context* ctx,
        struct ggml_tensor* a,
        struct ggml_tensor* b,
        bool(*func)(float, float),
        const std::vector<int32_t>& expected,
        const std::vector<int32_t>& expected_shape
) {
    std::cout << "Testing Compare" << std::endl;

    struct ggml_tensor* result_tensor = execute_tensor(ctx, tensor_compare(ctx, a, b, func));

    assert_tensor_matches_expected(result_tensor, expected, expected_shape, "Compare");

    std::cout << "Compare is correct\n";
}

void assert_arange_is_correct(
        struct ggml_context* ctx,
        int range,
        const std::vector<int32_t>& expected,
        const std::vector<int32_t>& expected_shape
) {
    std::cout << "Testing arange" << std::endl;

    struct ggml_tensor* result_tensor = execute_tensor(ctx, tensor_arange(ctx, range));

    assert_tensor_matches_expected(result_tensor, expected, expected_shape, "arange");

    std::cout << "arange is correct\n";
}

void assert_index_put_last_dim_is_correct(
        struct ggml_context* ctx,
        struct ggml_tensor* tensor,
        int index,
        float value,
        const std::vector<int32_t>& expected,
        const std::vector<int32_t>& expected_shape
) {
    std::cout << "Testing index_put_last_dim" << std::endl;

    struct ggml_tensor* result_tensor = execute_tensor(ctx, index_put_last_dim(ctx, tensor, index, value));

    assert_tensor_matches_expected(result_tensor, expected, expected_shape, "index_put_last_dim");

    std::cout << "index_put_last_dim is correct\n";
}

void assert_index_add_last_dim_is_correct(
        struct ggml_context* ctx,
        struct ggml_tensor* tensor,
        int index,
        float value,
        const std::vector<int32_t>& expected,
        const std::vector<int32_t>& expected_shape
) {
    std::cout << "Testing index_add_last_dim" << std::endl;

    struct ggml_tensor* result_tensor = execute_tensor(ctx, index_add_last_dim(ctx, tensor, index, value));

    assert_tensor_matches_expected(result_tensor, expected, expected_shape, "index_add_last_dim");

    std::cout << "index_add_last_dim is correct\n";
}

void assert_gather_is_correct(
        struct ggml_context* ctx,
        struct ggml_tensor* tensor,
        int dim,
        struct ggml_tensor* index,
        const std::vector<int32_t>& expected,
        const std::vector<int32_t>& expected_shape
) {
    std::cout << "Testing gather" << std::endl;

    struct ggml_tensor* result_tensor = execute_tensor(ctx, tensor_gather(ctx, tensor, dim, index));

    assert_tensor_matches_expected(result_tensor, expected, expected_shape, "gather");

    std::cout << "gather is correct\n";
}

void assert_masked_get_is_correct(
        struct ggml_context* ctx,
        struct ggml_tensor* tensor,
        struct ggml_tensor* mask,
        const std::vector<int32_t>& expected,
        const std::vector<int32_t>& expected_shape
) {
    std::cout << "Testing masked_get" << std::endl;

    struct ggml_tensor* result_tensor = execute_tensor(ctx, tensor_masked_get(ctx, tensor, mask));

    assert_tensor_matches_expected(result_tensor, expected, expected_shape, "masked_get");

    std::cout << "masked_get is correct\n";
}

void assert_masked_set_is_correct(
        struct ggml_context* ctx,
        struct ggml_tensor* tensor,
        struct ggml_tensor* mask,
        struct ggml_tensor* value,
        const std::vector<int32_t>& expected,
        const std::vector<int32_t>& expected_shape
) {
    std::cout << "Testing masked_set" << std::endl;

    struct ggml_tensor* result_tensor = execute_tensor(ctx, tensor_masked_set(ctx, tensor, mask, value));

    assert_tensor_matches_expected(result_tensor, expected, expected_shape, "masked_set");

    std::cout << "masked_set is correct\n";
}

struct ggml_tensor * create_tensor_with_data_and_shape(struct ggml_context * ctx, const std::vector<float>& data, int h, int w, int d) {
    auto tensor = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, h, w, d);
    memcpy(tensor->data, data.data(), ggml_element_size(tensor) * data.size());
    return tensor;
}

struct ggml_tensor * create_tensor_1d_with_data_and_shape(struct ggml_context * ctx, const std::vector<float>& data, int l) {
    auto tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, l);
    memcpy(tensor->data, data.data(), ggml_element_size(tensor) * data.size());
    return tensor;
}


int main(int argc, char ** argv) {
    struct ggml_init_params params = {
            .mem_size   = 256*1024*1024,
            .mem_buffer = nullptr,
    };

    struct ggml_context * ctx = ggml_init(params);

    std::vector<float> input_ids = {1, 2, 3, 4, 5, 6};
    auto input_ids_tensor = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 3, 2, 1);
    memcpy(input_ids_tensor->data, input_ids.data(), ggml_element_size(input_ids_tensor) * input_ids.size());


    assert_padding_is_correct(ctx, input_ids_tensor, {0, 0, 0, 0, 0, 0}, {1, 2, 3, 4, 5, 6}, {3, 2, 1});
    assert_padding_is_correct(ctx, input_ids_tensor, {0, 0, 0, 2, 0, 0}, {1, 2, 3, 4, 5, 6, 0, 0, 0, 0, 0, 0}, {3, 4, 1});
    assert_padding_is_correct(ctx, input_ids_tensor, {0, 0, 2, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6}, {3, 4, 1});
    assert_padding_is_correct(ctx, input_ids_tensor, {0, 0, 2, 1, 0, 0}, {0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 0, 0, 0}, {3, 5, 1});
    assert_padding_is_correct(ctx, input_ids_tensor, {0, 0, 0, 0, 0, 2}, {1, 2, 3, 0, 0, 4, 5, 6, 0, 0}, {5, 2, 1});
    assert_padding_is_correct(ctx, input_ids_tensor, {0, 0, 0, 0, 3, 0}, {0, 0, 0, 1, 2, 3, 0, 0, 0, 4, 5, 6}, {6, 2, 1});
    assert_padding_is_correct(ctx, input_ids_tensor, {1, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6}, {3, 2, 2});
    assert_padding_is_correct(ctx, input_ids_tensor, {0, 1, 0, 0, 0, 0}, {1, 2, 3, 4, 5, 6, 0, 0, 0, 0, 0, 0}, {3, 2, 2});

    // Slice

    assert_slice_is_correct(ctx, input_ids_tensor, {0, -1, 0, -1, 0, -1}, {1, 2, 3, 4, 5, 6}, {3, 2, 1});

    assert_slice_is_correct(ctx, input_ids_tensor, {0, -1, 0, -1, 0, 1}, {1, 2, 3, 4, 5, 6}, {3, 2, 1});

    assert_slice_is_correct(ctx, input_ids_tensor, {0, -1, 0, 1, 0, -1}, {1, 2, 3}, {3, 1, 1});
    assert_slice_is_correct(ctx, input_ids_tensor, {0, -1, 1, -1, 0, -1}, {4, 5, 6}, {3, 1, 1});

    assert_slice_is_correct(ctx, input_ids_tensor, {0, 2, 0, -1, 0, -1}, {1, 2, 4, 5}, {2, 2, 1});
    assert_slice_is_correct(ctx, input_ids_tensor, {2, -1, 0, -1, 0, -1}, {3, 6}, {1, 2, 1});


    auto tensor_abc = create_tensor_with_data_and_shape(ctx, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}, 2, 4, 2);

    assert_slice_is_correct(ctx, tensor_abc, {0, -1, 0, 1, 0, -1}, {1, 2, 9, 10}, {2, 1, 2});
    assert_slice_is_correct(ctx, tensor_abc, {0, -1, 1, 2, 0, -1}, {3, 4, 11, 12}, {2, 1, 2});
    assert_slice_is_correct(ctx, tensor_abc, {0, -1, 2, 3, 0, -1}, {5, 6, 13, 14}, {2, 1, 2});
    assert_slice_is_correct(ctx, tensor_abc, {0, -1, 3, 4, 0, -1}, {7, 8, 15, 16}, {2, 1, 2});

    // Split tests

    assert_split_is_correct(ctx, input_ids_tensor, 2, 1, 0,
                            {1, 2, 4, 5}, {2, 2, 1},
                            {3, 6}, {1, 2, 1});
    // Flip
    print_tensor(input_ids_tensor, "Input");
    assert_flip_is_correct(ctx, input_ids_tensor, 0, {3, 2, 1, 6, 5, 4}, {3, 2, 1});
    assert_flip_is_correct(ctx, input_ids_tensor, 1, {4, 5, 6, 1, 2, 3}, {3, 2, 1});

    auto tensor_a = create_tensor_with_data_and_shape(ctx, {1, 2, 3}, 3, 1, 1);
    auto tensor_b = create_tensor_with_data_and_shape(ctx, {1, 7, 8}, 3, 1, 1);

    assert_concat_is_correct(ctx, tensor_b, tensor_a, 1, {1, 7, 8, 1, 2, 3}, {3, 2, 1});

    assert_concat_is_correct(ctx, tensor_b, tensor_a, 0, {1, 7, 8, 1, 2, 3}, {6, 1, 1});

    auto tensor_start = create_tensor_with_data_and_shape(ctx, {1, 2, 3}, 3, 1, 1);
    std::vector<int> accum = {1, 2, 3};
    for(int k = 2; k < 12; k++) {
        auto tensor_i = create_tensor_with_data_and_shape(ctx, {(float) k, (float)k+1, (float)k+2}, 3, 1, 1);
        accum.push_back(k);
        accum.push_back(k+1);
        accum.push_back(k+2);
        tensor_start = assert_concat_is_correct(ctx, tensor_start, tensor_i, 1, accum, {3, k, 1});
    }

    assert_compare_is_correct(ctx, tensor_a, tensor_b, [](float a, float b) { return a < b; }, {0, 1, 1}, {3, 1, 1});

    assert_compare_is_correct(ctx, tensor_a, tensor_b, [](float a, float b) { return a >= b; }, {1, 0, 0}, {3, 1, 1});

    auto a = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_2d(ctx, tensor_b, tensor_b->ne[0], tensor_b->ne[1]), 1, 0, 2, 3));
    auto b = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_2d(ctx, input_ids_tensor, input_ids_tensor->ne[0], input_ids_tensor->ne[1]), 1, 0, 2, 3));
    print_tensor(execute_tensor(ctx, a), "a");
    print_tensor(execute_tensor(ctx, b), "b");
    assert_compare_is_correct(
            ctx,
            a,
            b,
            [](float a, float b) { return a >= b; },
            {1, 0, 1, 1, 1, 1},
            {2, 3}
    );


    auto tensor_d = create_tensor_with_data_and_shape(ctx, {1, 2, 3}, 3, 1, 1);
    assert_cumsum_is_correct(ctx, tensor_d, {1, 3, 6}, {3, 1, 1});

    auto tensor_c = create_tensor_with_data_and_shape(ctx, {6, 2, 3, 8, 4, 2}, 3, 2, 1);
    assert_max_is_correct(ctx, tensor_c, {8}, {1});

    assert_not_is_correct(ctx, create_tensor_with_data_and_shape(ctx, {0, 1, 0, 1, 0, 0}, 3, 2, 1), {1, 0, 1, 0, 1, 1}, {3, 2, 1});

    assert_index_put_last_dim_is_correct(
            ctx,
            ggml_dup(ctx, input_ids_tensor),
            0,
            10,
            {10, 2, 3, 10, 5, 6},
            {3, 2, 1}
    );

    assert_index_put_last_dim_is_correct(
            ctx,
            ggml_dup(ctx, input_ids_tensor),
            2,
            10,
            {1, 2, 10, 4, 5, 10},
            {3, 2, 1}
    );

    assert_index_put_last_dim_is_correct(
            ctx,
            ggml_dup(ctx, input_ids_tensor),
            1,
            5,
            {1, 5, 3, 4, 5, 6},
            {3, 2, 1}
    );


    assert_index_add_last_dim_is_correct(
            ctx,
            input_ids_tensor,
            1,
            9,
            {1, 11, 3, 4, 14, 6},
            {3, 2, 1}
    );

    assert_masked_set_is_correct(ctx,
        input_ids_tensor,
        create_tensor_with_data_and_shape(ctx, {0, 1, 0, 1, 1, 0}, 3, 2, 1),
        create_tensor_with_data_and_shape(ctx, {10, 10, 10, 10, 10, 10}, 3, 2, 1),
        {1, 10, 3, 10, 10, 6},
        {3, 2, 1}
    );
/*
    assert_masked_get_is_correct(ctx,
         input_ids_tensor,
         create_tensor_with_data_and_shape(ctx, {0, 1, 0, 1, 1, 0}, 3, 2, 1),
         {2, 4, 5},
         {3}
    );
*/
    assert_gather_is_correct(
            ctx,
            input_ids_tensor,
            0,
            create_tensor_1d_with_data_and_shape(ctx, {0, 2, 5}, 3),
            {1, 3, 6},
            {3}
    );

    assert_arange_is_correct(
            ctx,
            6,
            {0, 1, 2, 3, 4, 5},
            {6, 1, 1}
            );


    return 0;
}
