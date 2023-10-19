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
                std::cout << *(data + offset) << " (" << offset << ") ";
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
    auto result_tensor = tensor;

    ggml_build_forward_expand(&graph, result_tensor);
    int threads = std::min((int)std::thread::hardware_concurrency(), 2);
    auto plan = ggml_graph_plan(&graph, threads);
    if (plan.work_size > 0) {
        plan.work_data = (uint8_t*) malloc(plan.work_size);
    }

    ggml_graph_compute(&graph, &plan);

    return result_tensor;
}

void assert_tensor_matches_expected(
        struct ggml_tensor* tensor,
        const std::vector<int32_t>& expected,
        const std::vector<int32_t>& expected_shape,
        const std::string& op_name
) {
    ASSERT(expected.size() == expected_shape[0] * expected_shape[1] * expected_shape[2], "Expected shape should match expected data");
    ASSERT(tensor->n_dims == 3, "Result should be 3D");
    for (int i = 0; i < 3; ++i) {
        ASSERT(tensor->ne[i] == expected_shape[i], (op_name + " should have the correct shape").c_str());
    }

    int64_t total_elements = tensor->ne[0] * tensor->ne[1] * tensor->ne[2];
    auto data = (float*) ggml_get_data(tensor);
    bool equal = true;
    for (int i = 0; i < total_elements; ++i) {
        equal &= data[i] == expected[i];
    }

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
        const std::vector<int32_t>& expected,
        const std::vector<int32_t>& expected_shape
) {
    std::cout << "Testing Split" << std::endl;

    auto [left_tensor, right_tensor] = split_3d(ctx, input_ids_tensor, left, right, dim);
    struct ggml_tensor* left_result_tensor = execute_tensor(ctx, left_tensor);
    struct ggml_tensor* right_result_tensor = execute_tensor(ctx, right_tensor);

    assert_tensor_matches_expected(left_result_tensor, expected, expected_shape, "Split");
    assert_tensor_matches_expected(right_result_tensor, expected, expected_shape, "Split");

    std::cout << "Split is correct\n";
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

    // Split tests
    assert_split_is_correct()
    assert_split_is_correct()
    assert_split_is_correct()

    // Flip tests
    assert_flip_is_correct()

    return 0;
}
