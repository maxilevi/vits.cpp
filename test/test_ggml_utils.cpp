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

void assert_padding_is_correct(struct ggml_context* ctx, struct ggml_tensor* input_ids_tensor, std::vector<int> pads, std::vector<int32_t> expected, std::vector<int32_t> expected_shape) {
    std::cout << "Testing padding [";
    for (int i = 0; i < pads.size(); ++i) {
        std::cout << " " << pads[i];
    }
    std::cout << " ]\n";

    struct ggml_cgraph graph = {};
    ASSERT(expected.size() == expected_shape[0] * expected_shape[1] * expected_shape[2], "Expected shape should match expected data");

    auto padding = pad_3d(ctx, input_ids_tensor, pads);

    ggml_build_forward_expand(&graph, padding);

    int threads = std::min((int)std::thread::hardware_concurrency(), 2);
    auto plan = ggml_graph_plan(&graph, threads);
    if (plan.work_size > 0) {
        plan.work_data = (uint8_t*) malloc(plan.work_size);
    }

    ggml_graph_compute(&graph, &plan);
    ASSERT(padding->n_dims == 3, "Padding should be 3D");
    ASSERT(padding->ne[0] == expected_shape[0], "Padding should have the correct shape");
    ASSERT(padding->ne[1] == expected_shape[1], "Padding should have the correct shape");
    ASSERT(padding->ne[2] == expected_shape[2], "Padding should have the correct shape");

    print_tensor(padding, "Padding");

    int64_t total_elements = padding->ne[0] * padding->ne[1] * padding->ne[2];
    auto data = (float*) ggml_get_data(padding);
    bool equal = true;
    for (int i = 0; i < total_elements; ++i) {
        equal &= data[i] == expected[i];
    }

    ASSERT(equal, "Padding should be correct");
    std::cout << "Padding [";
    for (int i = 0; i < pads.size(); ++i) {
        std::cout << " " << pads[i];
    }
    std::cout << " ] is correct\n";
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

    std::vector<float> input_ids2 = {0, 1, 2, 3, 0, 4, 5, 6};
    auto input_ids_tensor2 = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 4, 2, 1);
    memcpy(input_ids_tensor2->data, input_ids2.data(), ggml_element_size(input_ids_tensor2) * input_ids2.size());
    printf("Input tensor: %d %d %d \n", input_ids_tensor2->nb[0], input_ids_tensor2->nb[1], input_ids_tensor2->nb[2]);
    print_tensor(input_ids_tensor2, "Input tensor");


    assert_padding_is_correct(ctx, input_ids_tensor, {0, 0, 0, 0, 0, 0}, {1, 2, 3, 4, 5, 6}, {3, 2, 1});
    assert_padding_is_correct(ctx, input_ids_tensor, {0, 0, 0, 2, 0, 0}, {1, 2, 3, 4, 5, 6, 0, 0, 0, 0, 0, 0}, {3, 4, 1});
    assert_padding_is_correct(ctx, input_ids_tensor, {0, 0, 2, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6}, {3, 4, 1});
    assert_padding_is_correct(ctx, input_ids_tensor, {0, 0, 2, 1, 0, 0}, {0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 0, 0, 0}, {3, 5, 1});

    assert_padding_is_correct(ctx, input_ids_tensor, {0, 0, 0, 0, 0, 2}, {1, 2, 3, 0, 0, 4, 5, 6, 0, 0}, {5, 2, 1});
    assert_padding_is_correct(ctx, input_ids_tensor, {0, 0, 0, 0, 3, 0}, {0, 0, 0, 1, 2, 3, 0, 0, 0, 4, 5, 6}, {6, 2, 1});

    assert_padding_is_correct(ctx, input_ids_tensor, {1, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6}, {3, 2, 2});
    assert_padding_is_correct(ctx, input_ids_tensor, {0, 1, 0, 0, 0, 0}, {1, 2, 3, 4, 5, 6, 0, 0, 0, 0, 0, 0}, {3, 2, 2});


    return 0;
}
