#include <benchmark/benchmark.h>
#include <random>
#include <vits.h>

using namespace benchmark;

vits_model* model = nullptr;
struct ggml_context *ctx;
struct ggml_tensor *cur_fp32, *filters_fp32, *cur, *filters, *colA, *colB, *colA_fp32, *colB_fp32;

static void GlobalSetup() {
    model = vits_model_load_from_file("/Users/maximilianolevi/Documents/Repositories/vits.cpp/scripts/vits-spanish.ggml");

    struct ggml_init_params params = {
            .mem_size   = (size_t)256*1024*1024,
            .mem_buffer = nullptr,
    };

    ctx = ggml_init(params); // Initialize context with appropriate parameters

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0, 0.1);

    // Initialize tensors
    cur_fp32 = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 64000, 256, 1);
    for (int i = 0; i < ggml_nelements(cur_fp32); ++i) {
        ((float*) cur_fp32->data)[i] = dis(gen);
    }

    filters_fp32 = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 16, 256, 256);
    for (int i = 0; i < ggml_nelements(filters_fp32); ++i) {
        ((float*) filters_fp32->data)[i] = dis(gen);
    }

    // Convert to other formats if necessary
    cur = ggml_new_tensor(ctx, GGML_TYPE_F16, cur_fp32->n_dims, cur_fp32->ne);
    ggml_fp32_to_fp16_row((float*)cur_fp32->data, (ggml_fp16_t*)cur->data, ggml_nelements(cur_fp32));

    filters = ggml_new_tensor(ctx, GGML_TYPE_F16, filters_fp32->n_dims, filters_fp32->ne);
    ggml_fp32_to_fp16_row((float*)filters_fp32->data, (ggml_fp16_t*)filters->data, ggml_nelements(filters_fp32));

    // cols

    colA_fp32 = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 8000, 768, 1);
    for (int i = 0; i < ggml_nelements(cur_fp32); ++i) {
        ((float*) colA_fp32->data)[i] = dis(gen);
    }

    colB_fp32 = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 8000, 1200, 1);
    for (int i = 0; i < ggml_nelements(filters_fp32); ++i) {
        ((float*) colB_fp32->data)[i] = dis(gen);
    }

    // Convert to other formats if necessary
    colA = ggml_new_tensor(ctx, GGML_TYPE_F16, colA_fp32->n_dims, colA_fp32->ne);
    ggml_fp32_to_fp16_row((float*)colA_fp32->data, (ggml_fp16_t*)colA->data, ggml_nelements(colA));

    colB = ggml_new_tensor(ctx, GGML_TYPE_F16, colB_fp32->n_dims, colB_fp32->ne);
    ggml_fp32_to_fp16_row((float*)colB_fp32->data, (ggml_fp16_t*)colB->data, ggml_nelements(colB_fp32));
}

static void GlobalCleanup() {
    ggml_free(ctx);
    vits_free_model(model);
}

struct ggml_tensor* execute_tensor(
        struct ggml_context* ctx,
        struct ggml_tensor* tensor
) {
    struct ggml_cgraph* graph = ggml_new_graph(ctx);

    ggml_build_forward_expand(graph, tensor);
    int threads = std::max((int)std::thread::hardware_concurrency(), 2);
    auto plan = ggml_graph_plan(graph, threads);
    if (plan.work_size > 0) {
        plan.work_data = (uint8_t*) malloc(plan.work_size);
    }
    ggml_graph_compute(graph, &plan);
    return tensor;
}
/*
static void BM_tensor_conv_1d(State& state) {
    for (auto _ : state) {
        struct ggml_init_params params = {
                .mem_size   = (size_t)16*1024*1024*1024,
                .mem_buffer = nullptr,
        };

        auto ctx2 = ggml_init({.mem_size   = (size_t)16*1024*1024*1024,});
        auto result = tensor_conv_1d(ctx2, cur_fp32, filters_fp32, 1, 1, 1);
        benchmark::DoNotOptimize(result);
        result = execute_tensor(ctx2, result);
        ggml_free(ctx2);
    }
}
BENCHMARK(BM_tensor_conv_1d);

static void BM_ggml_conv_1d(State& state) {
    for (auto _ : state) {
        auto ctx2 = ggml_init({.mem_size   = (size_t)16*1024*1024*1024,});
        auto result = ggml_conv_1d(ctx2, filters, cur_fp32, 1, 1, 1);
        result = execute_tensor(ctx2, result);
        benchmark::DoNotOptimize(result);
        ggml_free(ctx2);
    }
}
BENCHMARK(BM_ggml_conv_1d);

static void BM_tensor_conv_1d_inplace(State& state) {
    for (auto _ : state) {
        auto ctx2 = ggml_init({.mem_size   = (size_t)16*1024*1024*1024,});
        auto result = tensor_conv_1d_inplace(ctx2, cur, filters, 1, 1, 1);
        result = execute_tensor(ctx2, result);
        benchmark::DoNotOptimize(result);
        ggml_free(ctx2);
    }
}
BENCHMARK(BM_tensor_conv_1d_inplace);

static void BM_im2col_impl(State& state) {
    for (auto _ : state) {
        auto ctx2 = ggml_init({.mem_size   = (size_t)16*1024*1024*1024,});
        auto result = im2col_impl(ctx2, filters_fp32, cur_fp32, 1, 1, 1);
        result = execute_tensor(ctx2, result);
        benchmark::DoNotOptimize(result);
        ggml_free(ctx2);
    }
}
BENCHMARK(BM_im2col_impl);

static void BM_ggml_im2col_1d(State& state) {
    for (auto _ : state) {
        auto ctx2 = ggml_init({.mem_size   = (size_t)16*1024*1024*1024,});
        auto result = ggml_im2col_1d(ctx2, filters, cur_fp32, 1, 1, 1);
        result = execute_tensor(ctx2, result);
        benchmark::DoNotOptimize(result);
        ggml_free(ctx2);
    }
}
BENCHMARK(BM_ggml_im2col_1d);

static void BM_ggml_im2col_1d_float(State& state) {
    for (auto _ : state) {
        auto ctx2 = ggml_init({.mem_size   = (size_t)16*1024*1024*1024,});
        auto result = ggml_im2col_1d(ctx2, filters_fp32, cur_fp32, 1, 1, 1);
        result = execute_tensor(ctx2, result);
        benchmark::DoNotOptimize(result);
        ggml_free(ctx2);
    }
}
BENCHMARK(BM_ggml_im2col_1d_float);

static void BM_ggml_im2col_1d_fp16_and_cast(State& state) {
    for (auto _ : state) {
        auto ctx2 = ggml_init({.mem_size   = (size_t)16*1024*1024*1024,});
        auto result = ggml_im2col_1d(ctx2, filters, cur_fp32, 1, 1, 1);
        result = cast_tensor_fp16_to_fp32(ctx2, result);
        result = execute_tensor(ctx2, result);
        benchmark::DoNotOptimize(result);
        ggml_free(ctx2);
    }
}
BENCHMARK(BM_ggml_im2col_1d_fp16_and_cast);

static void BM_ggml_im2col(State& state) {
    for (auto _ : state) {
        auto ctx2 = ggml_init({.mem_size   = (size_t)16*1024*1024*1024,});
        auto result = ggml_im2col(ctx2, filters, cur_fp32, 1, 0, 1, 0, 1, 0, false);
        result = execute_tensor(ctx2, result);
        benchmark::DoNotOptimize(result);
        ggml_free(ctx2);
    }
}
BENCHMARK(BM_ggml_im2col);

static void BM_ggml_mul_mat_fp16(State& state) {
    for (auto _ : state) {
        auto ctx2 = ggml_init({.mem_size   = (size_t)16*1024*1024*1024,});
        auto result = ggml_mul_mat(ctx2, colA, colB);
        result = execute_tensor(ctx2, result);
        benchmark::DoNotOptimize(result);
        ggml_free(ctx2);
    }
}
BENCHMARK(BM_ggml_mul_mat_fp16);

static void BM_ggml_mul_mat_fp32(State& state) {
    for (auto _ : state) {
        auto ctx2 = ggml_init({.mem_size   = (size_t)16*1024*1024*1024,});
        auto result = ggml_mul_mat(ctx2, colA_fp32, colB_fp32);
        result = execute_tensor(ctx2, result);
        benchmark::DoNotOptimize(result);
        ggml_free(ctx2);
    }
}
BENCHMARK(BM_ggml_mul_mat_fp32);
*/
static const char* phrase = "Cada amanecer trae consigo nuevas oportunidades para crecer y aprender.";

static void BM_vits_model_process(State& state) {
    for (auto _ : state) {
        auto result = vits_model_process(model, phrase);
        benchmark::DoNotOptimize(result);
        vits_free_result(result);
    }
}
BENCHMARK(BM_vits_model_process);

int main(int argc, char** argv) {
    GlobalSetup();

    ::benchmark::Initialize(&argc, argv);
    if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
    ::benchmark::RunSpecifiedBenchmarks();

    GlobalCleanup();
}