#include <vector>
#include <chrono>
#include <iostream>
#include <random>
#include <iomanip>
#include <cmath>
#include <numeric>

// Include your libraries
#include <ggml/ggml.h>
#include <hwy/highway.h>

HWY_BEFORE_NAMESPACE();
namespace HWY_NAMESPACE {
    template<class T> void DotProductSIMD(const T* arr1, const T* arr2, T* result, size_t size) {
        const HWY_FULL(T) d;
        auto accum = Zero(d);

        size_t i = 0;
        const size_t limit = size - size % Lanes(d);
        for (; i < limit; i += Lanes(d)) {
            auto v1 = Load(d, arr1 + i);
            auto v2 = Load(d, arr2 + i);
            accum = MulAdd(v1, v2, accum);
        }
        *result = GetLane(SumOfLanes(d, accum));

        for (; i < size; ++i) {
            *result += arr1[i] * arr2[i];
        }
    }
}  // namespace HWY_NAMESPACE
HWY_AFTER_NAMESPACE();


template<class T> void DotProductStandard(const T* arr1, const T* arr2, T* result, size_t size) {
    T accum = 0;
    for (size_t i = 0; i < size; ++i) {
        accum += arr1[i] * arr2[i];
    }
    *result = accum;
}

#if defined(__ARM_NEON)
#include <arm_neon.h>

void DotProductSIMD_NEON(const float* arr1, const float* arr2, float* result, size_t size) {
    size_t i = 0;
    int lane = 4;
    const size_t limit = size - size % lane;
    float32x4_t acc = vdupq_n_f32(0.0f);

    for (; i < limit; i += lane) {
        float32x4_t a = vld1q_f32(arr1 + i);
        float32x4_t b = vld1q_f32(arr2 + i);
        acc = vfmaq_f32(a, b, acc);
    }
    float32x2_t sum = vadd_f32(vget_high_f32(acc), vget_low_f32(acc));
    float32_t final = vget_lane_f32(vpadd_f32(sum, sum), 0);
    *result = final;

    for (; i < size; ++i) {
        *result += arr1[i] * arr2[i];
    }
}

#else

void DotProductSIMD_NEON(const float* arr1, const float* arr2, float* result, size_t size) {
    printf("NEON not supported on this platform\n");
    DotProductStandard(arr1, arr2, result, size);
}

#endif

template<typename T>
std::pair<T, T> calculateMeanAndStd(const std::vector<T>& data) {
    T sum = std::accumulate(data.begin(), data.end(), T(0));
    T mean = sum / data.size();

    std::vector<T> diff(data.size());
    std::transform(data.begin(), data.end(), diff.begin(), [mean](T x) { return x - mean; });
    T sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), T(0));
    T std = std::sqrt(sq_sum / data.size());

    return {mean, std};
}

struct ggml_tensor * create_tensor_with_data_and_shape(struct ggml_context * ctx, ggml_fp16_t val, int h, int w, int d) {
    auto tensor = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, h, w, d);
    auto fp16_data = (ggml_fp16_t*)tensor->data;
    for (int i = 0; i < ggml_nelements(tensor); ++i) {
        fp16_data[i] = val;
    }
    return tensor;
}

void bench_conv1d() {
    int stride = 1;
    int dilation = 1;
    int padding = 0;

    struct ggml_init_params params = {
            .mem_size   = 256*1024*1024,
            .mem_buffer = nullptr,
    };

    struct ggml_context * ctx = ggml_init(params);

    auto dst = create_tensor_with_data_and_shape(ctx, 0, 1792, 256, 1);
    auto weights = create_tensor_with_data_and_shape(ctx, 0.5, 3, 256, 256);
    auto inputs = create_tensor_with_data_and_shape(ctx, 10, 1792, 256, 1);

    auto kernel_size = weights->ne[0];
    auto w_in_channels = weights->ne[1];
    auto out_channels = weights->ne[2];

    auto in_length = inputs->ne[0];
    auto i_in_channels = inputs->ne[1];
    auto batch_size = inputs->ne[2];

    auto output_size = ((in_length - dilation * (kernel_size - 1) - 1) / stride) + 1;
    printf("output_size: %d in_length: %d\n", output_size, in_length);
    auto work_data = new ggml_fp16_t[ggml_nelements(inputs)];
    memset(work_data, 0, ggml_nelements(inputs) * sizeof(ggml_fp16_t));
    auto storage = new int64_t[5] {stride, padding, dilation, output_size, (int64_t)work_data};

    auto* dst_ptr = (ggml_fp16_t*)dst->data;
    const auto* src0_ptr = (ggml_fp16_t*)inputs->data;
    const auto* weights_ptr = (ggml_fp16_t*)weights->data;

    auto input_size = ggml_element_size(inputs);
    auto weights_size = ggml_element_size(weights);
    auto work_size = ggml_element_size(dst);

    auto storage_ptr = (int64_t*) storage;
    auto* work_data_ptr = (ggml_fp16_t*)storage_ptr[4];

    auto channel_count = inputs->ne[1];

    auto traits = ggml_internal_get_type_traits(inputs->type);

    const int max_kernel_size = 16;
    const int max_channel_count = 768;
    const int max_buffer_size = max_channel_count * max_kernel_size;

    printf("conv1d stride: %d channels: %d, kernel_size: %d, in_+length %d\n", stride, channel_count, kernel_size, in_length);
    auto start = std::chrono::high_resolution_clock::now();
    ggml_fp16_t input_buffer[max_buffer_size];
    ggml_fp16_t kernel_buffer[max_buffer_size];
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
                        //input_buffer[n] = src0_ptr[input_tensor_idx];
                        //kernel_buffer[n] = weights_ptr[weights_idx];
                        n++;
                        w++;
                    }
                }
            }
            //traits.vec_dot(n, &sum, input_buffer, kernel_buffer);
            auto output_idx = (i * dst->nb[0] + co * dst->nb[1]) / work_size;
            work_data_ptr[output_idx] = (ggml_fp16_t) sum;
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto delta = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    printf("Done total: %f million elements. Took %lld ms\n", (w / 1e6), delta);
}

int bench_dot() {
    const size_t numIterations = 1;
    const size_t vectorSize = 1e8;

    std::vector<double> timingsStandard, timingsSIMD, timingsGGML32, timingsGGML16, timingsStandardDouble, timingsSIMDNeon;
    std::vector<double> errorsStandard, errorsSIMD, errorsGGML32, errorsGGML16, errorsStandardDouble, errorsSIMDNeon;

    for (size_t iteration = 0; iteration < numIterations; ++iteration) {
        std::cout << "\rCurrent iteration: " << iteration << std::flush;

        std::vector<float> arr1(vectorSize), arr2(vectorSize);
        std::vector<double> arr1_double(vectorSize), arr2_double(vectorSize);
        std::vector<ggml_fp16_t> arr1_fp16(vectorSize), arr2_fp16(vectorSize);

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dis(0.0, 0.1);

        for (size_t i = 0; i < vectorSize; ++i) {
            arr1_double[i] = dis(gen);
            arr2_double[i] = dis(gen);
            arr1[i] = static_cast<float>(arr1_double[i]);
            arr2[i] = static_cast<float>(arr2_double[i]);
            arr1_fp16[i] = ggml_fp32_to_fp16(arr1[i]);
            arr2_fp16[i] = ggml_fp32_to_fp16(arr2[i]);
        }

        double expectedResultDouble = 0;
        DotProductStandard<double>(arr1_double.data(), arr2_double.data(), &expectedResultDouble, vectorSize);

        auto start = std::chrono::high_resolution_clock::now();
        float resultStandard = 0;
        DotProductStandard<float>(arr1.data(), arr2.data(), &resultStandard, vectorSize);
        auto end = std::chrono::high_resolution_clock::now();
        timingsStandard.push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
        errorsStandard.push_back(fabs(resultStandard - expectedResultDouble));

        // SIMD float32
        start = std::chrono::high_resolution_clock::now();
        float resultSIMD = 0;
        HWY_NAMESPACE::DotProductSIMD(arr1.data(), arr2.data(), &resultSIMD, vectorSize);
        end = std::chrono::high_resolution_clock::now();
        timingsSIMD.push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
        errorsSIMD.push_back(fabs(resultSIMD - expectedResultDouble));

        start = std::chrono::high_resolution_clock::now();
        float resultSIMD_Neon = 0;
        DotProductSIMD_NEON(arr1.data(), arr2.data(), &resultSIMD_Neon, vectorSize);
        end = std::chrono::high_resolution_clock::now();
        timingsSIMDNeon.push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
        errorsSIMDNeon.push_back(fabs(resultSIMD_Neon - expectedResultDouble));

        // GGML float32
        start = std::chrono::high_resolution_clock::now();
        float resultGGML32 = 0;
        ggml_internal_get_type_traits(GGML_TYPE_F32).vec_dot(vectorSize, &resultGGML32, arr1.data(), arr2.data());
        end = std::chrono::high_resolution_clock::now();
        timingsGGML32.push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
        errorsGGML32.push_back(fabs(resultGGML32 - expectedResultDouble));

        // GGML float16
        start = std::chrono::high_resolution_clock::now();
        float resultGGML16 = 0;
        ggml_internal_get_type_traits(GGML_TYPE_F16).vec_dot(vectorSize, &resultGGML16, arr1_fp16.data(), arr2_fp16.data());
        end = std::chrono::high_resolution_clock::now();
        timingsGGML16.push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
        errorsGGML16.push_back(fabs(resultGGML16 - expectedResultDouble));

        // Standard double
        start = std::chrono::high_resolution_clock::now();
        double resultStandardDouble = 0;
        DotProductStandard<double>(arr1_double.data(), arr2_double.data(), &resultStandardDouble, vectorSize);
        end = std::chrono::high_resolution_clock::now();
        timingsStandardDouble.push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
        errorsStandardDouble.push_back(fabs(resultStandardDouble - expectedResultDouble));

        // print all results
        //std::cout << "\nresultStandard: " << resultStandard << " resultSIMD: " << resultSIMD << " resultGGML32: " << resultGGML32 << " resultGGML16: " << resultGGML16 << " resultStandardDouble: " << resultStandardDouble << "\n";
    }

    auto [meanTimingStandard, stdTimingStandard] = calculateMeanAndStd(timingsStandard);
    auto [meanErrorStandard, stdErrorStandard] = calculateMeanAndStd(errorsStandard);
    auto [meanTimingSIMD, stdTimingSIMD] = calculateMeanAndStd(timingsSIMD);
    auto [meanTimingSIMDNeon, stdTimingSIMDNeon] = calculateMeanAndStd(timingsSIMDNeon);
    auto [meanErrorSIMD, stdErrorSIMD] = calculateMeanAndStd(errorsSIMD);
    auto [meanTimingGGML32, stdTimingGGML32] = calculateMeanAndStd(timingsGGML32);
    auto [meanErrorGGML32, stdErrorGGML32] = calculateMeanAndStd(errorsGGML32);
    auto [meanTimingGGML16, stdTimingGGML16] = calculateMeanAndStd(timingsGGML16);
    auto [meanErrorGGML16, stdErrorGGML16] = calculateMeanAndStd(errorsGGML16);
    auto [meanTimingStandardDouble, stdTimingStandardDouble] = calculateMeanAndStd(timingsStandardDouble);
    auto [meanErrorStandardDouble, stdErrorStandardDouble] = calculateMeanAndStd(errorsStandardDouble);
    auto [meanErrorSIMDNeon, stdErrorSIMDNeon] = calculateMeanAndStd(errorsSIMDNeon);
    std::cout << "\n";
    std::cout << std::left << std::setw(20) << "Implementation"
              << std::setw(20) << "Mean Timing (ms)"
              << std::setw(20) << "Std Timing (ms)"
              << std::setw(20) << "Mean Error"
              << std::setw(20) << "Std Error" << "\n";

    std::cout << std::setw(20) << "Standard float32"
              << std::setw(20) << meanTimingStandard / 1000
              << std::setw(20) << stdTimingStandard / 1000
              << std::setw(20) << meanErrorStandard
              << std::setw(20) << stdErrorStandard << "\n";

    std::cout << std::setw(20) << "SIMD float32"
              << std::setw(20) << meanTimingSIMD / 1000
              << std::setw(20) << stdTimingSIMD / 1000
              << std::setw(20) << meanErrorSIMD
              << std::setw(20) << stdErrorSIMD << "\n";

    std::cout << std::setw(20) << "SIMD Neon float32"
              << std::setw(20) << meanTimingSIMDNeon / 1000
              << std::setw(20) << stdTimingSIMDNeon / 1000
              << std::setw(20) << meanErrorSIMDNeon
              << std::setw(20) << stdErrorSIMDNeon << "\n";

    std::cout << std::setw(20) << "GGML float32"
              << std::setw(20) << meanTimingGGML32 / 1000
              << std::setw(20) << stdTimingGGML32 / 1000
              << std::setw(20) << meanErrorGGML32
              << std::setw(20) << stdErrorGGML32 << "\n";

    std::cout << std::setw(20) << "GGML float16"
              << std::setw(20) << meanTimingGGML16 / 1000
              << std::setw(20) << stdTimingGGML16 / 1000
              << std::setw(20) << meanErrorGGML16
              << std::setw(20) << stdErrorGGML16 << "\n";

    std::cout << std::setw(20) << "Standard double"
              << std::setw(20) << meanTimingStandardDouble / 1000
              << std::setw(20) << stdTimingStandardDouble / 1000
              << std::setw(20) << meanErrorStandardDouble
              << std::setw(20) << stdErrorStandardDouble << "\n";
    return 0;
}

int main() {
    bench_conv1d();
    //bench_dot();
    return 0;
}