cmake_minimum_required (VERSION 3.0)
project(vits VERSION 0.1.0)

set(CMAKE_C_STANDARD   20)
set(CMAKE_CXX_STANDARD 20)

message(STATUS "CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    # Set the macro VITS_DEBUG=True
    message("Building in debug mode")
    add_compile_options(-fsanitize=address)
    add_link_options(-fsanitize=address)
else()
    add_compile_options(-O3 -march=native -flto -fno-omit-frame-pointer)
    add_link_options(-O3 -march=native -flto -fno-omit-frame-pointer)
endif()

#message("CMAKE_C_FLAGS_RELEASE: ${CMAKE_CXX_FLAGS_RELEASE}")
add_subdirectory(ggml)
#add_subdirectory(highway)

add_library(vits src/vits.cpp src/vits_model_data.cpp src/vits_tokenizer.cpp src/debug.cpp)
target_include_directories(vits PRIVATE ggml/include)

target_link_libraries(vits PRIVATE ggml)

find_library(LIBESPEAK_LIBRARIES espeak-ng)
if(LIBESPEAK_LIBRARIES)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(ESPEAK REQUIRED espeak-ng)
    add_definitions(-DVITS_ESPEAK)
endif()

if (NOT CMAKE_SYSTEM_NAME STREQUAL "iOS")
    add_executable(main test/main.cpp)

    add_executable(bench_e2e test/bench_e2e.cpp)

    target_include_directories(main PRIVATE src/include ggml/include ${ESPEAK_INCLUDE_DIRS})
    target_link_libraries(main PRIVATE vits ggml ${ESPEAK_LIBRARIES})

    target_include_directories(bench_e2e PRIVATE src/include ggml/include)
    target_link_libraries(bench_e2e PRIVATE vits ggml)

    add_executable(tests test/test_ggml_utils.cpp)
    target_include_directories(tests PRIVATE src/include ggml/include)
    target_link_libraries(tests PRIVATE vits ggml)

    add_executable(bench-simd test/bench_simd.cpp)
    target_include_directories(bench-simd PRIVATE src/include ggml/include highway/)
    target_link_libraries(bench-simd PRIVATE vits ggml)

    add_executable(bench test/bench.cpp)
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Disable Benchmark Testing")
    set(BENCHMARK_DOWNLOAD_DEPENDENCIES ON CACHE BOOL "Enable downloading of dependencies")
    add_subdirectory(benchmark)
    target_include_directories(bench PRIVATE src/include ggml/include)
    target_link_libraries(bench PRIVATE vits ggml benchmark::benchmark)
endif()