cmake_minimum_required (VERSION 3.0)
project(vits VERSION 0.1.0)

set(CMAKE_C_STANDARD   20)
set(CMAKE_CXX_STANDARD 20)

if(CMAKE_BUILD_TYPE MATCHES scripts/debug)
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -fsanitize=address")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=address")
    set(CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fsanitize=address")
endif()

add_subdirectory(ggml)

add_library(vits src/vits.cpp src/vits_model_data.cpp)
target_include_directories(vits PRIVATE ggml/include)

target_link_libraries(vits PRIVATE ggml)

add_executable(main test/main.cpp)
target_include_directories(main PRIVATE src/include ggml/include)
target_link_libraries(main PRIVATE vits)

add_executable(tests test/test_ggml_utils.cpp)
target_include_directories(tests PRIVATE src/include ggml/include)
target_link_libraries(tests PRIVATE vits ggml)