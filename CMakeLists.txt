cmake_minimum_required (VERSION 3.0)
project(vits VERSION 0.1.0)

set(CMAKE_C_STANDARD   20)
set(CMAKE_CXX_STANDARD 20)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    # Set the macro VITS_DEBUG=True
    message("Building in debug mode")
    add_compile_options(-fsanitize=address)
    add_link_options(-fsanitize=address)
else()
    add_compile_options(-O3 -march=native -flto -fno-omit-frame-pointer)
    add_link_options(-O3 -march=native -flto -fno-omit-frame-pointer)
endif()

add_subdirectory(ggml)

add_library(vits src/vits.cpp src/vits_model_data.cpp
        src/vits_tokenizer.cpp)
target_include_directories(vits PRIVATE ggml/include)

target_link_libraries(vits PRIVATE ggml)

add_executable(main test/main.cpp
        src/include/vits_tokenizer.h
        src/include/custom-ops.h
)
target_include_directories(main PRIVATE src/include ggml/include)
target_link_libraries(main PRIVATE vits)