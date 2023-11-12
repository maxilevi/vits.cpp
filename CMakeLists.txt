cmake_minimum_required (VERSION 3.0)
project(vits VERSION 0.1.0)

set(CMAKE_C_STANDARD   20)
set(CMAKE_CXX_STANDARD 20)

if(CMAKE_BUILD_TYPE MATCHES debug)
    # Set the macro VITS_DEBUG=True
    add_compile_definitions(VITS_DEBUG=1)
    add_compile_definitions(GGML_PERF=1)
    add_compile_definitions(GGML_DEBUG=10)
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -fsanitize=address")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=address")
    set(CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fsanitize=address")

else()
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -Ofast -march=native -flto -fno-omit-frame-pointer")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Ofast -march=native -flto -fno-omit-frame-pointer")
    set(CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -Ofast -march=native -flto -fno-omit-frame-pointer")
endif()

add_subdirectory(ggml)

add_library(vits src/vits.cpp src/vits_model_data.cpp
        src/vits_tokenizer.cpp)
target_include_directories(vits PRIVATE ggml/include)

target_link_libraries(vits PRIVATE ggml)

add_executable(main test/main.cpp
        src/include/vits_tokenizer.h)
target_include_directories(main PRIVATE src/include ggml/include)
target_link_libraries(main PRIVATE vits)

add_executable(tests test/test_ggml_utils.cpp)
target_include_directories(tests PRIVATE src/include ggml/include)
target_link_libraries(tests PRIVATE vits ggml)