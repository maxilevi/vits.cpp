//
// Created by Maximiliano Levi on 6/15/24.
//
#include "include/debug.h"
#include <execinfo.h>
#include <iostream>
#include <cstdlib>

void print_stack_trace() {
    const int maxFrames = 100;
    void* frames[maxFrames];
    int frameCount = backtrace(frames, maxFrames);
    char** symbols = backtrace_symbols(frames, frameCount);

    if (symbols) {
        for (int i = 0; i < frameCount; ++i) {
            std::cout << symbols[i] << std::endl;
        }
        free(symbols);
    } else {
        std::cerr << "Failed to generate backtrace symbols." << std::endl;
    }
}