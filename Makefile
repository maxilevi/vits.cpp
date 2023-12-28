run: build
	./build/main

build:
	mkdir -p build
	cmake -S . -B ./build -DCMAKE_BUILD_TYPE=Release
	(cd build && make)

clean:
	rm -rf ./build

tests: build
	./build/tests

bench-simd: build
	./build/bench-simd

bench: build
	./build/bench --benchmark_time_unit=ms  --benchmark_repetitions=1

ios:
	mkdir -p build_ios
	cmake -S . -B ./build_ios -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=./ios.toolchain.cmake
	(cd build_ios && make)

.PHONY: all build run clean bench