run: build
	./build/main

build:
	mkdir -p build
	cmake -S . -B ./build -DCMAKE_BUILD_TYPE=Debug
	(cd build && make)

clean:
	rm -rf ./build

tests: build
	./build/tests

.PHONY: all build run clean