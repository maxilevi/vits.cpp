all: build run

build:
	mkdir -p build
	cmake -S . -B ./build -DCMAKE_BUILD_TYPE=Debug
	(cd build && make)

run:
	./build/test

clean:
	rm -rf ./build

.PHONY: all build run clean