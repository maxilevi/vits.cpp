run: build
	./build/main

build:
	mkdir -p build
	cmake -S . -B ./build -DCMAKE_BUILD_TYPE=Release
	(cd build && make)

debug:
	mkdir -p build
	cmake -S . -B ./build -DCMAKE_BUILD_TYPE=Debug
	(cd build && make)

clean:
	rm -rf ./build

.PHONY: all build run clean bench