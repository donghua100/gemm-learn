CC=cc
NVCC=nvcc
CUDA_HOME = $(dirname $(dirname $(which nvcc)))
CUDA_LIB = $(CUDA_HOME)/lib64
INC_PATH = $(CUDA_HOME)/include
SRCS = $(shell find $(abspath ./) -name "*.cu")
BUILD_DIR = ./build
$(shell mkdir -p $(BUILD_DIR))

BINARY = $(BUILD_DIR)/gemm
default: $(BINARY)

all: default

$(BINARY): $(SRCS)
	$(NVCC) -I $(INC_PATH) $(SRCS) -o $(abspath $@)

run:$(BINARY)
	@$^

clean:
	rm -rf $(BUILD_DIR)

.PHONY:run clean all default
