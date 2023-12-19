# Makefile for CUDA matrix multiplication test

# Compiler
NVCC = nvcc

# Compiler flags
NVCCFLAGS = -lcublas

# Target executable
TARGET = matrix_mult_test

# Source files
SRCS = matrix_mult_test.cu kernel1.cu

# Object files
OBJS = $(SRCS:.cu=.o)

# Compile rule
%.o: %.cu
	$(NVCC) -c $< -o $@

# Link rule
$(TARGET): $(OBJS)
	$(NVCC) $(NVCCFLAGS) $^ -o $@

# Phony target to clean up intermediate files
clean:
	rm -f $(TARGET) $(OBJS)
