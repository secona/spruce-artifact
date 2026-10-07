.PHONY: all clean generated

CXXFLAGS ?= -O2 -Wall -Wextra
NVCC ?= nvcc
NVCCFLAGS ?= -O3 -std=c++17 --extended-lambda
NACHO_DIR := ../third_party/nacho
NACHO_LIBRARY := $(NACHO_DIR)/build/libnacho_library.a

all: spgemm

nacho: nacho.cpp $(NACHO_LIBRARY)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -std=c++20 -I$(NACHO_DIR)/include -MMD -MP $< $(LDFLAGS) $(NACHO_LIBRARY) $(LDLIBS) -o $@

$(NACHO_LIBRARY):
	$(MAKE) -C .. build-nacho

generated: nacho
	./nacho

spgemm: spgemm.cu ../common/cuda.h generated
	$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) -I.. -I$(NACHO_DIR)/runtime $(LDFLAGS) spgemm.cu generated/spgemm_gpu.cu $(LDLIBS) -o $@

clean:
	rm -f nacho nacho.d spgemm
	rm -rf generated

-include nacho.d
