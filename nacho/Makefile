.PHONY: all clean

CXXFLAGS ?= -O2 -Wall -Wextra
NACHO_DIR := ../third_party/nacho
NACHO_LIBRARY := $(NACHO_DIR)/build/libnacho_library.a

all: generated

nacho: nacho.cpp $(NACHO_LIBRARY)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -std=c++20 -I$(NACHO_DIR)/include -MMD -MP $< $(LDFLAGS) $(NACHO_LIBRARY) $(LDLIBS) -o $@

$(NACHO_LIBRARY):
	$(MAKE) -C .. build-nacho

generated: nacho
	./nacho

clean:
	rm -f nacho nacho.d
	rm -rf generated

-include nacho.d
