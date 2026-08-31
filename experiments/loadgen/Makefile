CXX ?= g++
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra
LDLIBS ?= -lcurl -pthread

.PHONY: all clean

all: loadgen

loadgen: loadgen.cpp
	$(CXX) $(CXXFLAGS) $< $(LDLIBS) -o $@

clean:
	$(RM) loadgen
