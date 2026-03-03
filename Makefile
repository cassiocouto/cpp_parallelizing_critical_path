CXX      := g++
CXXFLAGS := -std=c++17 -O2 -fopenmp -Iinclude
LDFLAGS  := -fopenmp

SRCDIR   := src
BUILDDIR := build

SOURCES  := $(wildcard $(SRCDIR)/*.cpp)
TARGETS  := $(patsubst $(SRCDIR)/%.cpp,$(BUILDDIR)/%,$(SOURCES))

.PHONY: all clean

all: $(TARGETS)

$(BUILDDIR)/%: $(SRCDIR)/%.cpp include/market_data.h | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

$(BUILDDIR):
	mkdir -p $(BUILDDIR)

clean:
	rm -rf $(BUILDDIR)
