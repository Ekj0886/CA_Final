# Compilers
CXX = g++
NVCC = nvcc

# Flags
CXXFLAGS = -std=c++11 -Wall -I$(INCDIR)
NVCCFLAGS = -std=c++11 -I$(INCDIR)

# Directories
SRCDIR = src
OBJDIR = obj
INCDIR = inc

# Files
CPP_SRCS = $(filter-out $(SRCDIR)/Image.cu, $(wildcard $(SRCDIR)/*.cpp)) main.cpp
CU_SRCS  = $(wildcard $(SRCDIR)/*.cu)

CPP_OBJS = $(patsubst %.cpp, $(OBJDIR)/%.o, $(notdir $(CPP_SRCS)))
CU_OBJS  = $(patsubst %.cu, $(OBJDIR)/%.o, $(notdir $(CU_SRCS)))

DEPS = $(CPP_OBJS:.o=.d)

TARGET = CA_Final

# Default rule
all: $(TARGET)

# Ensure obj directory exists
$(OBJDIR):
	@mkdir -p $(OBJDIR)

# Final target
$(TARGET): $(CPP_OBJS) $(CU_OBJS) | $(OBJDIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@

# Compile C++ sources
$(OBJDIR)/%.o: $(SRCDIR)/%.cpp | $(OBJDIR)
	$(CXX) $(CXXFLAGS) -MMD -c $< -o $@

# Compile CUDA sources
$(OBJDIR)/%.o: $(SRCDIR)/%.cu | $(OBJDIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Compile main.cpp separately
$(OBJDIR)/main.o: main.cpp | $(OBJDIR)
	$(CXX) $(CXXFLAGS) -MMD -c $< -o $@

# Include dependency files
-include $(DEPS)

# Clean rule
clean:
	rm -rf $(OBJDIR) $(TARGET)

.PHONY: all clean
