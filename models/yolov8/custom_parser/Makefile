CUDA_VER ?= 13.1
DS_ROOT  ?= /opt/nvidia/deepstream/deepstream-9.0

CC       = g++
CFLAGS   = -shared -fPIC -std=c++17 -O2
INCLUDES = -I$(DS_ROOT)/sources/includes \
           -I/usr/local/cuda-$(CUDA_VER)/include

TARGET   = ../libnvds_infercustomparser_yolov8.so

all: $(TARGET)

$(TARGET): nvdsparsebbox_yolov8.cpp
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $<
	@echo "Built: $@"

clean:
	rm -f $(TARGET)
