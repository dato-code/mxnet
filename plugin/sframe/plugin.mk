SFRAME_HOME = plugin/SFrameSubtree/oss_src
# Basic flexible types
SFRAME_SRC = $(wildcard $(SFRAME_HOME)/flexible_type/*.cpp)
SFRAME_SRC += $(wildcard $(SFRAME_HOME)/logger/*.cpp)
SFRAME_SRC += $(wildcard $(SFRAME_HOME)/timer/*.cpp)
SFRAME_SRC += $(wildcard $(SFRAME_HOME)/parallel/*.cpp)
SFRAME_SRC += $(wildcard $(SFRAME_HOME)/image/image_type.cpp)
# Image decoding stuff
SFRAME_SRC += $(wildcard $(SFRAME_HOME)/image/jpeg_io.cpp)
SFRAME_SRC += $(wildcard $(SFRAME_HOME)/image/png_io.cpp)
# Make dependency objects
SFRAME_OBJ = $(patsubst %.cpp, build/%.o, $(SFRAME_SRC))
PLUGIN_OBJ += $(SFRAME_OBJ)
CFLAGS += -I$(SFRAME_HOME)
LDFLAGS += -lboost_system -lboost_filesystem -lpng -ljpeg -lz
