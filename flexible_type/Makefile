ifndef CXX
CXX = g++
endif

SFRAME_SRC = $(PWD)/../plugin/SFrameSubtree/oss_src
DMLC_SRC = $(PWD)/../dmlc-core/include

CFLAGS = -std=c++11 -msse2 -fPIC -O3 -Wall -finline-functions -I$(SFRAME_SRC) -I$(DMLC_SRC) -include $(PWD)/flexible_type.h

all: flexible_type 

clean:
	rm -rf build

flexible_type: build/libflexible_type.a

test: build/flexible_type_test
OBJS = $(addprefix build/, flexible_type/flexible_type.o image/image_type.o)

build/libflexible_type.a: $(OBJS)
	ar crv $@ $(filter %.o, $?)

build/%.o: $(SFRAME_SRC)/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) -std=c++0x -MM -MT build/$*.o $< >build/$*.d
	$(CXX) $(CFLAGS) -c $< -o $@

build/flexible_type_test: test/flexible_type_test.cpp
	$(CXX) $(CFLAGS) $< -o $@ -L build -lflexible_type
