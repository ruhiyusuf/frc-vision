CXX=g++ 
RM=rm
CXXFLAGS=$(shell pkg-config --cflags gstreamer-1.0)
LIBS=$(shell pkg-config --libs gstreamer-1.0)

SOURCES=$(wildcard *.cpp)
OBJECTS=$(SOURCES:.cpp=.o)
EXECS=$(SOURCES:%.cpp=%)

.PHONY: all
all: $(OBJECTS) $(EXECS)

.cpp.o:
	$(CXX) -c -o $@ $< $(CXXFLAGS)

.o:
	$(CXX) -o $@ $^ $(CXXFLAGS) $(LIBS)

.PHONY: clean
clean: 
	-@ $(RM) *.o
	-@ $(RM) $(EXECS)
