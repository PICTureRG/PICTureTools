#######################################################################
#
#	Makefile 
#	
#######################################################################
# -Xcompiler -rdynamic -lineinfo
# ^is useful for cuda-memcheck line numbering/function naming capability on linux

CC =		g++ -g
#arch 35 is needed for kernel recursion, and rdc=true fixes the separate compilation issue.
CUDA =          nvcc -O3 -arch=sm_35 -rdc=true -Xcompiler -rdynamic -lineinfo -Xptxas -dlcm=ca -D_FORCE_INLINES
CFLAGS =	-L/usr/lib/x86_64-linux-gnu -lcudart -lcublas -lcublas_device
#-lblas
GOAL =		rbm
CFILES =	$(wildcard src/*.cpp)
CUDAFILES =     $(wildcard src/*.cu)
HFILES =	$(wildcard include/*.h)
OBJECTS =	$(addprefix obj/, $(notdir $(CFILES:.cpp=.o))) $(addprefix obj/, $(notdir $(CUDAFILES:.cu=.o)))
# CFILES_PCD =	$(wildcard src_PCD/*.cpp)
# HFILES_PCD =	$(wildcard include_PCD/*.h)
# OBJECTS_PCD =	$(addprefix obj_PCD/, $(notdir $(CFILES_PCD:.cpp=.o)))
# CFILES_G =	$(wildcard src_G/*.cpp)
# HFILES_G =	$(wildcard include_G/*.h)
# OBJECTS_G =	$(addprefix obj_G/, $(notdir $(CFILES_G:.cpp=.o)))


all: $(GOAL)
# $(GOAL_PCD) $(GOAL_G) 

$(GOAL): $(OBJECTS)
	$(CUDA) -o $@ $^ $(CFLAGS)

obj/%.o: src/%.cpp
	$(CC) -c -o $@ $< $(CFLAGS)

obj/%.o: src/%.cu
	$(CUDA) -c -o $@ $< $(CFLAGS)

obj/write_weights_kernel.o: src/write_weights_kernel.cu
	$(CUDA) -c -o $@ $< $(CFLAGS) -maxrregcount=32
# allows at least 2 warps to be
# running simultaneously

# $(GOAL_PCD): $(OBJECTS_PCD)
# 	$(CC) -o $@ $^ $(CFLAGS)

# obj_PCD/%.o: src_PCD/%.cpp
# 	$(CC) -c -o $@ $< $(CFLAGS)

# $(GOAL_G): $(OBJECTS_G)
# 	$(CC) -o $@ $^ $(CFLAGS)

# obj_G/%.o: src_G/%.cpp
# 	$(CC) -c -o $@ $< $(CFLAGS)

clean:
	rm -f $(OBJECTS) $(GOAL)
# rm -f $(OBJECTS_PCD) $(GOAL_PCD) 
# rm -f $(OBJECTS_G) $(GOAL_G) 
