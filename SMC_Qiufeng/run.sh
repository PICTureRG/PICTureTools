#!/bin/bash

source vars.sh;

# Run the tool on all benchmarks

# 1. HeartWall
echo "====================================================="
echo "HeartWall"
$BUILD_DIR/bin/smc $BENCHMARKS/heartwall/main.cu -- --cuda-host-only -I /usr/local/cuda/include -I $BENCHMARKS/heartwall -I $BENCHMARKS/heartwall/AVI
$BUILD_DIR/bin/smc $BENCHMARKS/heartwall/kernel.cu -- --cuda-host-only -I /usr/local/cuda/include -I $BENCHMARKS/heartwall -I $BENCHMARKS/heartwall/AVI

# 2. cfd
echo "====================================================="
echo "cfd"
$BUILD_DIR/bin/smc $BENCHMARKS/cfd/euler3d.cu -- --cuda-host-only -I /usr/local/cuda/include
$BUILD_DIR/bin/smc $BENCHMARKS/cfd/euler3d_double.cu -- --cuda-host-only -I /usr/local/cuda/include
$BUILD_DIR/bin/smc $BENCHMARKS/cfd/pre_euler3d.cu -- --cuda-host-only -I /usr/local/cuda/include
$BUILD_DIR/bin/smc $BENCHMARKS/cfd/pre_euler3d_double.cu -- --cuda-host-only -I /usr/local/cuda/include

# 3. LUD
echo "====================================================="
echo "LUD"
$BUILD_DIR/bin/smc $BENCHMARKS/lud/cuda/lud.cu -- --cuda-host-only -I /usr/local/cuda/include
$BUILD_DIR/bin/smc $BENCHMARKS/lud/cuda/lud_kernel.cu -- --cuda-host-only -I /usr/local/cuda/include

# 4. Backprop
echo "====================================================="
echo "Backprop"
$BUILD_DIR/bin/smc $BENCHMARKS/backprop/backprop_cuda.cu -- --cuda-host-only -I /usr/local/cuda/include -I $BENCHMARKS/backprop
$BUILD_DIR/bin/smc $BENCHMARKS/backprop/backprop_cuda_kernel.cu -- --cuda-host-only -I /usr/local/cuda/include -I $BENCHMARKS/backprop

# 5. StreamCluster
echo "====================================================="
echo "StreamCluster"
$BUILD_DIR/bin/smc $BENCHMARKS/streamcluster/streamcluster_cuda.cu -- --cuda-host-only -I /usr/local/cuda/include

# 6. PathFinder
echo "====================================================="
echo "PathFinder"
$BUILD_DIR/bin/smc $BENCHMARKS/pathfinder/pathfinder.cu -- --cuda-host-only -I /usr/local/cuda/include

# 7. lavaMD
echo "====================================================="
echo "LavaMD"
$BUILD_DIR/bin/smc $BENCHMARKS/lavaMD/kernel/kernel_gpu_cuda.cu -- --cuda-host-only -I /usr/local/cuda/include -I $BENCHMARKS/lavaMD/kernel/
$BUILD_DIR/bin/smc $BENCHMARKS/lavaMD/kernel/kernel_gpu_cuda_wrapper.cu -- --cuda-host-only -I /usr/local/cuda/include -I $BENCHMARKS/lavaMD/kernel/

# 8. Leukocyte
echo "====================================================="
echo "Leukocyte"
$BUILD_DIR/bin/smc $BENCHMARKS/leukocyte/CUDA/find_ellipse_kernel.cu -- --cuda-host-only -I /usr/local/cuda/include -I $BENCHMARKS/leukocyte/CUDA/ -I $BENCHMARKS/leukocyte/meschach_lib/
$BUILD_DIR/bin/smc $BENCHMARKS/leukocyte/CUDA/track_ellipse_kernel.cu -- --cuda-host-only -I /usr/local/cuda/include -I $BENCHMARKS/leukocyte/CUDA/ -I $BENCHMARKS/leukocyte/meschach_lib/

# 9. dwt2d
echo "====================================================="
echo "dwt2d"
$BUILD_DIR/bin/smc $BENCHMARKS/dwt2d/dwt_cuda/fdwt53.cu -- --cuda-host-only -I /usr/local/cuda/include -I $BENCHMARKS/dwt2d/dwt_cuda/
$BUILD_DIR/bin/smc $BENCHMARKS/dwt2d/dwt_cuda/fdwt97.cu -- --cuda-host-only -I /usr/local/cuda/include -I $BENCHMARKS/dwt2d/dwt_cuda/
$BUILD_DIR/bin/smc $BENCHMARKS/dwt2d/dwt_cuda/rdwt53.cu -- --cuda-host-only -I /usr/local/cuda/include -I $BENCHMARKS/dwt2d/dwt_cuda/
$BUILD_DIR/bin/smc $BENCHMARKS/dwt2d/dwt_cuda/rdwt97.cu -- --cuda-host-only -I /usr/local/cuda/include -I $BENCHMARKS/dwt2d/dwt_cuda/

# 10. MUMmerGPU
echo "====================================================="
echo "MUMmerGPU"
$BUILD_DIR/bin/smc $BENCHMARKS/mummergpu/src/mummergpu.cu -- --cuda-host-only -I /usr/local/cuda/include -I $BENCHMARKS/mummergpu/src/ -L $BENCHMARKS/mummergpu/lib
$BUILD_DIR/bin/smc $BENCHMARKS/mummergpu/src/mummergpu_kernel.cu -- --cuda-host-only -I /usr/local/cuda/include -I $BENCHMARKS/mummergpu/src/ -L $BENCHMARKS/mummergpu/lib

# 11. Hotspot
echo "====================================================="
echo "Hotspot"
$BUILD_DIR/bin/smc $BENCHMARKS/hotspot/hotspot.cu -- --cuda-host-only -I /usr/local/cuda/include

# 12. Hotspot3D
echo "====================================================="
echo "Hotspot3D"
$BUILD_DIR/bin/smc $BENCHMARKS/hotspot3D/3D.cu -- --cuda-host-only -I /usr/local/cuda/include -I /usr/include/x86_64-linux-gnu/sys/time.h
$BUILD_DIR/bin/smc $BENCHMARKS/hotspot3D/opt1.cu -- --cuda-host-only -I /usr/local/cuda/include -I /usr/include/x86_64-linux-gnu/sys/time.h

# 13. NW
echo "====================================================="
echo "NW"
$BUILD_DIR/bin/smc $BENCHMARKS/nw/needle.cu -- --cuda-host-only -I /usr/local/cuda/include -I $BENCHMARKS/nw/needle.h
$BUILD_DIR/bin/smc $BENCHMARKS/nw/needle_kernel.cu -- --cuda-host-only -I /usr/local/cuda/include -I $BENCHMARKS/nw/needle.h

# 14. BFS
echo "====================================================="
echo "BFS"
$BUILD_DIR/bin/smc $BENCHMARKS/bfs/bfs.cu -- --cuda-host-only -I /usr/local/cuda/include
$BUILD_DIR/bin/smc $BENCHMARKS/bfs/kernel.cu -- --cuda-host-only -I /usr/local/cuda/include
$BUILD_DIR/bin/smc $BENCHMARKS/bfs/kernel2.cu -- --cuda-host-only -I /usr/local/cuda/include

# 15. NN
echo "====================================================="
echo "NN"
$BUILD_DIR/bin/smc $BENCHMARKS/nn/nn_cuda.cu -- --cuda-host-only -I /usr/local/cuda/include

# 16. b+ tree
echo "====================================================="
echo "b+ tree"
$BUILD_DIR/bin/smc $BENCHMARKS/b+tree/kernel/kernel_gpu_cuda.cu -- --cuda-host-only -I /usr/local/cuda/include -I $BENCHMARKS/b+tree/kernel/kernel_gpu_cuda_wrapper.h -I $BENCHMARKS/b+tree/kernel/kernel_gpu_cuda_wrapper_2.h -I $BENCHMARKS/b+tree/main.h -I $BENCHMARKS/b+tree/common.h
$BUILD_DIR/bin/smc $BENCHMARKS/b+tree/kernel/kernel_gpu_cuda_2.cu -- --cuda-host-only -I /usr/local/cuda/include -I $BENCHMARKS/b+tree/kernel/kernel_gpu_cuda_wrapper.h -I $BENCHMARKS/b+tree/kernel/kernel_gpu_cuda_wrapper_2.h -I $BENCHMARKS/b+tree/main.h -I $BENCHMARKS/b+tree/common.h
$BUILD_DIR/bin/smc $BENCHMARKS/b+tree/kernel/kernel_gpu_cuda_wrapper.cu -- --cuda-host-only -I /usr/local/cuda/include -I $BENCHMARKS/b+tree/kernel/kernel_gpu_cuda_wrapper.h -I $BENCHMARKS/b+tree/kernel/kernel_gpu_cuda_wrapper_2.h -I $BENCHMARKS/b+tree/main.h -I $BENCHMARKS/b+tree/common.h
$BUILD_DIR/bin/smc $BENCHMARKS/b+tree/kernel/kernel_gpu_cuda_wrapper_2.cu -- --cuda-host-only -I /usr/local/cuda/include -I $BENCHMARKS/b+tree/kernel/kernel_gpu_cuda_wrapper.h -I $BENCHMARKS/b+tree/kernel/kernel_gpu_cuda_wrapper_2.h -I $BENCHMARKS/b+tree/main.h -I $BENCHMARKS/b+tree/common.h

