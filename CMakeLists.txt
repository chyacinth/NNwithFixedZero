cmake_minimum_required(VERSION 3.6)
project(NNwithFixedZero)
#add_definitions(-DGPU=1)
#add_definitions(-DFIXED_ZERO=1)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES
    src/activation.cpp
    src/activation.h
    src/CMakeLists.txt
    src/fault.cpp
    src/fault.h
    src/fullyconnected_layer.cpp
    src/fullyconnected_layer.h
    src/fullyconnected_layer_kernel.cu
    src/layer.h
    src/main.cpp
    src/main2.cpp
    src/mnist_parser.h
    src/network.h
    src/output_layer.cpp
    src/output_layer.h
    src/output_layer_kernel.cu
    src/t10k-images.idx3-ubyte
    src/t10k-labels.idx1-ubyte
    src/train-images.idx3-ubyte
    src/train-labels.idx1-ubyte
    src/util.cpp
    src/util.h
    src/weight.txt)
find_package(Boost 1.62.0 COMPONENTS system filesystem REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})
add_executable(NNwithFixedZero ${SOURCE_FILES})
target_link_libraries(NNwithFixedZero ${Boost_LIBRARIES})

