cmake_minimum_required(VERSION 3.6)
project(src)
#add_definitions(-DGPU=1)
#add_definitions(-DFIXED_ZERO=1)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES
    activation.cpp
    activation.h
    fault.cpp
    fault.h
    fullyconnected_layer.cpp
    fullyconnected_layer.h
    fullyconnected_layer_kernel.cu
    layer.h
    main.cpp
    main2.cpp
    mnist_parser.h
    network.h
    output_layer.cpp
    output_layer.h
    output_layer_kernel.cu
    util.cpp
    util.h)
find_package(Boost 1.62.0 COMPONENTS system filesystem REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})
add_executable(src ${SOURCE_FILES})
target_link_libraries(src ${Boost_LIBRARIES})
