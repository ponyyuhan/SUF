cmake_minimum_required(VERSION 3.15)
project(suf_fss LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_library(suf_fss INTERFACE)
target_include_directories(suf_fss INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(demo_relu src/demo/demo_relu.cpp)
target_link_libraries(demo_relu PRIVATE suf_fss)

add_executable(demo_gelu src/demo/demo_gelu.cpp)
target_link_libraries(demo_gelu PRIVATE suf_fss)

add_executable(demo_proto_reluars src/demo/demo_proto_reluars.cpp)
target_include_directories(demo_proto_reluars PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(demo_proto_gelu src/demo/demo_proto_gelu.cpp)
target_include_directories(demo_proto_gelu PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(demo_proto_endtoend src/demo/demo_proto_endtoend.cpp)
target_include_directories(demo_proto_endtoend PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(sim_harness src/demo/sim_harness.cpp)
target_include_directories(sim_harness PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
set_property(TARGET sim_harness PROPERTY CXX_STANDARD 20)
