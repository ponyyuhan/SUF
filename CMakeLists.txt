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
