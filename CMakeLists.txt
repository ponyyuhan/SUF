cmake_minimum_required(VERSION 3.15)
project(suf_fss LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_library(suf_fss INTERFACE)
target_include_directories(suf_fss INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

option(SUF_USE_MYL7_FSS "Enable myl7/fss backend adapter" ON)
option(SUF_FETCH_MYL7_FSS "Fetch myl7/fss from GitHub if not found" ON)
set(HAVE_MYL7_FSS OFF)

if(SUF_USE_MYL7_FSS)
  find_path(MYL7_FSS_INCLUDE_DIR fss/dcf.h)
  find_library(MYL7_FSS_LIB dcf)
  if((NOT MYL7_FSS_INCLUDE_DIR OR NOT MYL7_FSS_LIB) AND SUF_FETCH_MYL7_FSS)
    # Force-disable heavy optional parts in the upstream build.
    set(WITH_CUDA OFF CACHE BOOL "" FORCE)
    set(WITH_TEST OFF CACHE BOOL "" FORCE)
    set(WITH_SAMPLES OFF CACHE BOOL "" FORCE)
    include(FetchContent)
    FetchContent_Declare(
      myl7_fss
      GIT_REPOSITORY https://github.com/myl7/fss
      GIT_TAG v0.7.1
      CMAKE_ARGS -DWITH_CUDA=OFF -DWITH_TEST=OFF -DWITH_SAMPLES=OFF
    )
    FetchContent_MakeAvailable(myl7_fss)
    set(MYL7_FSS_INCLUDE_DIR ${myl7_fss_SOURCE_DIR}/include)
    set(MYL7_FSS_LIB dcf)
  endif()
  if(MYL7_FSS_INCLUDE_DIR AND MYL7_FSS_LIB)
    set(HAVE_MYL7_FSS ON)
    target_include_directories(suf_fss INTERFACE ${MYL7_FSS_INCLUDE_DIR})
    target_link_libraries(suf_fss INTERFACE ${MYL7_FSS_LIB})
  endif()
endif()

add_executable(sim_harness src/demo/sim_harness.cpp)
target_include_directories(sim_harness PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(sim_harness PRIVATE suf_fss)
set_property(TARGET sim_harness PROPERTY CXX_STANDARD 20)

add_executable(test_suf_ref_eval src/demo/test_suf_ref_eval.cpp)
target_include_directories(test_suf_ref_eval PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_suf_ref_eval PRIVATE suf_fss)

add_executable(test_mask_rewrite src/demo/test_mask_rewrite.cpp)
target_include_directories(test_mask_rewrite PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_mask_rewrite PRIVATE suf_fss)

add_executable(test_compile_pfss src/demo/test_compile_pfss.cpp src/compiler/suf_to_pfss.cpp)
target_include_directories(test_compile_pfss PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_compile_pfss PRIVATE suf_fss)
