cmake_minimum_required(VERSION 3.15)
project(suf_fss LANGUAGES C CXX)

set(CMAKE_CXX_EXTENSIONS ON)
# OpenMP (needed for myl7/fss); if missing, targets will not link.
find_package(OpenMP)
find_package(OpenSSL REQUIRED)

# Prefer C++20; fall back to gnu++2a on older toolchains that lack -std=gnu++20.
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-std=gnu++20" HAS_GNU_20)
if (HAS_GNU_20)
  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
else()
  add_compile_options(-std=gnu++2a)
  set(CMAKE_CXX_STANDARD 17)
endif()
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_library(suf_fss INTERFACE)
target_include_directories(suf_fss INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

option(SUF_USE_MYL7_FSS "Enable myl7/fss backend adapter" ON)
option(SUF_FETCH_MYL7_FSS "Fetch myl7/fss from GitHub if not found" ON)
set(HAVE_MYL7_FSS OFF)

if(SUF_USE_MYL7_FSS)
  find_path(MYL7_FSS_INCLUDE_DIR fss/dcf.h HINTS ${CMAKE_CURRENT_SOURCE_DIR}/third_party/myl7_fss/include)
  find_library(MYL7_FSS_LIB_DCF dcf HINTS ${CMAKE_CURRENT_SOURCE_DIR}/third_party/myl7_fss/build ${CMAKE_CURRENT_SOURCE_DIR}/third_party/myl7_fss/build/lib ${CMAKE_CURRENT_SOURCE_DIR}/third_party/myl7_fss/build/lib64 PATH_SUFFIXES lib)
  find_library(MYL7_FSS_LIB_DPF dpf HINTS ${CMAKE_CURRENT_SOURCE_DIR}/third_party/myl7_fss/build ${CMAKE_CURRENT_SOURCE_DIR}/third_party/myl7_fss/build/lib ${CMAKE_CURRENT_SOURCE_DIR}/third_party/myl7_fss/build/lib64 PATH_SUFFIXES lib)
  find_library(MYL7_FSS_LIB_CW cw_mac_bytes HINTS ${CMAKE_CURRENT_SOURCE_DIR}/third_party/myl7_fss/build ${CMAKE_CURRENT_SOURCE_DIR}/third_party/myl7_fss/build/lib ${CMAKE_CURRENT_SOURCE_DIR}/third_party/myl7_fss/build/lib64 PATH_SUFFIXES lib)
  if(NOT MYL7_FSS_LIB_DCF AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/myl7_fss/build/libdcf.a")
    set(MYL7_FSS_LIB_DCF "${CMAKE_CURRENT_SOURCE_DIR}/third_party/myl7_fss/build/libdcf.a")
  endif()
  if(NOT MYL7_FSS_LIB_DPF AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/myl7_fss/build/libdpf.a")
    set(MYL7_FSS_LIB_DPF "${CMAKE_CURRENT_SOURCE_DIR}/third_party/myl7_fss/build/libdpf.a")
  endif()
  if(NOT MYL7_FSS_LIB_CW AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/myl7_fss/build/libcw_mac_bytes.a")
    set(MYL7_FSS_LIB_CW "${CMAKE_CURRENT_SOURCE_DIR}/third_party/myl7_fss/build/libcw_mac_bytes.a")
  endif()
  if((NOT MYL7_FSS_INCLUDE_DIR OR NOT MYL7_FSS_LIB_DCF) AND SUF_FETCH_MYL7_FSS)
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
  if(MYL7_FSS_INCLUDE_DIR AND MYL7_FSS_LIB_DCF)
    set(HAVE_MYL7_FSS ON)
    add_library(myl7_core STATIC
      third_party/myl7_fss/src/group/bytes.c
      third_party/myl7_fss/src/group/u128_le.c
      third_party/myl7_fss/src/prg/aes128_mmo.c
    )
    set_target_properties(myl7_core PROPERTIES LINKER_LANGUAGE C)
    target_compile_definitions(myl7_core PUBLIC kBlocks=8)
    target_include_directories(myl7_core PUBLIC ${MYL7_FSS_INCLUDE_DIR})
    target_compile_definitions(myl7_core PUBLIC kLambda=16)
    target_link_libraries(myl7_core PUBLIC OpenSSL::Crypto)
    if(OpenMP_FOUND)
      if(TARGET OpenMP::OpenMP_C)
        target_link_libraries(myl7_core PUBLIC OpenMP::OpenMP_C)
      endif()
    endif()
    target_include_directories(suf_fss INTERFACE ${MYL7_FSS_INCLUDE_DIR})
    target_link_directories(suf_fss INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/third_party/myl7_fss/build ${CMAKE_CURRENT_SOURCE_DIR}/third_party/myl7_fss/build/lib ${CMAKE_CURRENT_SOURCE_DIR}/third_party/myl7_fss)
    if(OpenMP_FOUND)
      if(TARGET OpenMP::OpenMP_CXX)
        target_link_libraries(suf_fss INTERFACE OpenMP::OpenMP_CXX)
      elseif(TARGET OpenMP::OpenMP_C)
        target_link_libraries(suf_fss INTERFACE OpenMP::OpenMP_C)
      endif()
    endif()
    target_link_libraries(suf_fss INTERFACE ${MYL7_FSS_LIB_DCF})
    if(MYL7_FSS_LIB_DPF)
      target_link_libraries(suf_fss INTERFACE ${MYL7_FSS_LIB_DPF})
    endif()
    if(MYL7_FSS_LIB_CW)
      target_link_libraries(suf_fss INTERFACE ${MYL7_FSS_LIB_CW})
    endif()
    # Propagate to all executables as well (some compilers don’t honor INTERFACE static deps)
    set(MYL7_LINK_LIBS ${MYL7_FSS_LIB_DCF} ${MYL7_FSS_LIB_DPF} ${MYL7_FSS_LIB_CW} myl7_core OpenSSL::Crypto)
  endif()
endif()

add_executable(sim_harness src/demo/sim_harness.cpp)
target_include_directories(sim_harness PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(sim_harness PRIVATE suf_fss ${MYL7_LINK_LIBS})
if (HAS_GNU_20)
  set_property(TARGET sim_harness PROPERTY CXX_STANDARD 20)
else()
  set_property(TARGET sim_harness PROPERTY CXX_STANDARD 17)
endif()

add_executable(test_suf_ref_eval src/demo/test_suf_ref_eval.cpp)
target_include_directories(test_suf_ref_eval PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_suf_ref_eval PRIVATE suf_fss ${MYL7_LINK_LIBS})

add_executable(test_mask_rewrite src/demo/test_mask_rewrite.cpp)
target_include_directories(test_mask_rewrite PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_mask_rewrite PRIVATE suf_fss ${MYL7_LINK_LIBS})

add_executable(test_compile_pfss src/demo/test_compile_pfss.cpp src/compiler/suf_to_pfss.cpp)
target_include_directories(test_compile_pfss PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_compile_pfss PRIVATE suf_fss ${MYL7_LINK_LIBS})

add_executable(test_sigmafast src/demo/test_sigmafast.cpp)
target_include_directories(test_sigmafast PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_sigmafast PRIVATE suf_fss ${MYL7_LINK_LIBS})

add_executable(test_composite_runtime src/demo/test_composite_runtime.cpp src/compiler/suf_to_pfss.cpp)
target_include_directories(test_composite_runtime PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_composite_runtime PRIVATE suf_fss ${MYL7_LINK_LIBS})

add_executable(test_pred_semantics src/demo/test_pred_semantics.cpp)
target_include_directories(test_pred_semantics PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_pred_semantics PRIVATE suf_fss ${MYL7_LINK_LIBS})

add_executable(test_composite_equiv_proto src/demo/test_composite_equiv_proto.cpp src/compiler/suf_to_pfss.cpp)
target_include_directories(test_composite_equiv_proto PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_composite_equiv_proto PRIVATE suf_fss ${MYL7_LINK_LIBS})

add_executable(test_composite_equiv_proto_myl7 src/demo/test_composite_equiv_proto_myl7.cpp src/compiler/suf_to_pfss.cpp)
target_include_directories(test_composite_equiv_proto_myl7 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_composite_equiv_proto_myl7 PRIVATE suf_fss ${MYL7_LINK_LIBS})

add_executable(test_composite_equiv_proto_sigmafast src/demo/test_composite_equiv_proto_sigmafast.cpp src/compiler/suf_to_pfss.cpp)
target_include_directories(test_composite_equiv_proto_sigmafast PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_composite_equiv_proto_sigmafast PRIVATE suf_fss ${MYL7_LINK_LIBS})

add_executable(test_postproc_hooks src/demo/test_postproc_hooks.cpp)
target_include_directories(test_postproc_hooks PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_postproc_hooks PRIVATE suf_fss ${MYL7_LINK_LIBS})

add_executable(bench_sigmafast_pred src/bench/bench_sigmafast_pred.cpp)
target_include_directories(bench_sigmafast_pred PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(bench_sigmafast_pred PRIVATE suf_fss ${MYL7_LINK_LIBS})

add_executable(bench_sigmafast_coeff src/bench/bench_sigmafast_coeff.cpp)
target_include_directories(bench_sigmafast_coeff PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(bench_sigmafast_coeff PRIVATE suf_fss ${MYL7_LINK_LIBS})

add_executable(bench_sigmafast_gates src/bench/bench_sigmafast_gates.cpp src/compiler/suf_to_pfss.cpp)
target_include_directories(bench_sigmafast_gates PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(bench_sigmafast_gates PRIVATE suf_fss ${MYL7_LINK_LIBS})

if(HAVE_MYL7_FSS)
add_executable(test_myl7_bit_order src/demo/test_myl7_bit_order.cpp)
target_include_directories(test_myl7_bit_order PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_myl7_bit_order PRIVATE suf_fss ${MYL7_LINK_LIBS})
endif()
