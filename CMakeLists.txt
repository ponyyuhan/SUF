cmake_minimum_required(VERSION 3.15)
project(suf_fss LANGUAGES C CXX)
enable_testing()

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --std=c++17 --allow-unsupported-compiler")
set(CMAKE_CUDA_HOST_COMPILER "/usr/bin/g++")
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
  set(CMAKE_CUDA_ARCHITECTURES "80;86")
endif()

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
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-std=gnu++2a>)
  set(CMAKE_CXX_STANDARD 17)
endif()
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_library(suf_fss INTERFACE)
target_include_directories(suf_fss INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

option(SUF_USE_MYL7_FSS "Enable myl7/fss backend adapter" ON)
option(SUF_FETCH_MYL7_FSS "Fetch myl7/fss from GitHub if not found" ON)
option(SUF_ENABLE_CUDA "Build CUDA PFSS stubs/backends" OFF)
set(HAVE_MYL7_FSS OFF)
set(HAVE_SUF_CUDA OFF)

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

if(SUF_ENABLE_CUDA)
  find_package(CUDAToolkit)
  if(CUDAToolkit_FOUND)
    enable_language(CUDA)
    add_library(cuda_pfss STATIC
      cuda/pfss_kernels.cu
      cuda/poly_kernels.cu
      cuda/cuda_primitives.cu
      cuda/pfss_backend_gpu.cu)
    target_include_directories(cuda_pfss PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/cuda ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_link_libraries(cuda_pfss PUBLIC CUDA::cudart OpenSSL::Crypto)
    set_target_properties(cuda_pfss PROPERTIES POSITION_INDEPENDENT_CODE ON)
    set(HAVE_SUF_CUDA ON)
  endif()
endif()

if(NOT HAVE_SUF_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    add_library(cuda_pfss STATIC
      cuda/pfss_kernels.cu
      cuda/poly_kernels.cu
      cuda/cuda_primitives.cu
      cuda/pfss_backend_gpu.cu)
    target_include_directories(cuda_pfss PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/cuda ${CMAKE_CURRENT_SOURCE_DIR}/include)
    if(TARGET CUDA::cudart)
      target_link_libraries(cuda_pfss PUBLIC CUDA::cudart OpenSSL::Crypto)
    else()
      target_link_libraries(cuda_pfss PUBLIC cudart OpenSSL::Crypto)
    endif()
    set_target_properties(cuda_pfss PROPERTIES POSITION_INDEPENDENT_CODE ON CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
    set_target_properties(cuda_pfss PROPERTIES POSITION_INDEPENDENT_CODE ON)
    set(HAVE_SUF_CUDA ON)
  endif()
endif()

if(NOT HAVE_SUF_CUDA)
  add_library(cuda_pfss STATIC
    cuda/pfss_stub.cpp
    cuda/pfss_backend_stub.cpp)
  target_include_directories(cuda_pfss PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/cuda ${CMAKE_CURRENT_SOURCE_DIR}/include)
  set_target_properties(cuda_pfss PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

if(HAVE_SUF_CUDA)
  add_compile_definitions(SUF_HAVE_CUDA=1)
endif()

add_executable(sim_harness src/demo/sim_harness.cpp)
target_include_directories(sim_harness PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(sim_harness PRIVATE suf_fss cuda_pfss ${MYL7_LINK_LIBS})
if (HAS_GNU_20)
  set_property(TARGET sim_harness PROPERTY CXX_STANDARD 20)
else()
  set_property(TARGET sim_harness PROPERTY CXX_STANDARD 17)
endif()

add_executable(test_suf_ref_eval src/demo/test_suf_ref_eval.cpp)
target_include_directories(test_suf_ref_eval PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_suf_ref_eval PRIVATE suf_fss cuda_pfss ${MYL7_LINK_LIBS})

add_executable(test_mask_rewrite src/demo/test_mask_rewrite.cpp)
target_include_directories(test_mask_rewrite PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_mask_rewrite PRIVATE suf_fss cuda_pfss ${MYL7_LINK_LIBS})

add_executable(test_compile_pfss src/demo/test_compile_pfss.cpp)
target_include_directories(test_compile_pfss PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_compile_pfss PRIVATE compiler_passes suf_fss cuda_pfss ${MYL7_LINK_LIBS})

add_executable(test_gapars_selector src/demo/test_gapars_selector.cpp)
target_include_directories(test_gapars_selector PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_gapars_selector PRIVATE compiler_passes suf_fss cuda_pfss ${MYL7_LINK_LIBS})

add_library(compiler_passes STATIC
  src/compiler/truncation_pass.cpp
  src/compiler/rescale_pass.cpp
  src/compiler/layer_graph.cpp
  src/compiler/suf_to_pfss.cpp)
target_include_directories(compiler_passes PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(compiler_passes PUBLIC suf_fss cuda_pfss ${MYL7_LINK_LIBS})

add_executable(test_sigmafast src/demo/test_sigmafast.cpp)
target_include_directories(test_sigmafast PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_sigmafast PRIVATE suf_fss cuda_pfss ${MYL7_LINK_LIBS})

add_executable(test_composite_runtime src/demo/test_composite_runtime.cpp)
target_include_directories(test_composite_runtime PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_composite_runtime PRIVATE nn_ops compiler_passes suf_fss cuda_pfss ${MYL7_LINK_LIBS})

add_executable(test_pred_semantics src/demo/test_pred_semantics.cpp)
target_include_directories(test_pred_semantics PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_pred_semantics PRIVATE suf_fss cuda_pfss ${MYL7_LINK_LIBS})

add_executable(test_composite_equiv_proto src/demo/test_composite_equiv_proto.cpp)
target_include_directories(test_composite_equiv_proto PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_composite_equiv_proto PRIVATE compiler_passes suf_fss cuda_pfss ${MYL7_LINK_LIBS})

add_executable(test_composite_equiv_proto_myl7 src/demo/test_composite_equiv_proto_myl7.cpp)
target_include_directories(test_composite_equiv_proto_myl7 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_composite_equiv_proto_myl7 PRIVATE compiler_passes suf_fss cuda_pfss ${MYL7_LINK_LIBS})

add_executable(test_composite_equiv_proto_sigmafast src/demo/test_composite_equiv_proto_sigmafast.cpp)
target_include_directories(test_composite_equiv_proto_sigmafast PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_composite_equiv_proto_sigmafast PRIVATE compiler_passes suf_fss cuda_pfss ${MYL7_LINK_LIBS})

add_executable(test_postproc_hooks src/demo/test_postproc_hooks.cpp)
target_include_directories(test_postproc_hooks PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_postproc_hooks PRIVATE suf_fss cuda_pfss ${MYL7_LINK_LIBS})

add_executable(test_llm_gates src/demo/test_llm_gates.cpp)
target_include_directories(test_llm_gates PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_llm_gates PRIVATE suf_fss cuda_pfss ${MYL7_LINK_LIBS})

add_executable(bench_sigmafast_pred src/bench/bench_sigmafast_pred.cpp)
target_include_directories(bench_sigmafast_pred PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(bench_sigmafast_pred PRIVATE suf_fss ${MYL7_LINK_LIBS})

add_executable(bench_sigmafast_coeff src/bench/bench_sigmafast_coeff.cpp)
target_include_directories(bench_sigmafast_coeff PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(bench_sigmafast_coeff PRIVATE suf_fss ${MYL7_LINK_LIBS})

add_executable(bench_sigmafast_gates src/bench/bench_sigmafast_gates.cpp)
target_include_directories(bench_sigmafast_gates PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(bench_sigmafast_gates PRIVATE compiler_passes suf_fss ${MYL7_LINK_LIBS})

add_library(nn_ops STATIC
  src/nn/linops.cpp
  src/nn/matmul_publicW.cpp
  src/nn/matmul_beaver.cpp
  $<$<BOOL:${HAVE_SUF_CUDA}>:${CMAKE_CURRENT_SOURCE_DIR}/src/nn/matmul_gpu.cu>
  $<$<BOOL:${HAVE_SUF_CUDA}>:${CMAKE_CURRENT_SOURCE_DIR}/src/runtime/pfss_gpu_stager_cuda.cpp>
  src/nn/attention_block.cpp
  src/nn/mlp_block.cpp
  src/nn/transformer_layer.cpp
  src/gates/postproc_hooks.cpp
  src/runtime/pfss_superbatch.cpp
  src/runtime/open_collector.cpp
  src/runtime/phase_executor.cpp)
target_include_directories(nn_ops PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(nn_ops PUBLIC suf_fss compiler_passes ${MYL7_LINK_LIBS})
if(HAVE_SUF_CUDA)
  target_link_libraries(nn_ops PUBLIC CUDA::cudart)
endif()

add_executable(bench_llm_gates src/bench/bench_llm_gates.cpp)
target_include_directories(bench_llm_gates PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(bench_llm_gates PRIVATE suf_fss ${MYL7_LINK_LIBS})

add_executable(bench_attention src/bench/bench_attention.cpp)
target_include_directories(bench_attention PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(bench_attention PRIVATE nn_ops suf_fss ${MYL7_LINK_LIBS})

add_executable(test_matmul_and_attention src/demo/test_matmul_and_attention.cpp)
target_include_directories(test_matmul_and_attention PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_matmul_and_attention PRIVATE nn_ops suf_fss ${MYL7_LINK_LIBS})

add_executable(test_matmul src/demo/test_matmul.cpp)
target_include_directories(test_matmul PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_matmul PRIVATE nn_ops compiler_passes suf_fss ${MYL7_LINK_LIBS})

add_executable(test_truncation src/demo/test_truncation.cpp)
target_include_directories(test_truncation PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_truncation PRIVATE compiler_passes suf_fss ${MYL7_LINK_LIBS})

add_executable(test_softmax_task src/demo/test_softmax_task.cpp)
target_include_directories(test_softmax_task PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_softmax_task PRIVATE nn_ops compiler_passes suf_fss ${MYL7_LINK_LIBS})

add_executable(test_trunc_task src/demo/test_trunc_task.cpp)
target_include_directories(test_trunc_task PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_trunc_task PRIVATE nn_ops compiler_passes suf_fss ${MYL7_LINK_LIBS})
add_executable(test_staged_trunc src/demo/test_staged_trunc.cpp)
target_include_directories(test_staged_trunc PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_staged_trunc PRIVATE nn_ops compiler_passes suf_fss ${MYL7_LINK_LIBS})
add_executable(test_staged_softmax src/demo/test_staged_softmax.cpp)
target_include_directories(test_staged_softmax PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_staged_softmax PRIVATE nn_ops compiler_passes suf_fss ${MYL7_LINK_LIBS})

add_executable(test_layernorm_task src/demo/test_layernorm_task.cpp)
target_include_directories(test_layernorm_task PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_layernorm_task PRIVATE nn_ops compiler_passes suf_fss ${MYL7_LINK_LIBS})

add_executable(test_pfss_planner src/demo/test_pfss_planner.cpp)
target_include_directories(test_pfss_planner PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_pfss_planner PRIVATE nn_ops compiler_passes suf_fss ${MYL7_LINK_LIBS})

add_executable(test_pfss_async src/demo/test_pfss_async.cpp)
target_include_directories(test_pfss_async PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_pfss_async PRIVATE nn_ops compiler_passes suf_fss ${MYL7_LINK_LIBS})

add_executable(test_pfss_async_stress src/demo/test_pfss_async_stress.cpp)
target_include_directories(test_pfss_async_stress PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_pfss_async_stress PRIVATE nn_ops compiler_passes suf_fss ${MYL7_LINK_LIBS})

add_executable(test_transformer_async src/demo/test_transformer_async.cpp)
target_include_directories(test_transformer_async PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_transformer_async PRIVATE nn_ops compiler_passes suf_fss ${MYL7_LINK_LIBS})

add_executable(test_planner_causal src/demo/test_planner_causal.cpp)
target_include_directories(test_planner_causal PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_planner_causal PRIVATE nn_ops compiler_passes suf_fss ${MYL7_LINK_LIBS})

add_executable(test_planner_causal_bytes src/demo/test_planner_causal_bytes.cpp)
target_include_directories(test_planner_causal_bytes PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_planner_causal_bytes PRIVATE nn_ops compiler_passes suf_fss ${MYL7_LINK_LIBS})

if(HAVE_SUF_CUDA)
add_executable(test_pfss_gpu src/demo/test_pfss_gpu.cpp)
target_include_directories(test_pfss_gpu PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_pfss_gpu PRIVATE nn_ops compiler_passes suf_fss ${MYL7_LINK_LIBS})

add_executable(test_cuda_trunc_gapars src/demo/test_cuda_trunc_gapars.cpp)
target_include_directories(test_cuda_trunc_gapars PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_cuda_trunc_gapars PRIVATE nn_ops compiler_passes suf_fss cuda_pfss OpenSSL::Crypto)

add_executable(test_cuda_prg src/demo/test_cuda_prg.cpp)
target_include_directories(test_cuda_prg PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_cuda_prg PRIVATE suf_fss cuda_pfss OpenSSL::Crypto)

add_executable(test_cuda_pack_effbits src/demo/test_cuda_pack_effbits.cpp)
target_include_directories(test_cuda_pack_effbits PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_cuda_pack_effbits PRIVATE suf_fss cuda_pfss)
add_test(NAME test_cuda_pack_effbits COMMAND test_cuda_pack_effbits)

add_executable(test_cuda_pred_mask src/demo/test_cuda_pred_mask.cpp)
target_include_directories(test_cuda_pred_mask PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_cuda_pred_mask PRIVATE suf_fss cuda_pfss)

add_executable(test_cuda_packed_pfss src/demo/test_cuda_packed_pfss.cpp)
target_include_directories(test_cuda_packed_pfss PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_cuda_packed_pfss PRIVATE suf_fss cuda_pfss OpenSSL::Crypto)

add_executable(test_cuda_beaver_mul src/demo/test_cuda_beaver_mul.cpp)
target_include_directories(test_cuda_beaver_mul PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_cuda_beaver_mul PRIVATE suf_fss cuda_pfss OpenSSL::Crypto)

add_executable(test_cuda_trunc_postproc src/demo/test_cuda_trunc_postproc.cpp)
target_include_directories(test_cuda_trunc_postproc PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_cuda_trunc_postproc PRIVATE suf_fss cuda_pfss OpenSSL::Crypto)

add_executable(test_cuda_horner_cubic src/demo/test_cuda_horner_cubic.cpp)
target_include_directories(test_cuda_horner_cubic PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_cuda_horner_cubic PRIVATE suf_fss cuda_pfss OpenSSL::Crypto)

add_executable(test_cuda_recip_task src/demo/test_cuda_recip_task.cpp)
target_include_directories(test_cuda_recip_task PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_cuda_recip_task PRIVATE nn_ops compiler_passes suf_fss cuda_pfss OpenSSL::Crypto)

add_executable(test_softmax_gpu_smoke src/demo/test_softmax_gpu_smoke.cpp)
target_include_directories(test_softmax_gpu_smoke PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_softmax_gpu_smoke PRIVATE nn_ops compiler_passes suf_fss cuda_pfss ${MYL7_LINK_LIBS})
add_test(NAME test_softmax_gpu_smoke COMMAND test_softmax_gpu_smoke)
endif()

add_executable(bench_gemm src/bench/bench_gemm.cpp)
target_include_directories(bench_gemm PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(bench_gemm PRIVATE nn_ops suf_fss ${MYL7_LINK_LIBS})

if(HAVE_SUF_CUDA)
add_executable(bench_gemm_overlap src/bench/bench_gemm_overlap.cpp)
target_include_directories(bench_gemm_overlap PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(bench_gemm_overlap PRIVATE nn_ops suf_fss cuda_pfss OpenSSL::Crypto)
add_test(NAME bench_gemm_overlap COMMAND bench_gemm_overlap)

add_executable(bench_pfss_cpu_gpu src/bench/bench_pfss_cpu_gpu.cpp)
target_include_directories(bench_pfss_cpu_gpu PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(bench_pfss_cpu_gpu PRIVATE nn_ops compiler_passes suf_fss cuda_pfss OpenSSL::Crypto ${MYL7_LINK_LIBS})

add_executable(bench_softmax_norm src/bench/bench_softmax_norm.cpp)
target_include_directories(bench_softmax_norm PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(bench_softmax_norm PRIVATE nn_ops compiler_passes suf_fss cuda_pfss OpenSSL::Crypto ${MYL7_LINK_LIBS})
endif()

add_executable(bench_truncation src/bench/bench_truncation.cpp)
target_include_directories(bench_truncation PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(bench_truncation PRIVATE suf_fss ${MYL7_LINK_LIBS})

if(HAVE_MYL7_FSS)
add_executable(test_myl7_bit_order src/demo/test_myl7_bit_order.cpp)
  target_include_directories(test_myl7_bit_order PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(test_myl7_bit_order PRIVATE suf_fss ${MYL7_LINK_LIBS})
endif()

add_test(NAME test_trunc_task COMMAND test_trunc_task)
add_test(NAME test_layernorm_task COMMAND test_layernorm_task)
add_test(NAME test_staged_softmax COMMAND test_staged_softmax)
add_test(NAME test_pfss_planner COMMAND test_pfss_planner)
add_test(NAME test_pfss_async COMMAND test_pfss_async)
add_test(NAME test_pfss_async_stress COMMAND test_pfss_async_stress)
add_test(NAME test_transformer_async COMMAND test_transformer_async)
add_test(NAME test_planner_causal COMMAND test_planner_causal)
add_test(NAME test_gapars_selector COMMAND test_gapars_selector)
if(HAVE_SUF_CUDA)
add_test(NAME test_pfss_gpu COMMAND test_pfss_gpu)
add_test(NAME test_cuda_trunc_gapars COMMAND test_cuda_trunc_gapars)
add_test(NAME test_cuda_prg COMMAND test_cuda_prg)
add_test(NAME test_cuda_pred_mask COMMAND test_cuda_pred_mask)
add_test(NAME test_cuda_packed_pfss COMMAND test_cuda_packed_pfss)
add_test(NAME test_cuda_beaver_mul COMMAND test_cuda_beaver_mul)
add_test(NAME test_cuda_trunc_postproc COMMAND test_cuda_trunc_postproc)
add_test(NAME test_cuda_horner_cubic COMMAND test_cuda_horner_cubic)
add_test(NAME test_cuda_recip_task COMMAND test_cuda_recip_task)
endif()

# Placeholder executor coverage for MLP/attention (reuse existing binaries for smoke until dedicated harnesses are added).
add_test(NAME test_matmul_and_attention_executor COMMAND test_matmul_and_attention)
add_test(NAME test_matmul_executor COMMAND test_matmul)
