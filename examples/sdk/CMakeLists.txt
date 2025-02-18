# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Example CMakeLists.txt for building executor_runner with sdk support. In this
# example we link sdk and bundled_program libraries into executor_runner binary
cmake_minimum_required(VERSION 3.19)
project(sdk_example)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(NOT PYTHON_EXECUTABLE)
  set(PYTHON_EXECUTABLE python3)
endif()
# Source root directory for executorch.
if(NOT EXECUTORCH_ROOT)
  set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)
endif()
# Source root directory for pytorch.
if(NOT TORCH_ROOT)
  set(TORCH_ROOT ${EXECUTORCH_ROOT}/third-party/pytorch)
endif()

include(${EXECUTORCH_ROOT}/build/Utils.cmake)
include(${EXECUTORCH_ROOT}/build/Codegen.cmake)

set(_common_compile_options -Wno-deprecated-declarations -fPIC)

# Let files say "include <executorch/path/to/header.h>".
set(_common_include_directories ${EXECUTORCH_ROOT}/..)

# Find prebuilt libraries. executorch package should contain
# portable_ops_lib, etdump, bundled_program.
find_package(executorch CONFIG REQUIRED)
target_include_directories(executorch INTERFACE ${_common_include_directories})

find_package(
  gflags REQUIRED PATHS ${CMAKE_CURRENT_BINARY_DIR}/../../third-party
)

add_executable(sdk_example_runner
               sdk_example_runner/sdk_example_runner.cpp)
target_compile_options(executorch INTERFACE -DET_EVENT_TRACER_ENABLED)

# portable_ops_lib
gen_selected_ops("" "" "ON")
# Expect gen_selected_ops output file to be selected_operators.yaml
generate_bindings_for_kernels(
  ${EXECUTORCH_ROOT}/kernels/portable/functions.yaml ""
)
gen_operators_lib("portable_ops_lib" portable_kernels executorch)

target_compile_options(portable_ops_lib INTERFACE -DET_EVENT_TRACER_ENABLED)
target_include_directories(
  etdump
  INTERFACE
  ${CMAKE_CURRENT_BINARY_DIR}/../../sdk/include
  ${EXECUTORCH_ROOT}/third-party/flatcc/include
)
target_link_libraries(
  sdk_example_runner
  executorch
  gflags
  etdump
  extension_data_loader
  flatcc
  bundled_program
  portable_ops_lib)
