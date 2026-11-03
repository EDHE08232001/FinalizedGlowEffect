cmake_minimum_required(VERSION 3.18)

# GPU architectures (must be decided before the CUDA language is enabled):
# "native" = the GPU(s) in this machine (CMake >= 3.24), otherwise a common set.
# Override with e.g. -DCMAKE_CUDA_ARCHITECTURES=86 (RTX 30xx) / 89 (RTX 40xx) / 75 (RTX 20xx) / 120 (RTX 50xx).
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT DEFINED ENV{CUDAARCHS})
    if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
        set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures")
    else()
        set(CMAKE_CUDA_ARCHITECTURES 75 86 CACHE STRING "CUDA architectures")
    endif()
endif()

project(GlowEffect
    VERSION 2.0.0
    DESCRIPTION "TensorRT segmentation + CUDA mipmap glow effect for images, videos and cameras"
    LANGUAGES CXX CUDA)

# ----------------------------------------------------------------------------------------------------------------------
# Options
# ----------------------------------------------------------------------------------------------------------------------
set(TensorRT_ROOT "$ENV{TENSORRT_ROOT}" CACHE PATH "TensorRT install folder (contains include/ and lib/)")
option(GLOW_COPY_RUNTIME_DLLS "Windows: copy OpenCV/TensorRT DLLs next to the executable" ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# ----------------------------------------------------------------------------------------------------------------------
# Dependencies
# ----------------------------------------------------------------------------------------------------------------------
find_package(CUDAToolkit REQUIRED)
find_package(TensorRT REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs videoio highgui)

message(STATUS "CUDA      : ${CUDAToolkit_VERSION} (archs: ${CMAKE_CUDA_ARCHITECTURES})")
message(STATUS "TensorRT  : ${TensorRT_VERSION} (${TensorRT_INCLUDE_DIR})")
message(STATUS "OpenCV    : ${OpenCV_VERSION} (${OpenCV_DIR})")

# ----------------------------------------------------------------------------------------------------------------------
# glow_core: segmentation + glow rendering library
# ----------------------------------------------------------------------------------------------------------------------
add_library(glow_core STATIC
    src/cuda_utils.cpp
    src/glow_params.cpp
    src/segmenter.cpp
    src/cuda/segmentation_kernels.cu
    src/cuda/glow_renderer.cu)

target_include_directories(glow_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(glow_core
    PUBLIC CUDA::cudart ${OpenCV_LIBS}
    PRIVATE TensorRT::TensorRT)
set_target_properties(glow_core PROPERTIES CUDA_SEPARABLE_COMPILATION OFF POSITION_INDEPENDENT_CODE ON)

if(MSVC)
    target_compile_options(glow_core PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/permissive- /Zc:__cplusplus /utf-8>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
    target_compile_definitions(glow_core PUBLIC NOMINMAX _CRT_SECURE_NO_WARNINGS)
else()
    target_compile_options(glow_core PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wno-deprecated-declarations>)
endif()
target_compile_options(glow_core PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -lineinfo>)

# ----------------------------------------------------------------------------------------------------------------------
# glow_effect: command line app + Studio UI
# ----------------------------------------------------------------------------------------------------------------------
add_executable(glow_effect
    apps/glow_effect/main.cpp
    apps/glow_effect/cli.cpp
    apps/glow_effect/studio_ui.cpp
    apps/glow_effect/image_mode.cpp
    apps/glow_effect/video_mode.cpp)
target_link_libraries(glow_effect PRIVATE glow_core)
if(MSVC)
    target_compile_options(glow_effect PRIVATE /permissive- /Zc:__cplusplus /utf-8)
    set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT glow_effect)
    set_target_properties(glow_effect PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
else()
    target_compile_options(glow_effect PRIVATE -Wall -Wextra)
endif()
find_package(Threads REQUIRED)
target_link_libraries(glow_effect PRIVATE Threads::Threads)

# Windows has no rpath: copy the DLLs of imported targets (OpenCV, TensorRT) next to glow_effect.exe.
if(WIN32 AND GLOW_COPY_RUNTIME_DLLS AND CMAKE_VERSION VERSION_GREATER_EQUAL 3.21)
    add_custom_command(TARGET glow_effect POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E $<IF:$<BOOL:$<TARGET_RUNTIME_DLLS:glow_effect>>,copy_if_different,true>
                $<TARGET_RUNTIME_DLLS:glow_effect> $<TARGET_FILE_DIR:glow_effect>
        COMMAND_EXPAND_LISTS
        COMMENT "Copying runtime DLLs next to glow_effect.exe")
endif()

# ----------------------------------------------------------------------------------------------------------------------
# Install: cmake --install build --prefix <dir>
# ----------------------------------------------------------------------------------------------------------------------
include(GNUInstallDirs)
install(TARGETS glow_effect RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(FILES models/README.md DESTINATION models)
