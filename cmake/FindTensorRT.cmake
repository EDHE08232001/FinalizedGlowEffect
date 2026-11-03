# FindTensorRT
# ------------
# Locates NVIDIA TensorRT (nvinfer + nvonnxparser).
#
# Hints (CMake variable or environment variable):
#   TensorRT_ROOT / TENSORRT_ROOT   folder that contains include/ and lib/ (e.g. C:/TensorRT-10.x or /opt/TensorRT)
#
# Result:
#   TensorRT_FOUND, TensorRT_VERSION, TensorRT_INCLUDE_DIR
#   Imported targets: TensorRT::nvinfer, TensorRT::nvonnxparser, TensorRT::TensorRT (both)

set(_trt_hints
    ${TensorRT_ROOT} $ENV{TensorRT_ROOT}
    ${TENSORRT_ROOT} $ENV{TENSORRT_ROOT}
    ${TensorRT_DIR}
    /usr/local/tensorrt /opt/tensorrt /usr/local/TensorRT /opt/TensorRT
    "C:/Program Files/NVIDIA/TensorRT" "C:/TensorRT")

# Windows zip installs are usually versioned folders, e.g. C:/TensorRT-10.4.0.26 or C:/Program Files/NVIDIA/TensorRT-8.6.1.6
if(WIN32)
    file(GLOB _trt_versioned LIST_DIRECTORIES true
         "C:/TensorRT-*" "C:/Program Files/NVIDIA/TensorRT-*" "C:/Program Files/NVIDIA GPU Computing Toolkit/TensorRT*")
    list(SORT _trt_versioned ORDER DESCENDING)
    list(APPEND _trt_hints ${_trt_versioned})
endif()

find_path(TensorRT_INCLUDE_DIR NvInfer.h
    HINTS ${_trt_hints}
    PATH_SUFFIXES include include/x86_64-linux-gnu include/aarch64-linux-gnu)

# TensorRT 10 on Windows names the libraries nvinfer_10.lib / nvonnxparser_10.lib.
find_library(TensorRT_nvinfer_LIBRARY NAMES nvinfer nvinfer_10 nvinfer_11
    HINTS ${_trt_hints}
    PATH_SUFFIXES lib lib64 lib/x64 lib/x86_64-linux-gnu lib/aarch64-linux-gnu)
find_library(TensorRT_nvonnxparser_LIBRARY NAMES nvonnxparser nvonnxparser_10 nvonnxparser_11
    HINTS ${_trt_hints}
    PATH_SUFFIXES lib lib64 lib/x64 lib/x86_64-linux-gnu lib/aarch64-linux-gnu)

if(TensorRT_INCLUDE_DIR AND EXISTS "${TensorRT_INCLUDE_DIR}/NvInferVersion.h")
    file(STRINGS "${TensorRT_INCLUDE_DIR}/NvInferVersion.h" _trt_ver_lines REGEX "#define (NV_TENSORRT|TRT)_(MAJOR|MINOR|PATCH)(_[A-Z]+)? +[0-9]+")
    foreach(_part MAJOR MINOR PATCH)
        string(REGEX MATCH "#define (NV_TENSORRT|TRT)_${_part}(_[A-Z]+)? +([0-9]+)" _m "${_trt_ver_lines}")
        set(_trt_${_part} "${CMAKE_MATCH_3}")
    endforeach()
    if(NOT "${_trt_MAJOR}" STREQUAL "")
        set(TensorRT_VERSION "${_trt_MAJOR}.${_trt_MINOR}.${_trt_PATCH}")
    endif()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(TensorRT
    REQUIRED_VARS TensorRT_INCLUDE_DIR TensorRT_nvinfer_LIBRARY TensorRT_nvonnxparser_LIBRARY
    VERSION_VAR TensorRT_VERSION
    FAIL_MESSAGE "TensorRT not found. Pass -DTensorRT_ROOT=<path to TensorRT> (the folder containing include/ and lib/), or set the TENSORRT_ROOT environment variable.")

if(TensorRT_FOUND)
    foreach(_lib nvinfer nvonnxparser)
        if(NOT TARGET TensorRT::${_lib})
            set(_implib "${TensorRT_${_lib}_LIBRARY}")
            if(WIN32)
                # Record the DLL so $<TARGET_RUNTIME_DLLS> can copy it next to the executable.
                get_filename_component(_libname "${_implib}" NAME_WE)
                get_filename_component(_libdir "${_implib}" DIRECTORY)
                find_file(TensorRT_${_lib}_DLL "${_libname}.dll"
                    HINTS "${_libdir}/../bin" "${_libdir}/../lib" "${_libdir}" NO_DEFAULT_PATH)
            endif()
            if(WIN32 AND TensorRT_${_lib}_DLL)
                add_library(TensorRT::${_lib} SHARED IMPORTED)
                set_target_properties(TensorRT::${_lib} PROPERTIES
                    IMPORTED_IMPLIB "${_implib}"
                    IMPORTED_LOCATION "${TensorRT_${_lib}_DLL}")
            else()
                add_library(TensorRT::${_lib} UNKNOWN IMPORTED)
                set_target_properties(TensorRT::${_lib} PROPERTIES IMPORTED_LOCATION "${_implib}")
            endif()
            set_target_properties(TensorRT::${_lib} PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${TensorRT_INCLUDE_DIR}")
        endif()
    endforeach()
    if(NOT TARGET TensorRT::TensorRT)
        add_library(TensorRT::TensorRT INTERFACE IMPORTED)
        set_target_properties(TensorRT::TensorRT PROPERTIES INTERFACE_LINK_LIBRARIES "TensorRT::nvinfer;TensorRT::nvonnxparser")
    endif()
endif()

mark_as_advanced(TensorRT_INCLUDE_DIR TensorRT_nvinfer_LIBRARY TensorRT_nvonnxparser_LIBRARY)
