cmake_minimum_required(VERSION 3.20)
project(rt_renderer LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Vulkan REQUIRED)
find_package(glfw3 QUIET)
if(NOT glfw3_FOUND)
  include(FetchContent)
  set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
  set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(glfw GIT_REPOSITORY https://github.com/glfw/glfw.git GIT_TAG 3.4)
  FetchContent_MakeAvailable(glfw)
endif()

# NRD's Vulkan integration uses NRI. Pin NRI to the version required by NRD 4.18.
include(FetchContent)
set(NRD_STATIC_LIBRARY ON CACHE BOOL "" FORCE)
set(NRD_NRI ON CACHE BOOL "" FORCE)
set(NRD_SUPPORTS_QUAD_INTRINSICS OFF CACHE BOOL "" FORCE)
set(NRD_NORMAL_ENCODING "2" CACHE STRING "" FORCE)
set(NRD_ROUGHNESS_ENCODING "1" CACHE STRING "" FORCE)
set(NRD_EMBEDS_DXIL_SHADERS OFF CACHE BOOL "" FORCE)
set(NRD_EMBEDS_DXBC_SHADERS OFF CACHE BOOL "" FORCE)
set(SHADERMAKE_FIND_FXC OFF CACHE BOOL "" FORCE)
set(NRI_STATIC_LIBRARY ON CACHE BOOL "" FORCE)
set(NRI_ENABLE_D3D11_SUPPORT OFF CACHE BOOL "" FORCE)
set(NRI_ENABLE_D3D12_SUPPORT OFF CACHE BOOL "" FORCE)
set(NRI_ENABLE_NVAPI OFF CACHE BOOL "" FORCE)
set(NRI_ENABLE_AMDAGS OFF CACHE BOOL "" FORCE)
set(NRI_ENABLE_NVTX_SUPPORT OFF CACHE BOOL "" FORCE)
set(NRI_ENABLE_WGPU_SUPPORT OFF CACHE BOOL "" FORCE)
FetchContent_Declare(nri
  DOWNLOAD_EXTRACT_TIMESTAMP 1
  URL https://github.com/NVIDIA-RTX/NRI/archive/refs/tags/v192.zip)
FetchContent_Declare(nrd
  DOWNLOAD_EXTRACT_TIMESTAMP 1
  URL https://github.com/NVIDIA-RTX/NRD/archive/refs/tags/v4.18.0.zip)
FetchContent_MakeAvailable(nrd)

# Accept either the folder layout (src/, shaders/) or all files flat next to this file.
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp)
  set(MAIN_CPP ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp)
elseif(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
  set(MAIN_CPP ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
else()
  message(FATAL_ERROR "main.cpp not found next to CMakeLists.txt or in src/")
endif()
add_executable(rt_renderer ${MAIN_CPP})
target_link_libraries(rt_renderer PRIVATE Vulkan::Vulkan glfw NRD NRI NRDIntegration)
# world.hpp / stb_image.h live in src/ (or next to this file in a flat layout)
target_include_directories(rt_renderer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR})

# ---- optional NVIDIA Streamline DLSS Super Resolution
set(STREAMLINE_SDK_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/streamline-sdk-v2.14.16" CACHE PATH "NVIDIA Streamline SDK root")
set(RTENGINE_DLSS_DEFAULT OFF)
if(WIN32 AND EXISTS "${STREAMLINE_SDK_ROOT}/include/sl.h")
  set(RTENGINE_DLSS_DEFAULT ON)
endif()
option(RTENGINE_ENABLE_DLSS "Enable DLSS Super Resolution through Streamline" ${RTENGINE_DLSS_DEFAULT})
if(RTENGINE_ENABLE_DLSS)
  if(NOT WIN32)
    message(FATAL_ERROR "Streamline DLSS integration is currently supported only on Windows.")
  endif()
  foreach(REQUIRED_FILE
      "${STREAMLINE_SDK_ROOT}/include/sl.h"
      "${STREAMLINE_SDK_ROOT}/include/sl_dlss.h"
      "${STREAMLINE_SDK_ROOT}/include/sl_helpers_vk.h"
      "${STREAMLINE_SDK_ROOT}/lib/x64/sl.interposer.lib"
      "${STREAMLINE_SDK_ROOT}/bin/x64/sl.interposer.dll"
      "${STREAMLINE_SDK_ROOT}/bin/x64/sl.common.dll"
      "${STREAMLINE_SDK_ROOT}/bin/x64/sl.dlss.dll"
      "${STREAMLINE_SDK_ROOT}/bin/x64/nvngx_dlss.dll")
    if(NOT EXISTS "${REQUIRED_FILE}")
      message(FATAL_ERROR "RTENGINE_ENABLE_DLSS requires Streamline SDK file: ${REQUIRED_FILE}")
    endif()
  endforeach()
  target_include_directories(rt_renderer PRIVATE "${STREAMLINE_SDK_ROOT}/include")
  target_link_libraries(rt_renderer PRIVATE "${STREAMLINE_SDK_ROOT}/lib/x64/sl.interposer.lib")
  target_compile_definitions(rt_renderer PRIVATE RTENGINE_HAS_DLSS=1)
  foreach(F sl.interposer.dll sl.common.dll sl.dlss.dll nvngx_dlss.dll)
    add_custom_command(TARGET rt_renderer POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${STREAMLINE_SDK_ROOT}/bin/x64/${F}" "$<TARGET_FILE_DIR:rt_renderer>/${F}")
  endforeach()
  if(EXISTS "${STREAMLINE_SDK_ROOT}/bin/x64/nvngx_dlss.license.txt")
    add_custom_command(TARGET rt_renderer POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${STREAMLINE_SDK_ROOT}/bin/x64/nvngx_dlss.license.txt" "$<TARGET_FILE_DIR:rt_renderer>/nvngx_dlss.license.txt")
  endif()
  file(GLOB STREAMLINE_JSONS "${STREAMLINE_SDK_ROOT}/scripts/*.json")
  foreach(F ${STREAMLINE_JSONS})
    add_custom_command(TARGET rt_renderer POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different "${F}" "$<TARGET_FILE_DIR:rt_renderer>")
  endforeach()
endif()

# ---- shaders -> SPIR-V (Vulkan 1.2 target; ray query needs SPIR-V >= 1.4)
find_program(GLSLANG_VALIDATOR glslangValidator HINTS ENV VULKAN_SDK PATH_SUFFIXES bin Bin)
find_program(GLSLC glslc HINTS ENV VULKAN_SDK PATH_SUFFIXES bin Bin)
if(NOT GLSLANG_VALIDATOR AND NOT GLSLC)
  message(FATAL_ERROR "Need glslangValidator or glslc (install the Vulkan SDK).")
endif()

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/shaders/common.glsl)
  set(SHADER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)
elseif(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/common.glsl)
  set(SHADER_DIR ${CMAKE_CURRENT_SOURCE_DIR})
else()
  message(FATAL_ERROR "common.glsl not found next to CMakeLists.txt or in shaders/")
endif()
set(SHADER_OUT ${CMAKE_BINARY_DIR}/shaders)
file(MAKE_DIRECTORY ${SHADER_OUT})
set(SPV_FILES)
foreach(S gbuffer light_grid cache_update tile_lights di_initial di_temporal di_spatial direct gi nrd_prep composite)
  set(IN  ${SHADER_DIR}/${S}.comp)
  set(OUT ${SHADER_OUT}/${S}.comp.spv)
  if(GLSLANG_VALIDATOR)
    set(CMD ${GLSLANG_VALIDATOR} -V --target-env vulkan1.2 -I${SHADER_DIR} ${IN} -o ${OUT})
  else()
    set(CMD ${GLSLC} --target-env=vulkan1.2 -O -I ${SHADER_DIR} ${IN} -o ${OUT})
  endif()
  add_custom_command(OUTPUT ${OUT} COMMAND ${CMD} DEPENDS ${IN} ${SHADER_DIR}/common.glsl VERBATIM)
  list(APPEND SPV_FILES ${OUT})
endforeach()
add_custom_target(shaders DEPENDS ${SPV_FILES})
add_dependencies(rt_renderer shaders)

# Make the exe find ./shaders next to it (also works from build/ or build/Release/ via the fallback in main.cpp)
add_custom_command(TARGET rt_renderer POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${SHADER_OUT} $<TARGET_FILE_DIR:rt_renderer>/shaders)
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/shaders)
  add_custom_command(TARGET rt_renderer POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/shaders $<TARGET_FILE_DIR:rt_renderer>/shaders)
endif()
add_custom_command(TARGET rt_renderer POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SHADER_DIR}/common.glsl $<TARGET_FILE_DIR:rt_renderer>/shaders/common.glsl)

# Ship the example script and assets next to the exe (the app also searches the current directory)
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/scene.scr)
  add_custom_command(TARGET rt_renderer POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/scene.scr $<TARGET_FILE_DIR:rt_renderer>/scene.scr)
endif()
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/assets)
  add_custom_command(TARGET rt_renderer POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/assets $<TARGET_FILE_DIR:rt_renderer>/assets)
endif()
