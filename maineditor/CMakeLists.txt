cmake_minimum_required(VERSION 3.16)
project(scr_editor LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_POLICY_VERSION_MINIMUM "3.5" CACHE STRING "" )

include(FetchContent)

set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
FetchContent_Declare(glfw
  GIT_REPOSITORY https://github.com/glfw/glfw.git
  GIT_TAG        3.3.9
)
FetchContent_MakeAvailable(glfw)

FetchContent_Declare(glew
  GIT_REPOSITORY https://github.com/Perlmint/glew-cmake.git
  GIT_TAG        glew-cmake-2.2.0
)
FetchContent_MakeAvailable(glew)

FetchContent_Declare(imgui
  GIT_REPOSITORY https://github.com/ocornut/imgui.git
  GIT_TAG        v1.90.1
)
FetchContent_MakeAvailable(imgui)

FetchContent_Declare(stb
  GIT_REPOSITORY https://github.com/nothings/stb.git
  GIT_TAG        master
)
FetchContent_GetProperties(stb)
if(NOT stb_POPULATED)
  FetchContent_Populate(stb)
endif()

set(ASSIMP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(ASSIMP_BUILD_ASSIMP_TOOLS OFF CACHE BOOL "" FORCE)
set(ASSIMP_BUILD_SAMPLES OFF CACHE BOOL "" FORCE)
set(ASSIMP_BUILD_ALL_IMPORTERS_BY_DEFAULT OFF CACHE BOOL "" FORCE)
set(ASSIMP_BUILD_ALL_EXPORTERS_BY_DEFAULT OFF CACHE BOOL "" FORCE)
set(ASSIMP_BUILD_FBX_IMPORTER ON CACHE BOOL "" FORCE)
FetchContent_Declare(assimp
  GIT_REPOSITORY https://github.com/assimp/assimp.git
  GIT_TAG        v5.4.3
)
FetchContent_MakeAvailable(assimp)

add_library(imgui_lib STATIC
  ${imgui_SOURCE_DIR}/imgui.cpp
  ${imgui_SOURCE_DIR}/imgui_draw.cpp
  ${imgui_SOURCE_DIR}/imgui_tables.cpp
  ${imgui_SOURCE_DIR}/imgui_widgets.cpp
  ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
  ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
)
target_include_directories(imgui_lib PUBLIC
  ${imgui_SOURCE_DIR}
  ${imgui_SOURCE_DIR}/backends
  ${stb_SOURCE_DIR}
)
target_link_libraries(imgui_lib PUBLIC glfw)

if (TARGET libglew_static)
  set(GLEW_TARGET libglew_static)
elseif (TARGET libglew_shared)
  set(GLEW_TARGET libglew_shared)
else()
  set(GLEW_TARGET libglew)
endif()
target_compile_definitions(imgui_lib PUBLIC GLEW_STATIC)
target_link_libraries(imgui_lib PUBLIC ${GLEW_TARGET})

add_executable(scr_editor
  src/main.cpp
  src/renderer.cpp
  src/scene.cpp
  src/objloader.cpp
  src/fbxloader.cpp
  src/texture.cpp
)
target_link_libraries(scr_editor PRIVATE imgui_lib glfw ${GLEW_TARGET} assimp::assimp)
if (WIN32)
  target_link_libraries(scr_editor PRIVATE dwmapi)
endif()
target_compile_definitions(scr_editor PRIVATE GLEW_STATIC)