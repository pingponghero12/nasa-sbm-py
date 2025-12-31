cmake_minimum_required(VERSION 3.15)
project(nasa_sbm_bindings)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Python COMPONENTS Interpreter Development REQUIRED)

find_package(pybind11 CONFIG REQUIRED)

set(NASA_SBM_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/NASA-breakup-model-cpp")

include(FetchContent)

set(SPDLOG_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(SPDLOG_BUILD_EXAMPLE OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_CONTRIB OFF CACHE BOOL "" FORCE)
set(YAML_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)


# Fetch spdlog
FetchContent_Declare(
    spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG v1.8.5
)

# Fetch yaml-cpp
FetchContent_Declare(
    yaml-cpp
    GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
    GIT_TAG yaml-cpp-0.6.3
)

# Make dependencies available
FetchContent_MakeAvailable(spdlog yaml-cpp)

find_package(TBB)

include_directories(
    ${NASA_SBM_SOURCE_DIR}/src
    ${spdlog_SOURCE_DIR}/include
    ${yaml-cpp_SOURCE_DIR}/include
)

file(GLOB_RECURSE NASA_SBM_SOURCES
    "${NASA_SBM_SOURCE_DIR}/src/breakupModel/*.cpp"
)

list(FILTER NASA_SBM_SOURCES EXCLUDE REGEX ".*main\\.cpp$")

pybind11_add_module(_core MODULE
    src/bindings.cpp
    ${NASA_SBM_SOURCES}
)

target_link_libraries(_core PRIVATE
    pybind11::module
    spdlog::spdlog
    yaml-cpp
)

target_compile_definitions(_core PRIVATE
    SPDLOG_COMPILED_LIB
)

set_target_properties(_core PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/nasa_sbm"
)

install(TARGETS _core DESTINATION nasa_sbm)
