cmake_minimum_required(VERSION 3.16)
project(raytracing)

set(CMAKE_CXX_STANDARD 17)

add_executable(raytracing main.cpp)

include(FetchContent)

FetchContent_Declare(
        glm
        GIT_REPOSITORY	https://github.com/g-truc/glm.git
        GIT_TAG 	bf71a834948186f4097caa076cd2663c69a10e1e #refs/tags/0.9.9.8
)

FetchContent_MakeAvailable(glm)

target_link_libraries(raytracing PRIVATE glm::glm)

if(APPLE)
    find_package(OpenMP)
    if(NOT OpenMP_FOUND)
        # Try again with extra path info; required for libomp 15+ from Homebrew
        execute_process(COMMAND brew --prefix libomp
                OUTPUT_VARIABLE HOMEBREW_LIBOMP_PREFIX
                OUTPUT_STRIP_TRAILING_WHITESPACE)
        set(OpenMP_C_FLAGS
                "-Xpreprocessor -fopenmp -I${HOMEBREW_LIBOMP_PREFIX}/include")
        set(OpenMP_CXX_FLAGS
                "-Xpreprocessor -fopenmp -I${HOMEBREW_LIBOMP_PREFIX}/include")
        set(OpenMP_C_LIB_NAMES omp)
        set(OpenMP_CXX_LIB_NAMES omp)
        set(OpenMP_omp_LIBRARY ${HOMEBREW_LIBOMP_PREFIX}/lib/libomp.dylib)
        find_package(OpenMP REQUIRED)
    endif()
else()
    find_package(OpenMP REQUIRED)
endif()

target_link_libraries(raytracing PUBLIC OpenMP::OpenMP_CXX)
