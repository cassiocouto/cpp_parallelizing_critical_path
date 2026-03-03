cmake_minimum_required(VERSION 3.14)
project(openmp_critical_path LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)

set(CMAKE_CXX_FLAGS_RELEASE "-O2")
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

include_directories(${CMAKE_SOURCE_DIR}/include)

set(EXAMPLES
    benchmark
    parallel_for_ema
    race_condition_demo
    thread_local_storage
    scheduling_strategies
)

foreach(example ${EXAMPLES})
    add_executable(${example} src/${example}.cpp)
    target_link_libraries(${example} PRIVATE OpenMP::OpenMP_CXX)
endforeach()
