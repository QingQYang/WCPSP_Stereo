if(CMAKE_GENERATOR MATCHES Xcode AND XCODE_VERSION VERSION_GREATER 4.3)
  cmake_minimum_required(VERSION 2.8.8)
elseif(IOS)
  cmake_minimum_required(VERSION 2.8.0)
else()
  cmake_minimum_required(VERSION 2.6.3)
endif()

project(wcpsp_stereo)
# The version number.
set(WCPSP_STEREO_VERSION_MAJOR 1)
set(WCPSP_STEREO_VERSION_MINOR 0)

# Add C++11 support
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse4.2 -std=c++11")
# choose debug and release vesion
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")
set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall -DNDEBUG")

set(src
  timer.h
  timer.cc
  image.h
  image_proc-inl.h
  ctmf.h
  ctmf.c
  stereo_lib.h
  stereo_lib.cc
  match_cost.h
  match_cost.cc
  horiz_tree_cost_propagation.h
  )
source_group(src FILES ${src})

find_package(PNG)

add_executable(wcpsp_stereo wcpsp_stereo.cc ${src})

target_include_directories(wcpsp_stereo PUBLIC ${module_path} ${PNG_INCLUDE_DIRS})
target_link_libraries(wcpsp_stereo ${PNG_LIBRARIES})
