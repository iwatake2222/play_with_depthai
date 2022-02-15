cmake_minimum_required(VERSION 3.14)

# Create project
set(ProjectName "main")
project(${ProjectName})
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${ProjectName})

# Select build system and set compile options
include(${CMAKE_CURRENT_LIST_DIR}/../common_helper/cmakes/build_setting.cmake)

# Create executable file
add_executable(${ProjectName} main.cpp)

# Link ImageProcessor module
# add_subdirectory(./image_processor image_processor)
# target_include_directories(${ProjectName} PUBLIC ./image_processor)
# target_link_libraries(${ProjectName} ImageProcessor)

# Link OpenCV
set(OpenCV_DIR "${CMAKE_CURRENT_LIST_DIR}/../third_party/opencv/build/")
find_package(OpenCV REQUIRED)
target_include_directories(${ProjectName} PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${ProjectName} ${OpenCV_LIBS})
# set_target_properties(${ProjectName} PROPERTIES VS_DEBUGGER_ENVIRONMENT "PATH=%PATH%;${OpenCV_DIR}/x64/vc15/bin/")

# Link DepthAI
set(depthai_DIR "${CMAKE_CURRENT_LIST_DIR}/../third_party/depthai-core/lib/cmake/depthai/")
find_package(depthai REQUIRED)
target_include_directories(${ProjectName} PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${ProjectName} depthai::core depthai::opencv)
# set_target_properties(${ProjectName} PROPERTIES VS_DEBUGGER_ENVIRONMENT "PATH=%PATH%;${depthai_DIR}/../../../bin/")
set_target_properties(${ProjectName} PROPERTIES VS_DEBUGGER_ENVIRONMENT "PATH=%PATH%;${OpenCV_DIR}/x64/vc15/bin/;${depthai_DIR}/../../../bin/")

# Copy resouce
file(COPY ${CMAKE_CURRENT_LIST_DIR}/../resource DESTINATION ${CMAKE_BINARY_DIR}/)
add_definitions(-DRESOURCE_DIR="${CMAKE_BINARY_DIR}/resource/")
