cmake_minimum_required(VERSION 3.10)
project(armor_detector)

## Use C++14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

## By adding -Wall and -Werror, the compiler does not ignore warnings anymore,
## enforcing cleaner code.
add_definitions(-Wall -Werror)

## Export compile commands for clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

#######################
## Find dependencies ##
#######################

find_package(ament_cmake_auto REQUIRED)
find_package(OpenCV REQUIRED)
ament_auto_find_build_dependencies()

include_directories(/opt/intel/openvino_2023.3.0/runtime/include
/opt/intel/openvino_2023.3.0/samples/cpp/common/utils/include
/opt/intel/openvino_2023.3.0/samples/cpp/common/format_reader/include
/opt/intel/openvino_2023.3.0/samples/cpp/common/format_reader/src
/opt/intel/openvino_2023.3.0/runtime/include/ie
/opt/intel/openvino_2023.3.0/runtime/include/openvino/c
/usr/include/opencv4
)
link_directories(/opt/intel/openvino_2023.3.0/runtime/lib
/usr/lib/x86_64-linux-gnu
/opt/intel/openvino_2023.3.0/runtime/lib/intel64
/opt/intel/openvino_2023.3.0/runtime/lib/
)

###########
## Build ##
###########

ament_auto_add_library(${PROJECT_NAME} SHARED
  DIRECTORY src
)

target_include_directories(${PROJECT_NAME} PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(armor_detector
/opt/intel/openvino_2023.3.0/runtime/lib/intel64/libopenvino.so
/opt/intel/openvino_2023.3.0/runtime/lib/intel64/libopenvino_c.so
/opt/intel/openvino_2023.3.0/runtime/lib/intel64/libgna.so.3
/opt/intel/openvino_2023.3.0/runtime/lib/intel64/libopenvino.so.2330
/opt/intel/openvino_2023.3.0/runtime/lib/intel64/libgna.so.3.5.0.2116
${OpenCV_LIBS}
)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

rclcpp_components_register_node(${PROJECT_NAME}
  PLUGIN rm_auto_aim::ArmorDetectorNode
  EXECUTABLE armor_detector_node
)

#############
## Testing ##
#############

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  list(APPEND AMENT_LINT_AUTO_EXCLUDE
    ament_cmake_copyright
    ament_cmake_uncrustify
    ament_cmake_cpplint
  )
  ament_lint_auto_find_test_dependencies()

  find_package(ament_cmake_gtest)
  # ament_add_gtest(test_node_startup test/test_node_startup.cpp)
  # target_link_libraries(test_node_startup ${PROJECT_NAME})

  # ament_add_gtest(test_number_cls test/test_number_cls.cpp)
  # target_link_libraries(test_number_cls ${PROJECT_NAME})

endif()

#############
## Install ##
#############

ament_auto_package(
  INSTALL_TO_SHARE
  model
)


