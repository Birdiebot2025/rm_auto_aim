cmake_minimum_required(VERSION 3.8)
project(armor_detector_nn)
set(CMAKE_BUILD_TYPE Debug)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

## Export compile commands for clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

## enforcing cleaner code.
add_definitions(-Wall -Werror)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-pedantic -Wno-unused-parameter -Wno-error=pedantic -Wno-error=unused-parameter")


# find dependencies
# 设置 OpenVINO 路径
set(INC_DIR /opt/intel/openvino_2024.2.0/runtime/include)
set(LINK_DIR /opt/intel/openvino_2024.2.0/runtime/lib/intel64)
# 包含头文件和库路径
include_directories(${INC_DIR})
link_directories(${LINK_DIR})

find_package(ament_cmake_auto REQUIRED)
find_package(OpenCV REQUIRED)
ament_auto_find_build_dependencies()


ament_auto_add_library(${PROJECT_NAME} SHARED
  DIRECTORY src
)

target_include_directories(${PROJECT_NAME} PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} libopenvino.so)

rclcpp_components_register_node(${PROJECT_NAME}
  PLUGIN rm_auto_aim::ArmorDetectorNode
  EXECUTABLE armor_detector_nn_node
)

# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_auto_package(
  INSTALL_TO_SHARE
  model
  cache
  config
)
message("哈基米拉贝多亚锡达哈压库")

