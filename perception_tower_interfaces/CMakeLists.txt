cmake_minimum_required(VERSION 3.8)
project(perception_tower_interfaces)

if(CMAKE_COMPILER_IS_GNUCXX OR CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/TowerStatus.msg"
  "msg/TurntableStatus.msg"
  "srv/TowerCommand.srv"
  "srv/TurntableCommand.srv"
  DEPENDENCIES std_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
