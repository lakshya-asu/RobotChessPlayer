cmake_minimum_required(VERSION 3.8)
project(chess_manipulator_msgs)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(trajectory_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "action/ExecuteChessMove.action"
  "msg/ExecutionCommand.msg"
  "srv/GetBestMove.srv"
  DEPENDENCIES builtin_interfaces trajectory_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
