cmake_minimum_required(VERSION 3.5)
project(o30_right_description)

find_package(ament_cmake REQUIRED)

install(DIRECTORY meshes DESTINATION share/${PROJECT_NAME})
install(FILES linkerhand_o30_right.urdf DESTINATION share/${PROJECT_NAME})

ament_package()
