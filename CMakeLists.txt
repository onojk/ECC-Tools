cmake_minimum_required(VERSION 3.16)
project(ecc_tools C)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

find_package(PkgConfig REQUIRED)
pkg_check_modules(GMP REQUIRED gmp)
pkg_check_modules(ECM REQUIRED ecm)

add_executable(ecc-tools src/main.c)
target_include_directories(ecc-tools PRIVATE ${GMP_INCLUDE_DIRS} ${ECM_INCLUDE_DIRS})
target_link_libraries(ecc-tools PRIVATE ${GMP_LIBRARIES} ${ECM_LIBRARIES})
target_compile_options(ecc-tools PRIVATE -Wall -Wextra -Wpedantic -O3)
