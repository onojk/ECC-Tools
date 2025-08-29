cmake_minimum_required(VERSION 3.16)
project(ecc_tools C)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# --- Find GMP ---
find_path(GMP_INCLUDE_DIR gmp.h
  PATHS /usr/include /usr/local/include
)
find_library(GMP_LIBRARY NAMES gmp
  PATHS /usr/lib /usr/lib/x86_64-linux-gnu /usr/local/lib
)
if (NOT GMP_INCLUDE_DIR OR NOT GMP_LIBRARY)
  message(FATAL_ERROR "GMP not found. Install libgmp-dev.")
endif()

# --- Find libecm (gmp-ecm) ---
find_path(ECM_INCLUDE_DIR ecm.h
  PATHS /usr/include /usr/local/include
)
find_library(ECM_LIBRARY NAMES ecm
  PATHS /usr/lib /usr/lib/x86_64-linux-gnu /usr/local/lib
)
if (NOT ECM_INCLUDE_DIR OR NOT ECM_LIBRARY)
  message(FATAL_ERROR "libecm not found. Install libecm1-dev/libecm-dev.")
endif()

add_executable(ecc-tools src/main.c)
target_include_directories(ecc-tools PRIVATE ${GMP_INCLUDE_DIR} ${ECM_INCLUDE_DIR})
target_link_libraries(ecc-tools PRIVATE ${ECM_LIBRARY} ${GMP_LIBRARY})
target_compile_options(ecc-tools PRIVATE -Wall -Wextra -Wpedantic -O3)
