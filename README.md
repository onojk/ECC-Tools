# ECC-Tools — ECM factorization CLI (C)

Integer factoring via **Lenstra ECM** using **libecm** + **GMP**.

## Build
sudo apt-get update
sudo apt-get install -y build-essential cmake pkg-config libgmp-dev libecm-dev
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/ecc-tools --help
