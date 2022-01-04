#CC Build instructions
#module load python/3.7 and  module load scipy-stack and module load mariadb
#module load mariadb-connector-c

cmake_minimum_required(VERSION 3.14)
project(FlexibleNN LANGUAGES CUDA CXX)
#enable_language(CUDA)
enable_testing()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

#GCloud
#set(CMAKE_CXX_FLAGS "-O3 -lpthread -ldl -lm -I/home/khurramjaved/mariadb/mariadb-connector-c-3.1.13-ubuntu-focal-amd64/include/mariadb -I/home/khurramjaved/mariadb/mariadb-connector-c-3.1.13-ubuntu-focal-amd64/include/mariadb/mysql -L/home/khurramjaved/mariadb/mariadb-connector-c-3.1.13-ubuntu-focal-amd64/lib/mariadb/")

#Cedar
#message( "Found environment variable SLURM_TMPDIR=$ENV{SLURM_TMPDIR}" )
#set(CMAKE_CXX_FLAGS "-O3 -ltbb -lpthread")
#set(pybind11_DIR $ENV{SLURM_TMPDIR}/CAN/lib/python3.7/site-packages/pybind11/share/cmake/pybind11)
#find_package(Python COMPONENTS Interpreter Development REQUIRED)
#find_package(pybind11 REQUIRED)

#include_directories($ENV{SLURM_TMPDIR}/CAN/lib/python3.7/site-packages/pybind11/include/pybind11)


add_executable(RecurrentNetwork include/experiment/Experiment.h src/experiment/Experiment.cpp include/experiment/Database.h
        src/experiment/Database.cpp include/experiment/Metric.h src/experiment/Metric.cpp recurrent_experiment.cpp include/nn/synapse.h
        include/nn/synapse.h include/nn/message.h include/nn/dynamic_elem.h include/nn/utils.h include/nn/networks/neural_network.h
        include/nn/networks/recurrent_network.h src/nn/synapse.cpp
        src/nn/neuron.cpp src/nn/utils.cpp src/nn/dynamic_elem.cpp src/nn/networks/recurrent_network.cpp
        src/nn/networks/neural_network.cpp src/nn/utils.cpp src/utils.cpp include/utils.h src/nn/message.cpp
        include/environments/animal_learning/tracecondioning.h src/environments/animal_learning/traceconditioning.cpp
        src/environments/animal_learning/tracepatterning.cpp)



target_link_libraries(RecurrentNetwork PUBLIC tbb)
target_link_libraries(RecurrentNetwork PUBLIC mariadb)

# pybind specific config begin (comment this out if not working)
add_subdirectory(pybind11)
include_directories(pybind11/include/pybind11)

pybind11_add_module(FlexibleNN include/experiment/Experiment.h src/experiment/Experiment.cpp include/experiment/Database.h
        src/experiment/Database.cpp include/experiment/Metric.h src/experiment/Metric.cpp recurrent_experiment.cpp include/nn/synapse.h
        include/nn/synapse.h include/nn/message.h include/nn/dynamic_elem.h include/nn/utils.h include/nn/networks/neural_network.h
        include/nn/networks/recurrent_network.h src/nn/synapse.cpp
        src/nn/neuron.cpp src/nn/utils.cpp src/nn/dynamic_elem.cpp src/nn/networks/recurrent_network.cpp
        src/nn/networks/neural_network.cpp src/nn/utils.cpp src/utils.cpp include/utils.h src/nn/message.cpp
        include/environments/animal_learning/tracecondioning.h src/environments/animal_learning/traceconditioning.cpp
        src/environments/animal_learning/tracepatterning.cpp pybind_interface/interface.cpp)

target_link_libraries(FlexibleNN PUBLIC tbb)
target_link_libraries(FlexibleNN PUBLIC mariadb)
# pybind specific config  end

project(FlexibleNN LANGUAGES CXX)

