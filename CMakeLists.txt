#CC Build instructions
#module load python/3.7 and  module load scipy-stack and module load mariadb
#module load mariadb-connector-c

cmake_minimum_required(VERSION 3.14)
project(FlexibleNN LANGUAGES CXX)
#enable_language(CUDA)
#enable_testing()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(SOURCE_FILES graph_expeirments.cpp include/nn/networks/graph.h src/nn/networks/graph.cpp include/experiment/Metric.h include/experiment/Database.h include/experiment/Experiment.h
        src/experiment/Database.cpp src/experiment/Experiment.cpp src/experiment/Metric.cpp)

add_executable(Graph graph_expeirments.cpp include/nn/networks/graph.h include/utils.h src/utils.cpp src/nn/networks/graph.cpp include/experiment/Metric.h include/experiment/Database.h include/experiment/Experiment.h
        src/experiment/Database.cpp src/experiment/Experiment.cpp src/experiment/Metric.cpp src/nn/networks/graph.cpp src/nn/networks/graph_utility_prop_implementations.cpp src/nn/networks/edge.cpp src/nn/networks/vertex.cpp
        include/nn/networks/vertex.h include/nn/graphfactory.h src/nn/graphfactory.cpp include/nn/weight_initializer.h src/nn/weight_initializer.cpp)


add_executable(Count include/json.hpp include/utils.h src/utils.cpp include/experiment/Experiment.h include/experiment/Metric.h include/experiment/Database.h src/experiment/Metric.cpp src/experiment/Database.cpp src/experiment/Experiment.cpp
        count_configs.cpp)


target_link_libraries(Graph PUBLIC tbb)
target_link_libraries(Graph PUBLIC mariadb)
#

target_link_libraries(Count PUBLIC tbb)
target_link_libraries(Count PUBLIC mariadb)
#

project(FlexibleNN LANGUAGES CXX)

