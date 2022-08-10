#CC Build instructions
#module load python/3.7 and  module load scipy-stack and module load mariadb
#module load mariadb-connector-c

cmake_minimum_required(VERSION 3.14)
project(FlexibleNN LANGUAGES CXX)
#enable_language(CUDA)
#enable_testing()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

#GCloud
#set(CMAKE_CXX_FLAGS "-O3 -lpthread -ldl -lm -I/home/khurramjaved/mariadb/mariadb-connector-c-3.1.13-ubuntu-focal-amd64/include/mariadb -I/home/khurramjaved/mariadb/mariadb-connector-c-3.1.13-ubuntu-focal-amd64/include/mariadb/mysql -L/home/khurramjaved/mariadb/mariadb-connector-c-3.1.13-ubuntu-focal-amd64/lib/mariadb/")
#set(CMAKE_C_FLAGS "-I/usr/local/include")
#set(CMAKE_CXX_FLAGS "-L/usr/local/lib -lprotobuf")
#include_directories("/usr/local/include")
#Cedar
#message( "Found environment variable SLURM_TMPDIR=$ENV{SLURM_TMPDIR}" )
#set(CMAKE_CXX_FLAGS "-O3 -ltbb -lpthread")
#set(pybind11_DIR $ENV{SLURM_TMPDIR}/CAN/lib/python3.7/site-packages/pybind11/share/cmake/pybind11)
#find_package(Python COMPONENTS Interpreter Development REQUIRED)
#find_package(pybind11 REQUIRED)



add_executable(AtariExperiment atari_experiment.cpp proto_files/experience.pb.h proto_files/experience.pb.cc include/environments/proto_prediction_environments.h src/environments/proto_prediction_environments.cpp include/utils.h src/utils.cpp
        include/nn/networks/dense_lstm.h src/nn/networks/dense_lstm.cpp include/json.hpp include/experiment/Experiment.h src/experiment/Experiment.cpp include/experiment/Database.h
                src/experiment/Database.cpp include/experiment/Metric.h src/experiment/Metric.cpp include/nn/synapse.h
                include/nn/synapse.h include/nn/message.h include/nn/dynamic_elem.h include/nn/utils.h include/nn/networks/neural_network.h
                include/nn/networks/recurrent_network.h src/nn/synapse.cpp
                src/nn/neuron.cpp src/nn/utils.cpp src/nn/dynamic_elem.cpp src/nn/networks/recurrent_network.cpp
                src/nn/networks/neural_network.cpp src/nn/utils.cpp src/utils.cpp include/utils.h src/nn/message.cpp
                include/environments/animal_learning/tracecondioning.h src/environments/animal_learning/traceconditioning.cpp
                src/environments/animal_learning/tracepatterning.cpp include/nn/networks/td_lambda.h src/nn/networks/TDLambda.cpp src/nn/lstm.cpp include/nn/networks/snap1.h src/nn/networks/Snap1.cpp include/nn/networks/mlp_lstm.h src/nn/networks/mlp_lstm.cpp src/nn/lstm.cpp include/nn/networks/network_factory.h src/nn/networks/network_factory.cpp)

#add_executable(AnimalLearning include/nn/networks/dense_lstm.h src/nn/networks/dense_lstm.cpp include/json.hpp include/experiment/Experiment.h src/experiment/Experiment.cpp include/experiment/Database.h
#        src/experiment/Database.cpp include/experiment/Metric.h src/experiment/Metric.cpp include/nn/synapse.h
#        include/nn/synapse.h include/nn/message.h include/nn/dynamic_elem.h include/nn/utils.h include/nn/networks/neural_network.h
#        include/nn/networks/recurrent_network.h src/nn/synapse.cpp
#        src/nn/neuron.cpp src/nn/utils.cpp src/nn/dynamic_elem.cpp src/nn/networks/recurrent_network.cpp
#        src/nn/networks/neural_network.cpp src/nn/utils.cpp src/utils.cpp include/utils.h src/nn/message.cpp
#        include/environments/animal_learning/tracecondioning.h src/environments/animal_learning/traceconditioning.cpp
#        src/environments/animal_learning/tracepatterning.cpp include/nn/networks/td_lambda.h src/nn/networks/TDLambda.cpp src/nn/lstm.cpp include/nn/networks/snap1.h src/nn/networks/Snap1.cpp include/nn/networks/mlp_lstm.h src/nn/networks/mlp_lstm.cpp src/nn/lstm.cpp experiment_animal_learning.cpp)

#add_executable(AnimalLearningSnap include/experiment/Experiment.h src/experiment/Experiment.cpp include/experiment/Database.h
#        src/experiment/Database.cpp include/experiment/Metric.h src/experiment/Metric.cpp include/nn/synapse.h
#        include/nn/synapse.h include/nn/message.h include/nn/dynamic_elem.h include/nn/utils.h include/nn/networks/neural_network.h
#        include/nn/networks/recurrent_network.h src/nn/synapse.cpp
#        src/nn/neuron.cpp src/nn/utils.cpp src/nn/dynamic_elem.cpp src/nn/networks/recurrent_network.cpp
#        src/nn/networks/neural_network.cpp src/nn/utils.cpp src/utils.cpp include/utils.h src/nn/message.cpp
#        include/environments/animal_learning/tracecondioning.h src/environments/animal_learning/traceconditioning.cpp
#        src/environments/animal_learning/tracepatterning.cpp include/nn/networks/td_lambda.h src/nn/networks/TDLambda.cpp src/nn/lstm.cpp include/nn/networks/snap1.h src/nn/networks/Snap1.cpp include/nn/networks/mlp_lstm.h src/nn/networks/mlp_lstm.cpp src/nn/lstm.cpp experiment_animal_learning_snap.cpp)

#add_executable(bptt include/json.hpp include/experiment/Experiment.h src/experiment/Experiment.cpp include/experiment/Database.h
#                src/experiment/Database.cpp include/experiment/Metric.h src/experiment/Metric.cpp
#        include/nn/synapse.h
#        include/nn/synapse.h include/nn/message.h include/nn/dynamic_elem.h include/nn/utils.h include/nn/networks/neural_network.h
#        include/nn/networks/recurrent_network.h src/nn/synapse.cpp
#        src/nn/neuron.cpp src/nn/utils.cpp src/nn/dynamic_elem.cpp
#        src/nn/utils.cpp src/utils.cpp include/utils.h src/nn/message.cpp
#        include/environments/animal_learning/tracecondioning.h src/environments/animal_learning/traceconditioning.cpp
#        src/environments/animal_learning/tracepatterning.cpp  include/nn/networks/dense_lstm.h src/nn/networks/dense_lstm.cpp experiment_bptt.cpp)

#add_executable(mnist include/json.hpp include/experiment/Experiment.h src/experiment/Experiment.cpp include/experiment/Database.h
#        src/experiment/Database.cpp include/experiment/Metric.h src/experiment/Metric.cpp
#        include/nn/synapse.h
#        include/nn/synapse.h include/nn/message.h include/nn/dynamic_elem.h include/nn/utils.h include/nn/networks/neural_network.h
#        include/nn/networks/recurrent_network.h src/nn/synapse.cpp
#        src/nn/neuron.cpp src/nn/utils.cpp src/nn/dynamic_elem.cpp
#        src/nn/utils.cpp src/utils.cpp include/utils.h src/nn/message.cpp
#        include/environments/animal_learning/tracecondioning.h src/environments/animal_learning/traceconditioning.cpp
#        src/environments/animal_learning/tracepatterning.cpp  include/nn/networks/dense_lstm.h src/nn/networks/dense_lstm.cpp mnist_bptt.cpp include/environments/mnist/mnist_reader.hpp include/environments/mnist/mnist_utils.hpp)
#
#add_executable(mnistcol include/json.hpp include/experiment/Experiment.h src/experiment/Experiment.cpp include/experiment/Database.h
#        src/experiment/Database.cpp include/experiment/Metric.h src/experiment/Metric.cpp
#        include/nn/synapse.h
#        include/nn/synapse.h include/nn/message.h include/nn/dynamic_elem.h include/nn/utils.h include/nn/networks/neural_network.h
#        include/nn/networks/recurrent_network.h src/nn/synapse.cpp
#        src/nn/neuron.cpp src/nn/utils.cpp src/nn/dynamic_elem.cpp
#        src/nn/utils.cpp src/utils.cpp include/utils.h src/nn/message.cpp
#        include/environments/animal_learning/tracecondioning.h src/environments/animal_learning/traceconditioning.cpp
#        src/environments/animal_learning/tracepatterning.cpp  include/nn/networks/td_lambda.h src/nn/networks/TDLambda.cpp include/nn/networks/dense_lstm.h src/nn/networks/dense_lstm.cpp experiment_mnist_binary.cpp src/nn/lstm.cpp include/environments/mnist/mnist_reader.hpp include/environments/mnist/mnist_utils.hpp)


add_executable(Count include/json.hpp include/utils.h src/utils.cpp include/experiment/Experiment.h include/experiment/Metric.h include/experiment/Database.h src/experiment/Metric.cpp src/experiment/Database.cpp src/experiment/Experiment.cpp
        count_configs.cpp)
#

#target_link_libraries(bptt PUBLIC tbb)
#target_link_libraries(bptt PUBLIC mariadb)

#target_link_libraries(mnist PUBLIC mariadb)
#
#target_link_libraries(mnistcol PUBLIC mariadb)

#target_link_libraries(LSTMClassification PUBLIC tbb)
#target_link_libraries(LSTMClassification PUBLIC mariadb)
#target_link_libraries(FlexibleNN PUBLIC mariadb)
#target_link_libraries(FlexibleNN PUBLIC tbb)
#target_link_libraries(FlexibleNN PUBLIC "${TORCH_LIBRARIES}")
#
#set_property(TARGET FlexibleNN PROPERTY CXX_STANDARD 14)

target_link_libraries(AtariExperiment PUBLIC protobuf)
target_link_libraries(AtariExperiment PUBLIC tbb)
target_link_libraries(AtariExperiment PUBLIC mariadb)
#

#target_link_libraries(Snap1 PUBLIC tbb)


#target_link_libraries(MnistClassification PUBLIC tbb)
#target_link_libraries(MnistClassification PUBLIC mariadb)

#target_link_libraries(LSTMMLP PUBLIC tbb)
#target_link_libraries(LSTMMLP PUBLIC mariadb)

#target_link_libraries(AnimalLearning PUBLIC tbb)
#target_link_libraries(AnimalLearning PUBLIC mariadb)
#
target_link_libraries(Count PUBLIC tbb)
target_link_libraries(Count PUBLIC mariadb)

#target_link_libraries(bptt PUBLIC tbb)
#target_link_libraries(bptt PUBLIC mariadb)

#target_link_libraries(AnimalLearningSnap PUBLIC tbb)
#target_link_libraries(AnimalLearningSnap PUBLIC mariadb)

#target_link_libraries(MnistClassificationMLP PUBLIC tbb)
#target_link_libraries(MnistClassificationMLP PUBLIC mariadb)

# pybind specific config begin (comment this out if not working)
#add_subdirectory(pybind11)
#include_directories(pybind11/include/pybind11)
#
#pybind11_add_module(FlexibleNN include/experiment/Experiment.h src/experiment/Experiment.cpp include/experiment/Database.h
#        src/experiment/Database.cpp include/experiment/Metric.h src/experiment/Metric.cpp recurrent_experiment.cpp include/nn/synapse.h
#        include/nn/synapse.h include/nn/message.h include/nn/dynamic_elem.h include/nn/utils.h include/nn/networks/neural_network.h
#        include/nn/networks/recurrent_network.h src/nn/synapse.cpp
#        src/nn/neuron.cpp src/nn/utils.cpp src/nn/dynamic_elem.cpp src/nn/networks/recurrent_network.cpp
#        src/nn/networks/neural_network.cpp src/nn/utils.cpp src/utils.cpp include/utils.h src/nn/message.cpp
#        include/environments/animal_learning/tracecondioning.h src/environments/animal_learning/traceconditioning.cpp
#        include/nn/networks/recurrent_classifier_network.h src/nn/networks/recurrent_classifier_network.cpp
#        src/environments/animal_learning/tracepatterning.cpp pybind_interface/interface.cpp)

#target_link_libraries(FlexibleNN PUBLIC tbb)
#target_link_libraries(FlexibleNN PUBLIC mariadb)
# pybind specific config  end

project(FlexibleNN LANGUAGES CXX)

