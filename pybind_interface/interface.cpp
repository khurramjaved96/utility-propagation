#include <pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include "../include/nn/networks/neural_network.h"
#include "../include/nn/networks/recurrent_network.h"
#include "../include/nn/synapse.h"
#include "../include/experiment/Metric.h"

namespace py = pybind11;

PYBIND11_MODULE(FlexibleNN, m) {
    py::class_<NeuralNetwork>(m, "NeuralNetwork")
        .def(py::init<>())
        .def_readonly("all_neurons", &NeuralNetwork::all_neurons)
        .def_readonly("output_neurons", &NeuralNetwork::output_neurons)
        .def_readonly("input_neurons", &NeuralNetwork::input_neurons)
        .def_readonly("recurrent_features", &NeuralNetwork::recurrent_features)
        .def_readonly("all_synapses", &NeuralNetwork::all_synapses)
        .def_readonly("output_synapses", &NeuralNetwork::output_synapses)
        .def_readonly("all_heap_elements", &NeuralNetwork::all_heap_elements)
        .def("collect_garbage", &NeuralNetwork::collect_garbage)
        .def("get_timestep", &NeuralNetwork::get_timestep)
        .def("set_input_values", &NeuralNetwork::set_input_values)
        .def("step", &NeuralNetwork::step)
        .def("read_output_values", &NeuralNetwork::read_output_values)
        .def("read_all_values", &NeuralNetwork::read_all_values)
        .def("real_all_weights", &NeuralNetwork::read_all_weights)
        .def("introduce_targets", &NeuralNetwork::introduce_targets)
        .def("forward_pass_without_side_effects", &NeuralNetwork::forward_pass_without_side_effects)
        .def("get_input_size", &NeuralNetwork::get_input_size)
        .def("print_synapse_status", &NeuralNetwork::print_synapse_status)
        .def("print_neuron_status", &NeuralNetwork::print_neuron_status)
        .def("get_total_synapses", &NeuralNetwork::get_total_synapses)
        .def("get_total_neurons", &NeuralNetwork::get_total_neurons)
        .def("reset_trace", &NeuralNetwork::reset_trace)
        .def("viz_graph", &NeuralNetwork::viz_graph);


    py::class_<RecurrentNetwork, NeuralNetwork>(m, "RecurrentNetwork")
        .def(py::init<float, int, int, int, int, int>())
        .def_readonly("active_synapses", &RecurrentNetwork::active_synapses)
        .def_readonly("Recurrent_neuron_layer", &RecurrentNetwork::Recurrent_neuron_layer)
        .def("replace_feature", &RecurrentNetwork::replace_feature)
        .def("print_graph", &RecurrentNetwork::print_graph)
        .def("forward", &RecurrentNetwork::forward)
        .def("backward", &RecurrentNetwork::backward);

    py::class_<Metric>(m, "Metric")
        .def(py::init<std::string, std::string, std::vector<std::string>, std::vector<std::string>, std::vector<std::string>>())
        .def("add_value", &Metric::add_value)
        .def("add_values", &Metric::add_values)
        .def("record_value", &Metric::record_value)
        .def("commit_values", &Metric::commit_values);

    py::class_<Database>(m, "Database")
        .def(py::init<>())
        .def("create_database", &Database::create_database);

    py::class_<Synapse>(m, "Synapse")
        .def_readonly("id", &Synapse::id)
        .def_readonly("is_useless", &Synapse::is_useless)
        .def_readonly("age", &Synapse::age)
        .def_readonly("weight", &Synapse::weight)
        .def_readonly("credit", &Synapse::credit)
        .def_readonly("trace", &Synapse::trace)
        .def_readonly("step_size", &Synapse::step_size)
        .def_readonly("TH", &Synapse::TH)
        .def_readonly("meta_step_size", &Synapse::meta_step_size)
        .def_readonly("propagate_gradients", &Synapse::propagate_gradients)
        .def_readonly("disable_utility", &Synapse::disable_utility)
        .def_readonly("utility_to_keep", &Synapse::utility_to_keep)
        .def_readonly("synapse_utility", &Synapse::synapse_utility)
        .def_readonly("synapse_utility_to_distribute", &Synapse::synapse_utility_to_distribute)
        .def_readonly("input_neuron", &Synapse::input_neuron)
        .def_readonly("output_neuron", &Synapse::output_neuron);


    py::class_<Neuron>(m, "Neuron")
        .def_readonly("is_recurrent_neuron", &Neuron::is_recurrent_neuron)
        .def_readonly("id", &Neuron::id)
        .def_readonly("useless_neuron", &Neuron::useless_neuron)
        .def_readonly("neuron_age", &Neuron::neuron_age)
        .def_readonly("is_input_neuron", &Neuron::is_input_neuron)
        .def_readonly("is_output_neuron", &Neuron::is_output_neuron)
        .def_readonly("value", &Neuron::value)
        .def_readonly("neuron_utility", &Neuron::neuron_utility)
        .def_readonly("sum_of_utility_traces", &Neuron::sum_of_utility_traces)
        .def_readonly("incoming_synapses", &Neuron::incoming_synapses)
        .def_readonly("outgoing_synapses", &Neuron::outgoing_synapses);

    py::class_<RecurrentRelu, Neuron>(m, "RecurrentRelu")
        .def_readonly("learning", &RecurrentRelu::learning);
}
