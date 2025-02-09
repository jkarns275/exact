#include <algorithm>
using std::sort;
using std::upper_bound;

#include <chrono>

#include <iostream>
using std::cout;
using std::endl;

#include <limits>
using std::numeric_limits;

#include <iomanip>
using std::fixed;
using std::setw;

#include <iostream>
using std::cerr;
using std::endl;

#include <fstream>
using std::ofstream;

#include <random>
using std::minstd_rand0;
using std::uniform_real_distribution;

#include <vector>
using std::vector;


#include "rnn_edge.hxx"
#include "rnn_recurrent_edge.hxx"
#include "rnn_genome.hxx"
#include "rnn_node_interface.hxx"
#include "rnn_node.hxx"
#include "lstm_node.hxx"
#include "gru_node.hxx"
#include "mgu_node.hxx"
#include "mse.hxx"


RNN::RNN(vector<RNN_Node_Interface*> &_nodes, vector<RNN_Edge*> &_edges) {
    nodes = _nodes;
    edges = _edges;

    //sort edges by depth
    sort(edges.begin(), edges.end(), sort_RNN_Edges_by_depth());

    for (uint32_t i = 0; i < nodes.size(); i++) {
        if (nodes[i]->layer_type == INPUT_LAYER) {
            input_nodes.push_back(nodes[i]);
        } else if (nodes[i]->layer_type == OUTPUT_LAYER) {
            output_nodes.push_back(nodes[i]);
        }
    }
}

RNN::RNN(vector<RNN_Node_Interface*> &_nodes, vector<RNN_Edge*> &_edges, vector<RNN_Recurrent_Edge*> &_recurrent_edges) {
    nodes = _nodes;
    edges = _edges;
    recurrent_edges = _recurrent_edges;

    //sort nodes by depth
    //sort edges by depth

    for (uint32_t i = 0; i < nodes.size(); i++) {
        if (nodes[i]->layer_type == INPUT_LAYER) {
            input_nodes.push_back(nodes[i]);
        } else if (nodes[i]->layer_type == OUTPUT_LAYER) {
            output_nodes.push_back(nodes[i]);
        }
    }

    //cout << "got RNN with " << nodes.size() << " nodes, " << edges.size() << ", " << recurrent_edges.size() << " recurrent edges" << endl;
}

RNN::~RNN() {
    RNN_Node_Interface *node;

    while (nodes.size() > 0) {
        node = nodes.back();
        nodes.pop_back();
        delete node;
    }

    RNN_Edge *edge;

    while (edges.size() > 0) {
        edge = edges.back();
        edges.pop_back();
        delete edge;
    }

    RNN_Recurrent_Edge *recurrent_edge;

    while (recurrent_edges.size() > 0) {
        recurrent_edge = recurrent_edges.back();
        recurrent_edges.pop_back();
        delete recurrent_edge;
    }

    while (input_nodes.size() > 0) input_nodes.pop_back();
    while (output_nodes.size() > 0) output_nodes.pop_back();
}


int RNN::get_number_nodes() {
    return nodes.size();
}

int RNN::get_number_edges() {
    return edges.size();
}

RNN_Node_Interface* RNN::get_node(int i) {
    return nodes[i];
}

RNN_Edge* RNN::get_edge(int i) {
    return edges[i];
}


void RNN::get_weights(vector<double> &parameters) {
    parameters.resize(get_number_weights());

    uint32_t current = 0;

    for (uint32_t i = 0; i < nodes.size(); i++) {
        nodes[i]->get_weights(current, parameters);
        //if (nodes[i]->is_reachable()) nodes[i]->get_weights(current, parameters);
    }

    for (uint32_t i = 0; i < edges.size(); i++) {
        parameters[current++] = edges[i]->weight;
        //if (edges[i]->is_reachable()) parameters[current++] = edges[i]->weight;
    }

    for (uint32_t i = 0; i < recurrent_edges.size(); i++) {
        parameters[current++] = recurrent_edges[i]->weight;
        //if (recurrent_edges[i]->is_reachable()) parameters[current++] = recurrent_edges[i]->weight;
    }
}

void RNN::set_weights(const vector<double> &parameters) {
    if (parameters.size() != get_number_weights()) {
        cerr << "ERROR! Trying to set weights where the RNN has " << get_number_weights() << " weights, and the parameters vector has << " << parameters.size() << " weights!" << endl;
        exit(1);
    }

    uint32_t current = 0;

    for (uint32_t i = 0; i < nodes.size(); i++) {
        nodes[i]->set_weights(current, parameters);
        //if (nodes[i]->is_reachable()) nodes[i]->set_weights(current, parameters);
    }

    for (uint32_t i = 0; i < edges.size(); i++) {
        edges[i]->weight = parameters[current++];
        //if (edges[i]->is_reachable()) edges[i]->weight = parameters[current++];
    }

    for (uint32_t i = 0; i < recurrent_edges.size(); i++) {
        recurrent_edges[i]->weight = parameters[current++];
        //if (recurrent_edges[i]->is_reachable()) recurrent_edges[i]->weight = parameters[current++];
    }

}

uint32_t RNN::get_number_weights() {
    uint32_t number_weights = 0;

    for (uint32_t i = 0; i < nodes.size(); i++) {
        number_weights += nodes[i]->get_number_weights();
        //if (nodes[i]->is_reachable()) number_weights += nodes[i]->get_number_weights();
    }

    for (uint32_t i = 0; i < edges.size(); i++) {
        number_weights++;
        //if (edges[i]->is_reachable()) number_weights++;
    }

    for (uint32_t i = 0; i < recurrent_edges.size(); i++) {
        number_weights++;
        //if (recurrent_edges[i]->is_reachable()) number_weights++;
    }

    return number_weights;
}

void RNN::forward_pass(const vector< vector<double> > &series_data, bool using_dropout, bool training, double dropout_probability) {
    series_length = series_data[0].size();

    if (input_nodes.size() != series_data.size()) {
        cerr << "ERROR: number of input nodes (" << input_nodes.size() << ") != number of time series data input fields (" << series_data.size() << ")" << endl;
        exit(1);
    }

    //TODO: want to check that all vectors in series_data are of same length


    for (uint32_t i = 0; i < nodes.size(); i++) {
        nodes[i]->reset(series_length);
    }

    for (uint32_t i = 0; i < edges.size(); i++) {
        edges[i]->reset(series_length);
    }

    for (uint32_t i = 0; i < recurrent_edges.size(); i++) {
        recurrent_edges[i]->reset(series_length);
    }

    //do a propagate forward for time == -1 so that the the input
    //fired count on each node will be correct for the first pass
    //through the RNN
    for (uint32_t i = 0; i < recurrent_edges.size(); i++) {
        if (recurrent_edges[i]->is_reachable()) recurrent_edges[i]->first_propagate_forward();
    }

    for (int32_t time = 0; time < series_length; time++) {
        for (uint32_t i = 0; i < input_nodes.size(); i++) {
            if(input_nodes[i]->is_reachable()) input_nodes[i]->input_fired(time, series_data[i][time]);
        }

        //feed forward
        if (using_dropout) {
            for (uint32_t i = 0; i < edges.size(); i++) {
                if (edges[i]->is_reachable()) edges[i]->propagate_forward(time, training, dropout_probability);
            }
        } else {
            for (uint32_t i = 0; i < edges.size(); i++) {
                if (edges[i]->is_reachable()) edges[i]->propagate_forward(time);
            }
        }

        for (uint32_t i = 0; i < recurrent_edges.size(); i++) {
            if (recurrent_edges[i]->is_reachable()) recurrent_edges[i]->propagate_forward(time);
        }
    }
}

void RNN::backward_pass(double error, bool using_dropout, bool training, double dropout_probability) {
    //do a propagate forward for time == (series_length - 1) so that the
    // output fired count on each node will be correct for the first pass
    //through the RNN
    for (uint32_t i = 0; i < recurrent_edges.size(); i++) {
        if (recurrent_edges[i]->is_reachable()) recurrent_edges[i]->first_propagate_backward();
    }

    for (int32_t time = series_length - 1; time >= 0; time--) {

        for (uint32_t i = 0; i < output_nodes.size(); i++) {
            output_nodes[i]->error_fired(time, error);
        }

        if (using_dropout) {
            for (int32_t i = (int32_t)edges.size() - 1; i >= 0; i--) {
                if (edges[i]->is_reachable()) edges[i]->propagate_backward(time, training, dropout_probability);
            }
        } else {
            for (int32_t i = (int32_t)edges.size() - 1; i >= 0; i--) {
                if (edges[i]->is_reachable()) edges[i]->propagate_backward(time);
            }
        }

        for (int32_t i = (int32_t)recurrent_edges.size() - 1; i >= 0; i--) {
            if (recurrent_edges[i]->is_reachable()) recurrent_edges[i]->propagate_backward(time);
        }
    }
}


double RNN::calculate_error_mse(const vector< vector<double> > &expected_outputs) {
    double mse_sum = 0.0;
    double mse;
    double error;
    for (uint32_t i = 0; i < output_nodes.size(); i++) {
        output_nodes[i]->error_values.resize(expected_outputs[i].size());

        mse = 0.0;
        for (uint32_t j = 0; j < expected_outputs[i].size(); j++) {
            error = output_nodes[i]->output_values[j] - expected_outputs[i][j];
            output_nodes[i]->error_values[j] = error;
            mse += error * error;
        }
        mse_sum += mse / expected_outputs[i].size();
    }

    return mse_sum;
}

double RNN::calculate_error_mae(const vector< vector<double> > &expected_outputs) {
    double mae_sum = 0.0;
    double mae;
    double error;
    for (uint32_t i = 0; i < output_nodes.size(); i++) {
        output_nodes[i]->error_values.resize(expected_outputs[i].size());

        mae = 0.0;
        for (uint32_t j = 0; j < expected_outputs[i].size(); j++) {
            error = fabs(output_nodes[i]->output_values[j] - expected_outputs[i][j]);

            mae += error;

            if (error == 0) {
                error = 0;
            } else {
                error = (output_nodes[i]->output_values[j] - expected_outputs[i][j]) / error;
            }
            output_nodes[i]->error_values[j] = error;

        }
        mae_sum += mae / expected_outputs[i].size();
    }

    return mae_sum;
}

double RNN::prediction_mse(const vector< vector<double> > &series_data, const vector< vector<double> > &expected_outputs, bool using_dropout, bool training, double dropout_probability) {
    forward_pass(series_data, using_dropout, training, dropout_probability);
    return calculate_error_mse(expected_outputs);
}

double RNN::prediction_mae(const vector< vector<double> > &series_data, const vector< vector<double> > &expected_outputs, bool using_dropout, bool training, double dropout_probability) {
    forward_pass(series_data, using_dropout, training, dropout_probability);
    return calculate_error_mae(expected_outputs);
}

vector<double> RNN::get_predictions(const vector< vector<double> > &series_data, const vector< vector<double> > &expected_outputs, bool using_dropout, double dropout_probability) {
    forward_pass(series_data, using_dropout, false, dropout_probability);

    vector<double> result;

    for (uint32_t j = 0; j < series_length; j++) {
        for (uint32_t i = 0; i < output_nodes.size(); i++) {
            result.push_back(output_nodes[i]->output_values[j]);
        }
    }

    //TODO: returning a vector isn't the most efficient, but i don't think we'll be using this for things that are performance
    //critical -- Travis
    return result;
}


void RNN::write_predictions(string output_filename, const vector<string> &input_parameter_names, const vector<string> &output_parameter_names, const vector< vector<double> > &series_data, const vector< vector<double> > &expected_outputs, bool using_dropout, double dropout_probability) {
    forward_pass(series_data, using_dropout, false, dropout_probability);


    cerr << "series_length: " << series_length << ", series_data.size(): " << series_data.size() << ", series_data[0].size(): " << series_data[0].size() << endl;
    cerr << "input_nodes.size(): " << input_nodes.size() << ", output_nodes.size(): " << output_nodes.size() << endl;
    ofstream outfile(output_filename);

    outfile << "#";

    for (uint32_t i = 0; i < input_nodes.size(); i++) {
        if (i > 0) outfile << ",";
        outfile << input_parameter_names[i];

        cerr << "input_parameter_names[" << i << "]: " << input_parameter_names[i] << endl;
    }

    for (uint32_t i = 0; i < output_nodes.size(); i++) {
        outfile << ",";
        outfile << output_parameter_names[i];

        cerr << "output_parameter_names[" << i << "]: " << output_parameter_names[i] << endl;
    }

    for (uint32_t i = 0; i < output_nodes.size(); i++) {
        outfile << ",";
        outfile << "predicted_" << output_parameter_names[i];

        cerr << "output_parameter_names[" << i << "]: " << output_parameter_names[i] << endl;
    }
    outfile << endl;

    for (uint32_t j = 0; j < series_length; j++) {
        for (uint32_t i = 0; i < input_nodes.size(); i++) {
            if (i > 0) outfile << ",";
            outfile << series_data[i][j];
        }

        for (uint32_t i = 0; i < output_nodes.size(); i++) {
            outfile << ",";
            outfile << expected_outputs[i][j];
        }

        for (uint32_t i = 0; i < output_nodes.size(); i++) {
            outfile << ",";
            outfile << output_nodes[i]->output_values[j];
        }
        outfile << endl;
    }
    outfile.close();
}

void RNN::get_analytic_gradient(const vector<double> &test_parameters, const vector< vector<double> > &inputs, const vector< vector<double> > &outputs, double &mse, vector<double> &analytic_gradient, bool using_dropout, bool training, double dropout_probability) {
    analytic_gradient.assign(test_parameters.size(), 0.0);

    set_weights(test_parameters);
    forward_pass(inputs, using_dropout, training, dropout_probability);

    mse = calculate_error_mse(outputs);

    backward_pass(mse * (1.0 / outputs[0].size()) * 2.0, using_dropout, training, dropout_probability);

    vector<double> current_gradients;

    uint32_t current = 0;
    for (uint32_t i = 0; i < nodes.size(); i++) {
        if (nodes[i]->is_reachable()) {
            nodes[i]->get_gradients(current_gradients);

            for (uint32_t j = 0; j < current_gradients.size(); j++) {
                analytic_gradient[current] = current_gradients[j];
                current++;
            }
        }
    }

    for (uint32_t i = 0; i < edges.size(); i++) {
        if (edges[i]->is_reachable()) {
            analytic_gradient[current] = edges[i]->get_gradient();
            current++;
        }
    }

    for (uint32_t i = 0; i < recurrent_edges.size(); i++) {
        if (recurrent_edges[i]->is_reachable()) {
            analytic_gradient[current] = recurrent_edges[i]->get_gradient();
            current++;
        }
    }
}

void RNN::get_empirical_gradient(const vector<double> &test_parameters, const vector< vector<double> > &inputs, const vector< vector<double> > &outputs, double &mse, vector<double> &empirical_gradient, bool using_dropout, bool training, double dropout_probability) {
    empirical_gradient.assign(test_parameters.size(), 0.0);

    vector< vector<double> > deltas;


    set_weights(test_parameters);
    forward_pass(inputs, using_dropout, training, dropout_probability);
    double original_mse = calculate_error_mse(outputs);

    //cout << "EMPIRICAL statistics: " << endl;
    //print_output_statistics(outputs);

    double save;
    double diff = 0.00001;
    double mse1, mse2;

    vector<double> parameters = test_parameters;
    for (uint32_t i = 0; i < parameters.size(); i++) {
        save = parameters[i];

        parameters[i] = save - diff;
        set_weights(parameters);
        forward_pass(inputs, using_dropout, training, dropout_probability);
        get_mse(this, outputs, mse1, deltas);

        parameters[i] = save + diff;
        set_weights(parameters);
        forward_pass(inputs, using_dropout, training, dropout_probability);
        get_mse(this, outputs, mse2, deltas);

        empirical_gradient[i] = (mse2 - mse1) / (2.0 * diff);
        empirical_gradient[i] *= original_mse;

        parameters[i] = save;
    }

    mse = original_mse;
}

void RNN::initialize_randomly() {
    int number_of_weights = get_number_weights();
    vector<double> parameters(number_of_weights, 0.0);

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    minstd_rand0 generator(seed);
    uniform_real_distribution<double> rng(-0.5, 0.5);
    for (uint32_t i = 0; i < parameters.size(); i++) {
        parameters[i] = rng(generator);
    }
    set_weights(parameters);
}

RNN* RNN::copy() {
    vector<RNN_Node_Interface*> node_copies;
    vector<RNN_Edge*> edge_copies;
    vector<RNN_Recurrent_Edge*> recurrent_edge_copies;

    for (uint32_t i = 0; i < nodes.size(); i++) {
        node_copies.push_back( nodes[i]->copy() );
    }

    for (uint32_t i = 0; i < edges.size(); i++) {
        edge_copies.push_back( edges[i]->copy(node_copies) );
    }

    for (uint32_t i = 0; i < recurrent_edges.size(); i++) {
        recurrent_edge_copies.push_back( recurrent_edges[i]->copy(node_copies) );
    }


    return new RNN(node_copies, edge_copies, recurrent_edge_copies);
}

