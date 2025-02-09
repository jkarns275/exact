#include <chrono>

#include <condition_variable>
using std::condition_variable;

#include <iomanip>
using std::setw;
using std::fixed;

#include <iostream>
using std::cerr;
using std::cout;
using std::endl;

#include <mutex>
using std::mutex;

#include <string>
using std::string;

#include <thread>
using std::thread;

#include <vector>
using std::vector;

//for mkdir
#include <sys/stat.h>

#include "common/arguments.hxx"

#include "rnn/examm.hxx"

#include "time_series/time_series.hxx"

mutex examm_mutex;

vector<string> arguments;

EXAMM *examm;

bool finished = false;

string output_directory = "";

vector< vector< vector<double> > > training_inputs;
vector< vector< vector<double> > > training_outputs;
vector< vector< vector<double> > > validation_inputs;
vector< vector< vector<double> > > validation_outputs;

void examm_thread(int id) {

    while (true) {
        examm_mutex.lock();
        RNN_Genome *genome = examm->generate_genome();
        examm_mutex.unlock();

        if (genome == NULL) break;  //generate_individual returns NULL when the search is done

        genome->backpropagate_stochastic(training_inputs, training_outputs, validation_inputs, validation_outputs);

        examm_mutex.lock();
        examm->insert_genome(genome);
        examm_mutex.unlock();

        delete genome;
    }
}

int main(int argc, char** argv) {
    arguments = vector<string>(argv, argv + argc);

    int number_threads;
    get_argument(arguments, "--number_threads", true, number_threads);

    int32_t time_offset = 1;
    get_argument(arguments, "--time_offset", true, time_offset);

    int32_t population_size;
    get_argument(arguments, "--population_size", true, population_size);

    int32_t number_islands;
    get_argument(arguments, "--number_islands", true, number_islands);

    int32_t max_genomes;
    get_argument(arguments, "--max_genomes", true, max_genomes);

    int32_t num_genomes_check_on_island;
    get_argument(arguments, "--num_genomes_check_on_island", false, num_genomes_check_on_island);
    
    string check_on_island_method = "";
    get_argument(arguments, "--check_on_island_method", false, check_on_island_method);

    int32_t bp_iterations;
    get_argument(arguments, "--bp_iterations", true, bp_iterations);

    double learning_rate = 0.001;
    get_argument(arguments, "--learning_rate", false, learning_rate);

    double high_threshold = 1.0;
    bool use_high_threshold = get_argument(arguments, "--high_threshold", false, high_threshold);

    double low_threshold = 0.05;
    bool use_low_threshold = get_argument(arguments, "--low_threshold", false, low_threshold);

    double dropout_probability = 0.0;
    bool use_dropout = get_argument(arguments, "--dropout_probability", false, dropout_probability);

    int32_t rec_delay_min = 1;
    get_argument(arguments, "--rec_delay_min", false, rec_delay_min);
    
    int32_t rec_delay_max = 10;
    get_argument(arguments, "--rec_delay_max", false, rec_delay_max);

    string rec_sampling_population = "global";
    get_argument(arguments, "--rec_sampling_population", false, rec_sampling_population);

    string rec_sampling_distribution = "uniform";
    get_argument(arguments, "--rec_sampling_distribution", false, rec_sampling_distribution);

    double decay_rate = 0;
    get_argument(arguments, "--rec_depth_pheromone_decay_rate", false, decay_rate);

    double baseline_pheromone = 0;
    get_argument(arguments, "--rec_depth_pheromone_baseline", false, baseline_pheromone);

    get_argument(arguments, "--output_directory", true, output_directory);

    string output_filename;
    get_argument(arguments, "--output_filename", true, output_filename);

    TimeSeriesSets *time_series_sets = TimeSeriesSets::generate_from_arguments(arguments);

    int32_t number_slices;
    get_argument(arguments, "--number_slices", true, number_slices);

    time_series_sets->split_all(number_slices);

    int32_t repeats = 5;

    if (output_directory != "") {
        mkdir(output_directory.c_str(), 0777);
    }
    ofstream overall_results(output_directory + "/overall_results.txt");

    for (uint32_t i = 0; i < time_series_sets->get_number_series(); i++) {
        vector<int> training_indexes;
        vector<int> test_indexes;

        for (int j = 0; j < time_series_sets->get_number_series(); j++) {
            if (j == i) {
                test_indexes.push_back(j);
            } else {
                training_indexes.push_back(j);
            }

        }
        time_series_sets->set_training_indexes(training_indexes);
        time_series_sets->set_test_indexes(training_indexes);

        time_series_sets->export_training_series(time_offset, training_inputs, training_outputs);
        time_series_sets->export_test_series(time_offset, validation_inputs, validation_outputs);

        overall_results << "results for slice " << i << " of " << time_series_sets->get_number_series() << " as test data." << endl;

        for (uint32_t k = 0; k < repeats; k++) {
            examm = new EXAMM(population_size, number_islands, max_genomes, num_genomes_check_on_island, check_on_island_method,
                time_series_sets->get_input_parameter_names(),
                time_series_sets->get_output_parameter_names(),
                time_series_sets->get_normalize_mins(),
                time_series_sets->get_normalize_maxs(), 
                bp_iterations, learning_rate, 
                use_high_threshold, high_threshold,
                use_low_threshold, low_threshold,
                use_dropout, dropout_probability,
                rec_delay_min, rec_delay_max,
                decay_rate, baseline_pheromone,
                rec_sampling_population, rec_sampling_distribution,
                output_directory + "/slice_" + to_string(i) + "_repeat_" + to_string(k));

            vector<thread> threads;
            for (int32_t i = 0; i < number_threads; i++) {
                threads.push_back( thread(examm_thread, i) );
            }

            for (int32_t i = 0; i < number_threads; i++) {
                threads[i].join();
            }

            finished = true;

            cout << "completed!" << endl;

            RNN_Genome *best_genome = examm->get_best_genome();

            vector<double> best_parameters = best_genome->get_best_parameters();
            cout << "training MSE: " << best_genome->get_mse(best_parameters, training_inputs, training_outputs) << endl;
            cout << "training MSE: " << best_genome->get_mae(best_parameters, training_inputs, training_outputs) << endl;
            cout << "validation MSE: " << best_genome->get_mse(best_parameters, validation_inputs, validation_outputs) << endl;
            cout << "validation MSE: " << best_genome->get_mae(best_parameters, validation_inputs, validation_outputs) << endl;

            overall_results << setw(15) << fixed << best_genome->get_mse(best_parameters, training_inputs, training_outputs) << ", "
                << setw(15) << fixed << best_genome->get_mae(best_parameters, training_inputs, training_outputs) << ", "
                << setw(15) << fixed << best_genome->get_mse(best_parameters, validation_inputs, validation_outputs) << ", "
                << setw(15) << fixed << best_genome->get_mae(best_parameters, validation_inputs, validation_outputs) << endl;

            best_genome->write_to_file(output_directory + "/" + output_filename + "_slice_" + to_string(i) + "_repeat_" + to_string(k) + ".bin", false);
            best_genome->write_graphviz(output_directory + "/" + output_filename + "_slice_" + to_string(i) + "_repeat_" + to_string(k) + ".gv");

            cout << "deleting genome" << endl;
            delete best_genome;
            cout << "deleting exact" << endl;
            delete examm;
            cout << "deleted exact" << endl;
        }
        overall_results << endl;
    }

    return 0;
}
