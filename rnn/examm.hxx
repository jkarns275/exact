#ifndef EXAMM_HXX
#define EXAMM_HXX

#include <fstream>
using std::ofstream;

#include <map>
using std::map;

#include <sstream>
using std::ostringstream;

#include <string>
using std::string;
using std::to_string;

#include <vector>
using std::vector;

#include "rnn_genome.hxx"

#define ISLAND_INITIALIZING 0
#define ISLAND_FILLED 1
#define ISLAND_REPOPULATING 2

#define GLOBAL_POPULATION 0
#define ISLAND_POPULATION 1

#define UNIFORM_DISTRIBUTION 0
#define HISTOGRAM_DISTRIBUTION 1
#define NORMAL_DISTRIBUTION 2
#define PHEROMONE_DISTRIBUTION 3

// Forward declare this
class RecDepthPheromoneDist;

class EXAMM {
    private:
        int32_t population_size;
        int32_t number_islands;

        vector<int32_t> island_states;
        vector< vector<RNN_Genome*> > genomes;

        int32_t max_genomes;
        int32_t generated_genomes;
        int32_t inserted_genomes;
        int32_t total_bp_epochs;

        int32_t num_genomes_check_on_island;
        string check_on_island_method;

        int32_t edge_innovation_count;
        int32_t node_innovation_count;

        map<string, int32_t> inserted_from_map;
        map<string, int32_t> generated_from_map;

        int32_t number_inputs;
        int32_t number_outputs;
        int32_t bp_iterations;
        double learning_rate;

        bool use_high_threshold;
        double high_threshold;

        bool use_low_threshold;
        double low_threshold;

        bool use_dropout;
        double dropout_probability;

        minstd_rand0 generator;
        uniform_real_distribution<double> rng_0_1;
        uniform_real_distribution<double> rng_crossover_weight;

        int32_t min_recurrent_depth;
        int32_t max_recurrent_depth;

        bool epigenetic_weights;
        double mutation_rate;
        double crossover_rate;
        double island_crossover_rate;

        double more_fit_crossover_rate;
        double less_fit_crossover_rate;

        double clone_rate;

        double add_edge_rate;
        double add_recurrent_edge_rate;
        double enable_edge_rate;
        double disable_edge_rate;
        double split_edge_rate;

        double add_node_rate;
        double enable_node_rate;
        double disable_node_rate;
        double split_node_rate;
        double merge_node_rate;

        vector<int> possible_node_types;

        string output_directory;
        ofstream *log_file;

        vector<string> input_parameter_names;
        vector<string> output_parameter_names;

        map<string,double> normalize_mins;
        map<string,double> normalize_maxs;

        ostringstream memory_log;

        std::chrono::time_point<std::chrono::system_clock> startClock;
    
    public:
        int32_t rec_sampling_population;
        int32_t rec_sampling_distribution;
        // If using global pop there will only be 1 in here, otherwise
        // size will be equal to the number of islands.
        vector<RecDepthPheromoneDist> rec_sampling_pheromone_dists;

        EXAMM(int32_t _population_size, int32_t _number_islands, int32_t _max_genomes, int32_t _num_genomes_check_on_island, string _check_on_island_method,
            const vector<string> &_input_parameter_names,
            const vector<string> &_output_parameter_names, 
            const map<string,double> &_normalize_mins,
            const map<string,double> &_normalize_maxs,
            int32_t _bp_iterations, double _learning_rate, 
            bool _use_high_threshold, double _high_threshold, 
            bool _use_low_threshold, double _low_threshold, 
            bool _use_dropout, double _dropout_probability,
            int32_t _min_recurrent_depth, int32_t _max_recurrent_depth,
            double decay_rate, double baseline_pheromone,
            string _rec_sampling_population, string _rec_sampling_distribution, string _output_directory);

        ~EXAMM();

        void print_population();
        void write_memory_log(string filename);

        void set_possible_node_types(vector<string> possible_node_type_strings);

        int32_t population_contains(RNN_Genome* genome, int32_t island);
        bool populations_full() const;

        bool insert_genome(RNN_Genome* genome);

        Distribution *get_recurrent_depth_dist(int32_t island);

        int get_random_node_type();

        int32_t check_on_island();
        // methods
        int32_t clear_island_with_worst_best_genome();

        void initialize_genome_parameters(RNN_Genome* genome);
        RNN_Genome* generate_genome();
        void mutate(RNN_Genome *p1);

        void attempt_node_insert(vector<RNN_Node_Interface*> &child_nodes, const RNN_Node_Interface *node, const vector<double> &new_weights);
        void attempt_edge_insert(vector<RNN_Edge*> &child_edges, vector<RNN_Node_Interface*> &child_nodes, RNN_Edge *edge, RNN_Edge *second_edge, bool set_enabled);
        void attempt_recurrent_edge_insert(vector<RNN_Recurrent_Edge*> &child_recurrent_edges, vector<RNN_Node_Interface*> &child_nodes, RNN_Recurrent_Edge *recurrent_edge, RNN_Recurrent_Edge *second_edge, bool set_enabled);
        RNN_Genome* crossover(RNN_Genome *p1, RNN_Genome *p2);

        RNN_Genome* get_best_genome();
        RNN_Genome* get_worst_genome();

        double get_best_fitness();
        double get_worst_fitness();

        string get_output_directory() const;
};

#endif
