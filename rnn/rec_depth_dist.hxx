#ifndef RNN_REC_DEPTH_DIST
#define RNN_REC_DEPTH_DIST

#include <chrono>

// random included in rnn_genome.hxx
#include <random>
using std::normal_distribution;
using std::mt19937;

#include <vector>
using std::vector;

#include "distribution.hxx"
#include "rnn_genome.hxx"

class RecDepthFrequencyTable {
    public:
        int32_t *frequencies;
        int32_t n_samples;

        RecDepthFrequencyTable();

        RecDepthFrequencyTable(vector<RNN_Genome*> &genomes, 
                                int32_t _min_recurrent_depth, int32_t max_recurrent_depth);

        RecDepthFrequencyTable(vector<vector<RNN_Genome*>> &islands, 
                                int32_t _min_recurrent_depth, int32_t max_recurrent_depth);

        ~RecDepthFrequencyTable();

        int32_t &operator[] (int32_t i);

    private:
        void count_island_frequencies(vector<RNN_Genome*> &genomes);
};

class RecDepthNormalDist : public Distribution {
    private:
        int32_t min, max;
        normal_distribution<double> distribution;

    public:
        RecDepthNormalDist(vector<RNN_Genome*> &genomes, int32_t min_recurrent_depth, int32_t max_recurrent_depth); 
        RecDepthNormalDist(vector<vector<RNN_Genome*>> &islands, int32_t min_recurrent_depth, int32_t max_recurrent_depth);
        virtual ~RecDepthNormalDist() {}
        int32_t sample() override;
    
    private:
        void calculate_distribution(RecDepthFrequencyTable &freqs);
};

class RecDepthHistDist : public Distribution {
    private:
        int32_t min, max;
        vector<int32_t> distribution;

    public:
        RecDepthHistDist(vector<RNN_Genome*> &genomes, int32_t min_recurrent_depth, int32_t max_recurrent_depth);
        RecDepthHistDist(vector<vector<RNN_Genome*>> &islands, int32_t min_recurrent_depth, int32_t max_recurrent_depth);
        virtual ~RecDepthHistDist() {}
        int32_t sample() override;

    private:
        void calculate_distribution(RecDepthFrequencyTable& freqs);
};

class RecDepthUniformDist : public Distribution {
    private:
        int32_t min, max;

    public:
        RecDepthUniformDist(int32_t min_recurrent_depth, int32_t max_recurrent_depth);
        ~RecDepthUniformDist() {}
        int32_t sample() override;
};

class RecDepthPheromoneDist : public Distribution {
    private:
        int32_t min, max;
        vector<double> dist;
        double decay_rate, baseline_pheromone;
    public:
        RecDepthPheromoneDist(int32_t min, int32_t max, double decay_rate, double baseline_pheromone);
        ~RecDepthPheromoneDist() {}
        int32_t sample() override;
        void decay();
        void deposit(int32_t i);
};

#endif
