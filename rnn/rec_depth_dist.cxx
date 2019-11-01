#include "rec_depth_dist.hxx"
#include <assert.h>

RecDepthFrequencyTable::RecDepthFrequencyTable() {}

RecDepthFrequencyTable::RecDepthFrequencyTable(
        vector<RNN_Genome*> &genomes, int32_t _min_recurrent_depth, int32_t max_recurrent_depth) {
    n_samples = 0;
    frequencies = new int32_t[max_recurrent_depth + 1](); // the () ensures the memory is zeroed
    count_island_frequencies(genomes);
}

RecDepthFrequencyTable::RecDepthFrequencyTable(
        vector<vector<RNN_Genome*>> &islands, int32_t _min_recurrent_depth, int32_t max_recurrent_depth) {
    n_samples = 0;
    frequencies = new int32_t[max_recurrent_depth + 1](); // the () ensures the memory is zeroed
    for (int32_t i = 0; i < islands.size(); i += 1) count_island_frequencies(islands[i]);
}

RecDepthFrequencyTable::~RecDepthFrequencyTable() {
    if (frequencies) {
        delete frequencies;
        frequencies = 0;
    }
}

int32_t & RecDepthFrequencyTable::operator[] (int32_t i) {
    return frequencies[i];
}

void RecDepthFrequencyTable::count_island_frequencies(vector<RNN_Genome*> &genomes) {
    for (int32_t gi = 0; gi < genomes.size(); gi += 1) {
        RNN_Genome *genome = genomes[gi];
        for (int32_t ei = 0; ei < genome->recurrent_edges.size(); ei += 1) {
            RNN_Recurrent_Edge *edge = genome->recurrent_edges[ei];
            // TODO: Figure out if i should count disabled edges as well?
            // intuition says no.
            if (edge->enabled) {
                n_samples += 1;
                frequencies[edge->recurrent_depth] += 1;
            }
        }
    }
}

RecDepthNormalDist::RecDepthNormalDist(vector<RNN_Genome*> &genomes, 
            int32_t min_recurrent_depth, int32_t max_recurrent_depth)
    : Distribution() {
    min = min_recurrent_depth;
    max = max_recurrent_depth;

    RecDepthFrequencyTable freqs(genomes, min_recurrent_depth, max_recurrent_depth);
    calculate_distribution(freqs);
}

RecDepthNormalDist::RecDepthNormalDist(vector<vector<RNN_Genome*>> &islands,
                int32_t min_recurrent_depth, int32_t max_recurrent_depth) 
    : Distribution() {
    min = min_recurrent_depth;
    max = max_recurrent_depth;

    RecDepthFrequencyTable freqs(islands, min_recurrent_depth, max_recurrent_depth);
    calculate_distribution(freqs);
}

void RecDepthNormalDist::calculate_distribution(RecDepthFrequencyTable &freqs) {
    // Yes, inclusive range
    int32_t n_samples, sum;
    for (int32_t i = min; i <= max; i += 1) {
        assert(sum >= 0); // Check for overflows! 
        int32_t f = freqs[i];
        n_samples += f;
        sum += (f * i);
    }
    double mean = ((double) sum) / ((double) n_samples);
    
    double total_sq_deviation = 0.0;
    for (int32_t i = min; i <= max; i += 1) {
        double f = freqs[i];
        double deviation = ((double) i) - mean;
        total_sq_deviation += f * deviation * deviation;
    }
    double sd = sqrt(total_sq_deviation / n_samples);
    distribution = normal_distribution<double>(mean, sd);
}

int32_t RecDepthNormalDist::sample() {
start:;
    double sample = distribution(rng);
    int32_t rounded = round(sample);
    if (rounded < min || rounded > max)
        goto start; // Try again since it is not within our range
    return rounded;
}

RecDepthHistDist::RecDepthHistDist(vector<RNN_Genome*> &genomes,
                int32_t min_recurrent_depth, int32_t max_recurrent_depth)
    : Distribution() {
    min = min_recurrent_depth;
    max = max_recurrent_depth;

    RecDepthFrequencyTable freqs(genomes, min_recurrent_depth, max_recurrent_depth);

    calculate_distribution(freqs);
}
                                                                          
RecDepthHistDist::RecDepthHistDist(vector<vector<RNN_Genome*>> &islands,
                int32_t min_recurrent_depth, int32_t max_recurrent_depth)
    : Distribution() {
    min = min_recurrent_depth;
    max = max_recurrent_depth;
    RecDepthFrequencyTable freqs(islands, min_recurrent_depth, max_recurrent_depth);
    
    calculate_distribution(freqs);   
}
                                                                          
void RecDepthHistDist::calculate_distribution(RecDepthFrequencyTable &freqs) {
    // The number of samples plus one extra slot for every value within min and max
    distribution.reserve(freqs.n_samples + 1 + (max - min));
    for (int32_t i = min; i <= max; i += 1) {
        int32_t slots = freqs[i] + 1;
        for (int32_t j = 0; j < slots; j += 1) {
            distribution.push_back(i);
        }
    }
}

int32_t RecDepthHistDist::sample() {
    int32_t size = distribution.size();
    // Ignore the sign bit since some RNGs return unsigned numbers.
    int32_t r = rng() & 0x7FFFFFFF;
    int32_t index = r % size;
    return distribution[index];
}

RecDepthUniformDist::RecDepthUniformDist(int32_t min_recurrent_depth, int32_t max_recurrent_depth)
    : Distribution() {
    min = min_recurrent_depth;
    max = max_recurrent_depth;
}

int32_t RecDepthUniformDist::sample() {
    // Ignore the sign bit since some RNGs return unsigned numbers.
    int32_t rand_int = rng() & 0x7FFFFFFF; 
    return min + (rand_int % (max - min));
}

RecDepthPheromoneDist::RecDepthPheromoneDist(int32_t _min, int32_t _max, 
        double _decay_rate, double _baseline_pheromone) : Distribution() {
    min = _min; max = _max; decay_rate = _decay_rate; baseline_pheromone = _baseline_pheromone;
    dist = vector(max + 1, 0.0);
}

int32_t RecDepthPheromoneDist::sample() {
    // Calculate the sum of pheromones
    double sum = 0.0;
    for (int32_t i = min; i <= max; i += 1) {
        sum += dist[i] + baseline_pheromone;
    }
    
    // random number between 0 and sum of pheromones
    std::uniform_real_distribution<double> uniform(0.0, sum);
    double r = uniform(rng);
    
    // Sample the distribution with r
    sum = 0.0;
    for (int32_t i = min; i <= max; i += 1) {
        sum += dist[i] + baseline_pheromone;
        if (r < sum)
            return i;
    }
    return max;
}

void RecDepthPheromoneDist::decay() {
    for (int32_t i = min; i <= max; i += 1)
        dist[i] *= decay_rate;
}

void RecDepthPheromoneDist::deposit(int32_t ix) {
    for (int32_t i = min; i <= max; i += 1) {
        int32_t power = abs(i - ix);
        double d = pow(2.0, power);
        dist[i] += 1.0 / d;
    }
}
