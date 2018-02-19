#ifndef EXALT_RNN_GENOME_HXX
#define EXALT_RNN_GENOME_HXX

#include <string>
using std::string;

#include <vector>
using std::vector;

class RNN_Genome {
    private:
        vector<RNN_Node_Interface*> input_nodes;
        vector<RNN_Node_Interface*> output_nodes;

        vector<RNN_Node_Interface*> nodes;
        vector<RNN_Edge*> edges;

    public:
        RNN_Genome(vector<RNN_Node_Interface*> &_nodes, vector<RNN_Edge*> &_edges);

        double classify(const vector< vector<double> > &series_data, double expected_class);
        double predict(const vector< vector<double> > &series_data, const vector<string> &fields);
        double test_backprop(const vector< vector<double> > &series_data, const vector<string> &fields);

        void set_weights(const vector<double> &parameters);

        uint32_t get_number_weights();
};

#endif
