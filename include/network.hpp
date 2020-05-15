#pragma once

#include <vector>
#include <random>
#include <array>
#include "matrix.hpp"

class Network
{
    public:
    // contains (number, image_data)
    struct Training_example
    {
        uint8_t label { 0 };
        std::array<uint8_t, 784> img_data {};
    }; 

    enum Flags
    {
        Monitor_eval_cost = 1,
        Monitor_eval_accuracy = 2,
        Monitor_training_cost = 4,
        Monitor_training_accuracy = 8,
        Early_stop = 16,
        Schedule_learning_rate = 32,
    };

    explicit Network(const std::vector<unsigned int> &neuron_counts);
    // training_examples: examples used to train
    // epochs: epochs to train for, or, if Early_stop is enabled, 
    // stop after no improvement is shown after epochs
    // eta: learning rate
    // lambda: regularization term
    // tests: examples used to set hyper-parameters
    // flags: parameters
    void train(std::vector<Training_example> &training_examples,
        unsigned epochs,
        unsigned mini_batch_size,
        double eta = 3.0,
        double lambda = 0.0,
        const std::vector<Training_example> &tests = {},
        int flags = 0);
    void mini_batch_train(const std::vector<Training_example> &mini_batch,
        double eta,
        double lambda,
        size_t n);
    void backprop(const Training_example &x,
        std::vector<std::vector<double>> &bias_gradient,
        std::vector<Matrix<double>> &weight_gradient);
    size_t test(const std::vector<Training_example> &tests) const;
    double total_cost(const std::vector<Training_example> &examples,
        double lambda) const;
    std::vector<double> feedforward(const std::vector<double> &) const;
    
    private:
    bool check_answer(const std::vector<double> &activations,
        uint8_t answer) const;

    static double cost(const std::vector<double> &a, uint8_t label);
    static double sigmoid(double);
    static std::vector<double> sigmoid(const std::vector<double> &);
    static double sigmoid_prime(double);
    static std::vector<double> sigmoid_prime(const std::vector<double> &);
    static std::vector<double> dcost_wrt_a(const std::vector<double> &actual,
        uint8_t expected_val);
    static std::vector<double> dot_product(const Matrix<double> &a,
        const std::vector<double> &b);
    static double dot_product(const std::vector<double> &a,
        const std::vector<double> &b);
    static Matrix<double> outer_product(const std::vector<double> &a,
        const std::vector<double> &b);

    private:
    size_t layers_ {0};
    std::vector<unsigned int> layer_sizes_; // amount of neurons in a layer
    std::vector<std::vector<double>> biases_;
    std::vector<Matrix<double>> weights_;
    std::mt19937 rand_engine_;
};

// TODO:
// overload matrix+vector, matrix-vector