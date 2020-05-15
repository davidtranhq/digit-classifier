#include "network.hpp"

#include <random> // normal_distribution
#include <stdexcept>
#include <algorithm> // shuffle, for_each
#include <cmath> // exp
#include <iostream> // cout

// elementwise vector multiplication, Hadamard product
template <typename T>
std::vector<T> operator*(const std::vector<T> &a, const std::vector<T> &b)
{
    size_t len { a.size() };
    if (len != b.size())
        throw std::runtime_error("Can't multiply vectors of different \
            sizes");
    std::vector<T> out(len);
    for (size_t i = 0; i < len; ++i)
    {
        out[i] = a[i] * b[i];
    }
    return out;
}

// vector*scalar
template <typename T>
std::vector<T> operator*(const std::vector<T> &v, double s)
{
    size_t len { v.size() };
    std::vector<T> out(len);
    for (size_t i = 0; i < len; ++i)
    {
        out[i] = v[i] * s;
    }
    return out;
}

// scalar*vector
template <typename T>
std::vector<T> operator*(double s, const std::vector<T> &v)
{
    return v * s;
}

// vector+vector
template <typename T>
std::vector<T> operator+(const std::vector<T> &a, const std::vector<T> &b)
{
    size_t len { a.size() };
    if (len != b.size())
        throw std::runtime_error("Can't sum vectors of different lengths");
    std::vector<T> out(len);
    for (size_t i = 0; i < len; ++i)
    {
        out[i] = a[i] + b[i];
    }
    return out;
}

// scalar*matrix
template <typename T>
Matrix<T> operator*(double s, const Matrix<T> &m)
{
    return m * s;
}

Network::Network(const std::vector<unsigned int> &neuron_counts)
    : layers_ {neuron_counts.size()},
      layer_sizes_(neuron_counts),
      biases_(),
      weights_(),
      rand_engine_()
{
    std::random_device rd;
    rand_engine_.seed(rd());
    // initialize biases and weights
    biases_.resize(layers_);
    weights_.resize(layers_);
    std::normal_distribution<double> nd(0.0, 1.0); // mean 0, std dev 1
    // start at 1 because the first layer is inputs
    for (int l = 1; l < layers_; ++l)
    {
        weights_[l].resize(layer_sizes_[l-1], layer_sizes_[l]);
        biases_[l].resize(layer_sizes_[l]);
        // fill randomly
        for (int j = 0; j < layer_sizes_[l]; ++j)
        {
            biases_[l][j] = nd(rand_engine_);
            for (int k = 0; k < layer_sizes_[l-1]; ++k)
            {
                weights_[l].set(j, k, nd(rand_engine_));
            }
        }
    }
}

void Network::train(std::vector<Training_example> &training_examples,
    unsigned epochs,
    unsigned mini_batch_size,
    double eta,
    const std::vector<Training_example> &tests)
{
    size_t num_examples { training_examples.size() };
    if (mini_batch_size > num_examples)
        throw std::invalid_argument("Mini-batch size cannot be greater than \
            number of training examples");
    size_t num_tests { tests.size() };
    if (num_tests > 0)
    {
        size_t num_correct { test(tests) };
        std::cout << "Baseline (before training): "
            << num_correct << " / " << num_tests
            << " images identified correctly.\n";
    }
    for (unsigned i = 0; i < epochs; ++i)
    {
        // shuffle and split training data into mini-batches
        std::shuffle(training_examples.begin(), training_examples.end(),
            rand_engine_);
        std::vector<std::vector<Training_example>> mini_batches(num_examples /
            mini_batch_size);
        for (size_t mb = 0; mb * mini_batch_size < num_examples; ++mb)
        {
            // create new mini batch
            std::vector<Training_example> mini_batch(
                training_examples.begin() + mini_batch_size * mb,
                training_examples.begin() + mini_batch_size * (mb + 1));
            mini_batches[mb] = std::move(mini_batch);
        }
        // update weights and biases after each mini-batch
        for (auto &mb : mini_batches)
        {
            mini_batch_train(mb, eta);
        }
        // optionally test the network if tests were provided
        if (num_tests > 0)
        {
            size_t num_correct { test(tests) };
            std::cout << "Results from epoch " << i << ": "
                << num_correct << " / " << num_tests
                << " images identified correctly.\n";
        }
        std::cout << "Finished epoch " << i << ".\n";
    }
}

void Network::mini_batch_train(const std::vector<Training_example> &mini_batch,
    double eta)
{
    // create gradients for the biases and weights
    std::vector<std::vector<double>> b_gradient(layers_);
    std::vector<Matrix<double>> w_gradient(layers_);
    // start at 1 because the first layer is input neurons
    for (size_t l = 1; l < layers_; ++l)
    {
        b_gradient[l].resize(layer_sizes_[l]);
        w_gradient[l].resize(layer_sizes_[l - 1], layer_sizes_[l]);
    }
    // get cost gradient wrt biases and wrt weights
    for (const Training_example &x : mini_batch)
    {
        backprop(x, b_gradient, w_gradient);
    }
    // update weights with cost gradient
    size_t mb_size { mini_batch.size() };
    for (size_t l = 1; l < layers_; ++l)
    {
        // matrix subtraction
        weights_[l] -= (eta / mb_size)*w_gradient[l]; 
        // vector subtraction
        for (size_t j = 0; j < layer_sizes_[l]; ++j)
            biases_[l][j] -= (eta / mb_size)*b_gradient[l][j];
    }
    

}

void Network::backprop(const Training_example &x,
    std::vector<std::vector<double>> &bias_gradient,
    std::vector<Matrix<double>> &weight_gradient)
{
    // vector to store activation vectors of each layer
    std::vector<std::vector<double>> activations(layers_);
    // vector to store z vector (weighted sum) of each layer
    std::vector<std::vector<double>> zs(layers_);
    // initialize first activation vector with image data
    std::vector<double> a(x.img_data.begin(), x.img_data.end());
    // input pixels will have a value between 0.0 and 1.0
    // divide each element in activation vector by 255
    std::for_each(a.begin(), a.end(), [](double &d){ d /= 255; });
    activations[0] = a;
    // calculate the z vector for each layer
    std::vector<double> z {};
    // start at the second layer because the first layer is input
    for (size_t l = 1; l < layers_; ++l)
    {
        // dot product + vector addition
        z = dot_product(weights_[l], a) + biases_[l];
        zs[l] = z;
        a = sigmoid(z);
        activations[l] = a;
    }
    // compute error from the last layer of activations with
    // partial derivative of cost wrt activation * sigmoid_prime(z)
    // (dot product of two vectors)
    std::vector<double> error { dcost_wrt_a(activations.back(), x.label)
        * sigmoid_prime(zs.back()) };
    // partial derivative of C wrt bias is equal to the error
    // partial derivative of C wrt weight is the dot product of the
    // activation vector from the previous layer and the error vector
    bias_gradient.back() = bias_gradient.back() + error;
    weight_gradient.back() += outer_product(error, activations[layers_ - 2]);
    // transpose and dot product here ? ^^

    // repeat this process for all previous layers
    // back to front, starting at the second last layer (we just did the first
    // layer) and ending at the second layer (first layer is inputs)

    // the error vector in layer L = weight vector in L+1 dotted with error
    // vector in layer L+1 times sigmoid prime of z
    for (size_t l = layers_ - 2; l > 0; --l)
    {
        z = zs[l];
        std::vector<double> sp(sigmoid_prime(z));
        error = dot_product(weights_[l + 1].get_transposed(), error) * sp;
        bias_gradient[l] = bias_gradient[l] + error;
        weight_gradient[l] += outer_product(error, activations[l-1]);
    }
}

size_t Network::test(const std::vector<Training_example> &tests) const
{
    std::vector<double> activations;
    size_t correct { 0 };
    for (const Training_example &x : tests)
    {
        std::vector<double> a(x.img_data.begin(), x.img_data.end());
        // input pixels will have a value between 0.0 and 1.0
        // divide each element in activation vector by 255
        std::for_each(a.begin(), a.end(), [](double &d) { d /= 255; });
        activations = feedforward(a);
        // find most confident answer
        double highest_confidence { 0 };
        int answer { 0 };
        for (int i = 0; i < activations.size(); ++i)
        {
            double confidence { activations[i] };
            if (confidence > highest_confidence)
            {
                answer = i;
                highest_confidence = confidence;
            }
        }
        // verify answer
        if (answer == x.label)
            ++correct;
    }
    return correct;
}

std::vector<double> Network::feedforward(const std::vector<double> &in) const
{
    std::vector<double> activations(layers_);
    activations = in;
    for (size_t l = 1; l < layers_; ++l)
    {
        activations = sigmoid(dot_product(weights_[l], activations) 
            + biases_[l]);
    }
    return activations;
}

double Network::sigmoid(double z)
{
    return 1.0 / (1.0 + std::exp(-z));
}

std::vector<double> Network::sigmoid(const std::vector<double> &in)
{
    size_t len { in.size() };
    std::vector<double> out(len);
    // perform sigmoid function on every element
    for (size_t i = 0; i < len; ++i)
    {
        double z = in[i];
        out[i] = sigmoid(z);
    }
    return out;
}

double Network::sigmoid_prime(double z)
{
    return sigmoid(z) * (1 - sigmoid(z));
}

std::vector<double> Network::sigmoid_prime(const std::vector<double> &in)
{
    size_t len { in.size() };
    std::vector<double> out(len);
    // perform sigmoid function on every element
    for (size_t i = 0; i < len; ++i)
    {
        double z = in[i];
        out[i] = sigmoid_prime(z);
    }
    return out;
}

std::vector<double> Network::dcost_wrt_a(
    const std::vector<double> &actual,
    uint8_t expected_val)
{
    if (actual.size() != 10)
        throw std::runtime_error("Last layer should have 10 activations");
    // perform vector subtraction: actual-v, where v is a 0-filled
    // 10-dimensional vector. Only the entry who's index == expected_val is
    // set to one
    std::vector<double> da(10);
    for (uint8_t i = 0; i < 10; ++i)
    {
        if (i == expected_val)
        {
            da[i] = actual[i] - 1.0;
        }
        else
        {
            // v is 0 if the entry is not the expected value, so the result
            // is just actual
            da[i] = actual[i];
        }
    }
    return da;
}

// dot product between 2 vectors
double Network::dot_product(const std::vector<double> &a,
    const std::vector<double> &b)
{
    if (a.size() != b.size())
        throw std::runtime_error("Both vectors need to be the same size to \
            dot product");
    double sum { 0 };
    for (size_t i = 0; i < a.size(); ++i)
    {
        sum += (a[i] * b[i]);
    }
    return sum;
}

// dot product between matrix and vector
std::vector<double> Network::dot_product(const Matrix<double> &a, 
    const std::vector<double> &b)
{
    size_t w { a.width() };
    size_t h { a.height() };
    size_t len { b.size() };
    std::vector<double> out(h);
    if (w != len)
        throw std::runtime_error("Can't compute dot product between \
            matrix with unequal width to vector size");
    for (size_t i = 0; i < h; ++i)
    {
        double sum { 0.0 };
        for (size_t j = 0; j < w; ++j)
        {
            sum += a.get(i, j) * b[j];
        }
        out[i] = sum;
    }
    return out;
}

Matrix<double> Network::outer_product(const std::vector<double> &a, 
    const std::vector<double> &b)
{
    size_t w { b.size() }, h { a.size() };
    Matrix<double> out(w, h);
    for (size_t j = 0; j < h; ++j)
    {
        for (size_t k = 0; k < w; ++k)
        {
            out.set(j, k, a[j] * b[k]);
        }
    }
    return out;
}
