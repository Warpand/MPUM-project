#include <iostream>
#include <cmath>
#include <vector>
#include <random>
#include <iomanip>

#include "Data.h"

class sigmoid {
public:
	double operator () (double x) {
		return 1.0 / (1.0 + exp(-x));
	}

	double derivative(double x) {
		return x * (1.0 - x);
	}
};

template<typename Activation_Func, unsigned dim = FEATURE_NR>
class NeuralNetwork {
	struct node {
		double* weights;
		const unsigned size;
		double output;
		double delta;
		double dot_product(double* x) {
			double result = 0.0;
			for (unsigned i = 0; i < size; i++)
				result += weights[i] * x[i];
			return result;
		}
	};
	double** train_x;
	unsigned* train_y;
	const unsigned input_size;
	Activation_Func func;
	std::vector<std::vector<node>> network;

	std::vector<node> create_layer(unsigned in_size, unsigned out_size, std::mt19937_64& gen) {
		std::vector<node> layer;
		layer.reserve(out_size);
		for (unsigned i = 0; i < out_size; i++) {
			double* weights = new double[dim];
			std::uniform_real_distribution<double> dist(0.0, 1.0);
			for (unsigned j = 0; j < in_size; j++)
				weights[j] = dist(gen);
			layer.push_back({ weights, in_size });
		}
		return layer;
	}

	double* forward_pass(double* x) {
		double* x_in = x;
		for (auto& layer : network) {
			double* x_out = new double[layer.size()];
			for (unsigned i = 0; i < layer.size(); i++)
				layer[i].output = x_out[i] = func(layer[i].dot_product(x_in));
			if (x_in != x)
				delete[]x_in;
			x_in = x_out;
		}
		return x_in;
		// remember to free the memory under the return value !
	}

	void backward_pass(double y_hot[]) {
		for (unsigned j = 0; j < network.front().size(); j++) {
			node& node = network.front()[j];
			node.delta = (node.output - y_hot[j]) * func.derivative(node.output);
		}
		for (int i = static_cast<int>(network.size()) - 2; i >= 0; i--) {
			for (unsigned j = 0; j < network[i].size(); j++) {
				double err = 0.0;
				for (node& n : network[i + 1])
					err += n.weights[j] * n.delta;
				network[i][j].delta = err * func.derivative(network[i][j].output);
			}
		}
	}

	void update_weights(double* x, double eta) {
		for (node& n : network[0])
			for (unsigned j = 0; j < dim; j++)
				n.weights[j] -= eta * n.delta * x[j];
		for (unsigned i = 1; i < network.size(); i++) {
			for (node& n : network[i]) {
				for (unsigned j = 0; j < network[i - 1].size(); j++)
					n.weights[j] -= eta * n.delta * network[i - 1][j].output;
			}
		}
	}

public:
	NeuralNetwork(double** x, unsigned* y, const unsigned size, std::vector<unsigned>& layers_sizes) : 
		train_x(x), train_y(y), input_size(size) {
		std::random_device rd;
		std::mt19937_64 gen(rd());
		network.push_back(create_layer(dim, layers_sizes[0], gen));
		for (unsigned i = 1; i < layers_sizes.size(); i++)
			network.push_back(create_layer(network.front().size(), layers_sizes[i], gen));
		network.push_back(create_layer(network.front().size(), CLASS_NR, gen));
	}

	~NeuralNetwork() {
		for (unsigned i = 0; i < network.size(); i++)
			for (auto& n : network[i])
				delete[]n.weights;
	}

	void fit(double eta = 0.5, unsigned iter_nr = 500) {
		for (unsigned i = 0; i < iter_nr; i++) {
			for (unsigned j = 0; j < input_size; j++) {
				double* to_delete = forward_pass(train_x[j]);
				delete[]to_delete;
				double y_hot[CLASS_NR]{};
				y_hot[train_y[j]] = 1.0;
				backward_pass(y_hot);
				update_weights(train_x[j], eta);
			}
		}
	}
	
	double check(double** test_x, unsigned* test_y, unsigned size = TEST_SIZE) {
		int guessed = 0;
		for (unsigned i = 0; i < size; i++) {
			double* probs = forward_pass(test_x[i]);
			unsigned arg_max = 0;
			for (unsigned j = 1; j < CLASS_NR; j++)
				arg_max = (probs[j] > probs[arg_max]) ? j : arg_max;
			if (arg_max == test_y[i])
				guessed += 1;
			/*for (unsigned k = 0; k < CLASS_NR; k++)
				std::cout << probs[k] << ' ';
			std::cout << '\n';*/
			delete[]probs;
		}
		return static_cast<double>(guessed) * 100.0 / static_cast<double>(size);
	}
};

int main() {
	set_holder data = prepare_data();
	data.standardize();
	std::vector<unsigned> layers{ 5 };
	NeuralNetwork<sigmoid> nn(data.train_x, data.train_y, TRAIN_SIZE, layers);
	nn.fit(0.1, 50);
	std::cout << "Fitting done" << std::endl;
	std::cout << "Accuracy: " << nn.check(data.val_x, data.val_y) << std::endl;
	data.clear();
}