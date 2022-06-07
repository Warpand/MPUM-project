#include <iostream>
#include <cmath>
#include <vector>
#include <limits>
#include <exception>

#include "Stump.h"
#include "Data.h"

class AdaBoost {
	double* alphas = nullptr;
	std::vector<stump> h;

	unsigned predict(double* obs) {
		double max_val = std::numeric_limits<double>::min();
		unsigned prediction = CLASS_NR;
		for (unsigned p = 0; p < CLASS_NR; p++) {
			double val = 0.0;
			for (unsigned i = 0; i < h.size(); i++) 
				val += (h[i].predict(obs) == p) ? alphas[i] : 0.0;
			if (val > max_val) {
				max_val = val;
				prediction = p;
			}
		}
		return prediction;
	}
public:
	~AdaBoost() {
		delete[]alphas;
	}

	void fit(double** x, unsigned* y, unsigned iter_nr = 1000, unsigned size = TRAIN_SIZE) {
		h.reserve(iter_nr);
		double* weights = new double[size];
		alphas = new double[iter_nr];
		double n = static_cast<double>(size);
		for (unsigned i = 0; i < size; i++)
			weights[i] = 1.0 / n;
		for (unsigned t = 0; t < iter_nr; t++) {
			stump st;
			double err = st.fit(x, y, weights, size);
			double alpha = log((1.0 - err) / err) + log(static_cast<double>(CLASS_NR) - 1.0);
			alphas[t] = alpha;
			for (unsigned i = 0; i < size; i++)
				weights[i] *= exp(alpha * (st.predict(x[i]) != y[i]) ? 1.0 : -1.0);
			double sum = 0.0;
			for (unsigned i = 0; i < size; i++)
				sum += weights[i];
			for (unsigned i = 0; i < size; i++)
				weights[i] /= sum;
			h.push_back(st);
		}
		delete[]weights;
	}

	double check(double** x, unsigned* y, unsigned size = TEST_SIZE) {
		if (h.empty())
			throw std::runtime_error("Error: check called before fitting the model.");
		int correct = 0;
		for (unsigned i = 0; i < size; i++)
			if (predict(x[i]) == y[i])
				correct++;
		return static_cast<double>(correct) * 100.0 / static_cast<double>(size);
	}
};

int main() {
	set_holder data = prepare_data();
	AdaBoost ad;
	try {
		ad.fit(data.train_x, data.train_y);
		std::cout << ad.check(data.test_x, data.test_y) << std::endl;
	}
	catch(const std::runtime_error& er) {
		std::cerr << er.what() << std::endl;
	}
	data.clear();
}