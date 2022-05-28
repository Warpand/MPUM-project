#include <iostream>
#include <algorithm>
#include <cmath>

#include "Data.h"

class NearestNeighbours {
	double** train_x;
	unsigned* train_y;
	unsigned train_size;
	unsigned k;
	double (*d)(double*, double*, unsigned);

	double check_prediction(double** x, unsigned* y, unsigned size) {
		double guessed = 0.0;
		std::pair<double, unsigned>* dbuffer = new std::pair<double, unsigned>[train_size];
		unsigned class_counters[CLASS_NR + 1]{};
		for (unsigned i = 0; i < size; i++) {
			// calculate the distances
			for (unsigned j = 0; j < train_size; j++)
				dbuffer[j] = { d(train_x[j], x[i], FEATURE_NR), y[j] };
			// sort them
			std::sort(dbuffer, dbuffer + size);
			// count the classes of k nearest neighbours
			for (unsigned z = 0; z < k; z++)
				class_counters[dbuffer[z].second]++;
			// choose the best class
			unsigned max_count = 0, res = 0;
			for (unsigned z = 1; z <= CLASS_NR; z++) {
				if (class_counters[z] > max_count) {
					max_count = class_counters[z];
					res = z;
				}
				class_counters[z] = 0; // clean the counters along the way
			}
			if (res == y[i])
				guessed += 1;
		}
		delete[]dbuffer;
		return guessed * 100.0 / static_cast<double>(size);
	}

public:
	NearestNeighbours(set_holder& sets, double(*d)(double*, double*, unsigned), unsigned train_size = TRAIN_SIZE) :
		train_x(sets.train_x), train_y(sets.train_y), train_size(train_size),
		k(1), d(d) {}

	unsigned get_k() {
		return k;
	}

	void find_best_k(double** val_x, unsigned* val_y, unsigned val_size = VAL_SIZE) {
		unsigned best_k = 1;
		double best_res = 0.0;
		for (unsigned z = 1; z * z <= train_size; z += 2) {
			k = z;
			double res = check_prediction(val_x, val_y, val_size);
			if (res > best_res) {
				best_res = res;
				best_k = k;
			}
		}
		k = best_k;
	}

	double evaluate(double** test_x, unsigned* test_y, unsigned test_size = TEST_SIZE) {
		return check_prediction(test_x, test_y, test_size);
	}
};

double euclidian(double* x1, double* x2, unsigned len) {
	double res = 0.0;
	for (unsigned i = 0; i < len; i++)
		res += (x1[i] - x2[i]) * (x1[i] - x2[i]);
	return sqrt(res);
}

int main() {
	set_holder data;
	try {
		data = prepare_data();
	}
	catch (const std::ios_base::failure& er) {
		std::cout << er.what() << '\n';
		return 1;
	}
	NearestNeighbours nn(data, euclidian);
	nn.find_best_k(data.val_x, data.val_y);
	std::cout << nn.get_k() << '\n';
	std::cout << nn.evaluate(data.test_x, data.test_y) << '\n';
	delete[]data.train_x;
	delete[]data.val_x;
	delete[]data.test_x;
	delete[]data.train_y;
	delete[]data.val_y;
	delete[]data.test_y;
}