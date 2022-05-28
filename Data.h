#ifndef DATA_H
#define DATA_H

#include <fstream>
#include <cmath>
#include <limits>

constexpr unsigned TRAIN_SIZE = 6400;
constexpr unsigned VAL_SIZE = 800;
constexpr unsigned TEST_SIZE = 800;
constexpr unsigned FEATURE_NR = 518;
constexpr unsigned CLASS_NR = 8;

struct set_holder {
	double** train_x;
	double** val_x;
	double** test_x;
	unsigned* train_y;
	unsigned* val_y;
	unsigned* test_y;

	void standardize() {
		double* means = new double[FEATURE_NR], *stds = new double[FEATURE_NR];
		for (unsigned i = 0; i < FEATURE_NR; i++)
			means[i] = stds[i] = 0.0;
		for (unsigned i = 0; i < TRAIN_SIZE; i++) {
			for (unsigned j = 0; j < FEATURE_NR; j++) {
				means[j] += train_x[i][j];
				stds[j] += (train_x[i][j] * train_x[i][j]);
			}
		}
		for (unsigned j = 0; j < FEATURE_NR; j++)
			means[j] /= static_cast<double>(TRAIN_SIZE);
		for (unsigned j = 0; j < FEATURE_NR; j++)
			stds[j] = sqrt(stds[j] / static_cast<double>(TRAIN_SIZE));
		for (unsigned i = 0; i < TRAIN_SIZE; i++)
			for (unsigned j = 0; j < FEATURE_NR; j++)
				train_x[i][j] = (train_x[i][j] - means[j]) / stds[j];
		for (unsigned i = 0; i < VAL_SIZE; i++)
			for (unsigned j = 0; j < FEATURE_NR; j++)
				val_x[i][j] = (val_x[i][j] - means[j]) / stds[j];
		for (unsigned i = 0; i < TEST_SIZE; i++)
			for (unsigned j = 0; j < FEATURE_NR; j++)
				test_x[i][j] = (test_x[i][j] - means[j]) / stds[j];
		delete[]means;
		delete[]stds;
	}

	void normalize() {
		double* mins = new double[FEATURE_NR], * maxs = new double[FEATURE_NR];
		for (unsigned i = 0; i < FEATURE_NR; i++)
			mins[i] = std::numeric_limits<double>::max();
		for (unsigned i = 0; i < FEATURE_NR; i++)
			maxs[i] = std::numeric_limits<double>::min();
		for (unsigned i = 0; i < TRAIN_SIZE; i++) {
			for (unsigned j = 0; j < FEATURE_NR; j++) {
				mins[j] = std::min(mins[j], train_x[i][j]);
				maxs[j] = std::max(maxs[j], train_x[i][j]);
			}
		}
		for (unsigned i = 0; i < TRAIN_SIZE; i++)
			for (unsigned j = 0; j < FEATURE_NR; j++)
				train_x[i][j] = (train_x[i][j] - mins[j]) / (maxs[j] - mins[j]);
		for (unsigned i = 0; i < VAL_SIZE; i++)
			for (unsigned j = 0; j < FEATURE_NR; j++)
				val_x[i][j] = (val_x[i][j] - mins[j]) / (maxs[j] - mins[j]);
		for (unsigned i = 0; i < TEST_SIZE; i++)
			for (unsigned j = 0; j < FEATURE_NR; j++)
				train_x[i][j] = (train_x[i][j] - mins[j]) / (maxs[j] - mins[j]);
		delete[]mins;
		delete[]maxs;
	}
};

static const char* TRAIN_FILE = "train.data";
static const char* VAL_FILE = "val.data";
static const char* TEST_FILE = "test.data";

inline void init_pointers(double** set, unsigned size) {
	for (unsigned i = 0; i < size; i++)
		set[i] = new double[FEATURE_NR];
}

void fill_data(double** x, unsigned* y, unsigned size, const char* filename) {
	std::fstream input;
	input.open(filename, std::fstream::in);
	if (!input.is_open())
		throw std::ios_base::failure("File opening failed. Check if appropriate files are in the directory.\nError");
	for (unsigned i = 0; i < size; i++) {
		for (unsigned j = 0; j < FEATURE_NR; j++)
			input >> x[i][j];
		input >> y[i];
	}
	input.close();
}

set_holder prepare_data() {
	set_holder data{
		new double* [TRAIN_SIZE],
		new double* [VAL_SIZE],
		new double* [TEST_SIZE],
		new unsigned[TRAIN_SIZE],
		new unsigned[VAL_SIZE],
		new unsigned[TEST_SIZE]
	};
	init_pointers(data.train_x, TRAIN_SIZE);
	init_pointers(data.val_x, VAL_SIZE);
	init_pointers(data.test_x, VAL_SIZE);
	fill_data(data.train_x, data.train_y, TRAIN_SIZE, TRAIN_FILE);
	fill_data(data.val_x, data.val_y, VAL_SIZE, VAL_FILE);
	fill_data(data.test_x, data.test_y, TEST_SIZE, TEST_FILE);
	return data;
}

#endif // DATA_H