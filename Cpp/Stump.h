#ifndef STUMP_H
#define STUMP_H

#include <algorithm>
#include <limits>
#include "Data.h"

class stump {
	double threshold;
	unsigned feature;
	unsigned left;
	unsigned right;
public:
	double fit(double** x, unsigned* y, double* weights, unsigned size = TRAIN_SIZE) {
		std::pair<double, unsigned>* feat_buffer = new std::pair<double, unsigned>[size];
		unsigned left_counter[CLASS_NR]{}, right_counter[CLASS_NR]{};
		double error = std::numeric_limits<double>::max();
		for (unsigned k = 0; k < FEATURE_NR; k++) {
			// fill the buffer with feature values
			for (unsigned i = 0; i < size; i++)
				feat_buffer[i] = { x[i][k], i };
			std::sort(feat_buffer, feat_buffer + size);
			// first count all the classes and place them in the right counter
			for (unsigned i = 0; i < size; i++)
				right_counter[y[feat_buffer[i].second]]++;
			unsigned old_left = CLASS_NR, old_right = CLASS_NR;
			double old_error = 0.0;
			// j is the threshold index
			for (unsigned j = 0; j < size; j++) {
				// move one class from right counter to left counter
				left_counter[y[feat_buffer[j].second]]++;
				right_counter[y[feat_buffer[j].second]]--;
				unsigned count = 0, left_class = 0, right_class = 0;
				// choose most common class for both sides
				for (unsigned z = 0; z < CLASS_NR; z++) {
					if (left_counter[z] > count) {
						count = left_counter[z];
						left_class = z;
					}
				}
				count = 0;
				for (unsigned z = 0; z < CLASS_NR; z++) {
					if (right_counter[z] > count) {
						count = right_counter[z];
						right_class = z;
					}
				}
				// calculate the error
				// a heuristic is used: if chosen classes for left and right side don't change, 
				// only the prediction on threshold changes (else branch)
				double new_error = 0;
				if (left_class != old_left || right_class != old_right) {
					for (unsigned z = 0; z <= j; z++)
						if (y[feat_buffer[z].second] != left_class)
							new_error += weights[z];
					for (unsigned z = j + 1; z < size; z++)
						if (y[feat_buffer[z].second] != right_class)
							new_error += weights[z];
					old_left = left_class;
					old_right = right_class;
				}
				else {
					new_error = old_error;
					unsigned c = y[feat_buffer[j].second];
					if (c == left_class && c != right_class)
						new_error -= weights[j];
					else if (c != left_class && c == right_class)
						new_error += weights[j];
				}
				old_error = new_error;
				if (new_error < error) {
					error = new_error;
					threshold = feat_buffer[j].first;
					feature = k;
					left = left_class;
					right = right_class;
				}
			}
			for (unsigned i = 0; i < CLASS_NR; i++)
				left_counter[i] = right_counter[i] = 0;
		}
		delete[]feat_buffer;
		return error;
	}

	unsigned predict(double* obs) {
		return (obs[feature] <= threshold) ? left : right;
	}
};

#endif // STUMP_H
