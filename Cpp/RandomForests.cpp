#include "Data.h"
#include "DecisionTree.cpp"
#include <cmath>
#include <iostream>
#include <random>
#include <algorithm>

using namespace std;

class RandomForests
{
    DecisionTree* trees;
    int B;
    int max_depth;
    int tab[FEATURE_NR];
    public:
    RandomForests(int B, int max_depth)
    {
        this->B = B;
        this->max_depth = max_depth;
        trees = new DecisionTree[B];
    }

    void fit(double** train_x, unsigned* train_y, int size=TRAIN_SIZE, string impurity="entropy")
    {
        srand(time(NULL));
        random_device rd;
        mt19937_64 gen(rd());
        uniform_int_distribution<> distr(0, size-1);
        double** cur_x = new double* [size];
        unsigned* cur_y = new unsigned [size];

        for(int i=0;i<B;i++)
        {
            cout<<"Tree "<<i<<endl;
            for(int j=0;j<size;j++)
            {
                int index = distr(gen);
                cur_y[j] = train_y[index];
                cur_x[j] = train_x[index];
            }

            trees[i] = DecisionTree(max_depth, FEATURE_NR, impurity, true);
            trees[i].fit(cur_x, cur_y);
        }

        delete [] cur_y;
        delete [] cur_x;

    }

    double check(double** test_x, unsigned* test_y, int size = TEST_SIZE)
    {
        int guessed = 0;
		for (unsigned i = 0; i < size; i++) 
        {
            int classes[8]={};
            for(int j=0;j<B;j++)
            {
                unsigned pred_val = trees[j].traverse_tree(test_x[i], trees[j].root);
                classes[pred_val]++;
            }
			
            unsigned arg_max = 0;
			for (unsigned j = 1; j < CLASS_NR; j++)
				arg_max = (classes[j] > classes[arg_max]) ? j : arg_max;

            if (arg_max == test_y[i])
				guessed += 1;
		}
		return static_cast<double>(guessed) * 100.0 / static_cast<double>(size);
    }

    ~RandomForests()
    {
        delete [] trees;
    }
};

int main()
{
    set_holder data = prepare_data();
	data.standardize();
    RandomForests rf = RandomForests(100, 30);
    rf.fit(data.train_x, data.train_y);
    cout<<"Accuracy "<<rf.check(data.test_x, data.test_y)<<endl;
    data.clear();
}