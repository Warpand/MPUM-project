#include "Data.h"
#include <cmath>
#include <iostream>
#include <future>
#include <algorithm>
using namespace std;

class Node
{
    friend class DecisionTree;

    Node* left=nullptr;
    Node* right=nullptr;
    int feauture;
    double val;
    unsigned pred_class;

    Node(int feauture=0, double val=0.0, Node* left = nullptr, Node* right = nullptr)
    {
        this->feauture = feauture;
        this->val = val;
        this->left = left;
        this->right = right;
    }

    Node(unsigned pred_class)
    {
        this->pred_class=pred_class;
    }

    bool is_leaf()
    {
        return (left==nullptr && right==nullptr);
    }
};

class DecisionTree
{
    Node* root;
    int max_depth;
    int features;
    string impurity;
    //epsilon, minsamples split

    void del(Node* cur_node)
    {
        if(cur_node->left!=nullptr)
        {
            del(cur_node->left);
        }
        if(cur_node->right!=nullptr)
        {
            del(cur_node->right);
        }
        delete cur_node;
    }

    Node* build_tree(int depth, double** cur_x, unsigned* cur_y, int cur_size)
    {
        cout<<"Depth "<<depth<<endl;
        if(depth>=max_depth)
        {
            int classes[8]={0};
            for(int i=0;i<cur_size;i++)
            {
                classes[cur_y[i]]++;
            }
            unsigned index=0;
            int max_val=classes[0];
            for(int i=1;i<8;i++)
            {
                if(max_val<classes[i])
                {
                    max_val = classes[i];
                    index = i;
                }
            }
            Node* new_node = new Node(index);
            return new_node;
        }

        double best_score=-1;
        int best_feature;
        double best_val;
        pair<double, unsigned>* buf = new pair<double, unsigned>[cur_size];
        for(int i=0;i<features;i++)
        {
            for(int j=0;j<cur_size-1;j++)
            {
                double val = (cur_x[j][i] + cur_x[j+1][i])/2;
                int classes_below[8]={0};
                int classes_above[8]={0};
                int size_above=0;
                int size_below=0;
                for(int ii=0;ii<cur_size;ii++)
                {
                    if(cur_x[ii][i]>val)
                    {
                        classes_above[cur_y[ii]]++;
                        size_above++;
                    }
                    else
                    {
                        classes_below[cur_y[ii]]++;
                        size_below++;
                    }
                }
                if(impurity=="entropy")
                {
                    double entropy=0;
                    double entropy_above=0;
                    double entropy_below=0;
                    if (size_above!=0 && size_below!=0)
                    {
                        for(int ii=0; ii<8;ii++)
                        {
                            double prob = static_cast<double>(classes_below[ii]+classes_above[ii])/static_cast<double>(cur_size);
                            double vall = (prob>0) ? prob*log2(prob) : 0;
                            entropy -= vall;
                            prob = static_cast<double>(classes_below[ii])/static_cast<double>(size_below);
                            vall = (prob>0) ? prob*log2(prob) : 0;
                            entropy_below -= vall;
                            prob = static_cast<double>(classes_above[ii])/static_cast<double>(size_above);
                            vall = (prob>0) ? prob*log2(prob) : 0;
                            entropy_above -= vall;

                        }
                        entropy = entropy - (static_cast<double>(size_below)/static_cast<double>(cur_size)*entropy_below + 
                                                                static_cast<double>(size_above)/static_cast<double>(cur_size)*entropy_above);
                    }

                    if(best_score==-1 || entropy>best_score)
                    {
                        best_score = entropy;
                        best_feature = i;
                        best_val = val;
                    }
                }
                else if(impurity=="gini")
                {
                    double gini = 0;
                    double gini_above=0;
                    double gini_below=0;
                    for(int ii=0; ii<8;ii++)
                    {
                        double prob = static_cast<double>(classes_below[ii])/static_cast<double>(size_below);
                        gini_below += prob*(1-prob);
                        prob = static_cast<double>(classes_above[ii])/static_cast<double>(size_above);
                        gini_above += prob*(1-prob);

                    }

                    gini = gini_below*static_cast<double>(size_below)/static_cast<double>(cur_size) + gini_above*static_cast<double>(size_above)/static_cast<double>(cur_size);

                    if (best_score==-1 || gini < best_score)
                    {
                        best_score = gini;
                        best_feature = i;
                        best_val = val;
                    }
                    
                }
            }
        }

        delete [] buf;

        double* x_below[cur_size];
        unsigned y_below[cur_size];
        double* x_above[cur_size];
        unsigned y_above[cur_size];
        int size_above=0;
        int size_below=0;
        for(;size_above+size_below<cur_size;)
        {
            int ii = size_above+size_below;
            if(cur_x[ii][best_feature]>best_val)
            {
                x_above[size_above]=cur_x[ii];
                y_above[size_above]=cur_y[ii];
                size_above++;
            }
            else
            {
                x_below[size_below]=cur_x[ii];
                y_below[size_below]=cur_y[ii];
                size_below++;
            }
        }

        auto a1 = async(launch::async, [this, depth, &x_below, &y_below, size_below](){return build_tree(depth+1, x_below, y_below, size_below);});
        auto a2 = async(launch::async, [this, depth, &x_above, &y_above, size_above](){return build_tree(depth+1, x_above, y_above, size_above);});
        Node* left = a1.get();
        Node* right = a2.get();

        //thread a1([left, this, depth, &x_below, &y_below, size_below](){left = build_tree(depth+1, x_below, y_below, size_below)});
        //thread a2([right, this, depth, &x_above, &y_above, size_above](){ right = build_tree(depth+1, x_above, y_above, size_above)});

        Node* cur = new Node(best_feature, best_val, left, right);

        return cur;
    }

    unsigned traverse_tree(double* x, Node* cur_node)
    {
        if(cur_node->is_leaf())
        {
            return cur_node->pred_class;
        }

        if(x[cur_node->feauture]<=cur_node->val)
        {
            return traverse_tree(x, cur_node->left);
        }
        return traverse_tree(x, cur_node->right);
    }

    public:
    DecisionTree(int max_depth, int features, string impurity="entropy")
    {
        this->max_depth=max_depth;
        this->features=features;
        this->impurity = impurity;
    }

    void fit(double** train_x, unsigned* train_y, int size=TRAIN_SIZE)
    {
        root = build_tree(0, train_x, train_y, size);
    }

    double check(double** test_x, unsigned* test_y, int size = TEST_SIZE) 
    {
		int guessed = 0;
		for (unsigned i = 0; i < size; i++) 
        {
			unsigned pred_val = traverse_tree(test_x[i], root);
			if (pred_val == test_y[i])
				guessed += 1;
		}
		return static_cast<double>(guessed) * 100.0 / static_cast<double>(size);
    }

    void clear()
    {
        del(root);
    }
    ~DecisionTree()
    {
        del(root);
    }

};

int main()
{
    set_holder data = prepare_data();
	data.standardize();
    DecisionTree dt = DecisionTree(10, FEATURE_NR, "entropy");
    dt.fit(data.train_x, data.train_y);
    cout<<"Accuracy "<<dt.check(data.val_x, data.val_y)<<endl;
}