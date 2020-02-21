#include <iostream>
#include <torch/torch.h>

#include "main.h"

using namespace std;

int main(int argc, char* argv[]) {
	cout << DATA_DIRECTORY << endl;
	torch::Tensor tensor = torch::rand({2, 3});
	cout << tensor << endl;
	return 0;
}
