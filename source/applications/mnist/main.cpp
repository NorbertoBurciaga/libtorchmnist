#include <iostream>
#include <torch/torch.h>

#include "main.h"
#include "TorchImplementation.h"

using namespace std;


int main(int argc, char* argv[]) {

	auto train_dataset = torch::data::datasets::MNIST(DATA_DIRECTORY)
							.map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
							.map(torch::data::transforms::Stack<>());

	auto test_dataset = torch::data::datasets::MNIST(DATA_DIRECTORY)
							.map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
							.map(torch::data::transforms::Stack<>());

	TorchImplementation *torchImp = new TorchImplementation();

	cout << DATA_DIRECTORY << endl;

	torch::Tensor tensor = torch::rand({2, 3});
	cout << tensor << endl;

	delete torchImp;
	return 0;
}
