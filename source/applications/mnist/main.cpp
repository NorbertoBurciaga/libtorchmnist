#include <iostream>
#include <torch/torch.h>

#include "main.h"
#include "TorchNetwork.h"
#include "TorchImplementation.h"

using namespace std;

torch::DeviceType IdentifyDeviceType() {
	if (torch::cuda::is_available()) {
		cout << "CUDA available! Training on GPU." << endl;
		return torch::kCUDA;
	} else {
		cout << "Training on CPU." << endl;
		return torch::kCPU;
	}
}

int main(int argc, char* argv[]) {
	torch::manual_seed(1);

	cout << DATA_DIRECTORY << endl;

	auto train_dataset = torch::data::datasets::MNIST(DATA_DIRECTORY)
							.map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
							.map(torch::data::transforms::Stack<>());

	auto test_dataset = torch::data::datasets::MNIST(DATA_DIRECTORY)
							.map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
							.map(torch::data::transforms::Stack<>());

	torch::Device device(IdentifyDeviceType());

	TorchNetwork model;
	model.to(device);

	TorchImplementation *torchImp = new TorchImplementation(device, model);

	torch::Tensor tensor = torch::rand({2, 3});
	cout << tensor << endl;

	delete torchImp;
	return 0;
}
