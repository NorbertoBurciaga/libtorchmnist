#include <iostream>
#include <torch/torch.h>

#include "main.h"
#include "TorchImplementation.h"

using namespace std;


int main(int argc, char* argv[]) {
//	torch::Device device(IdentifyDeviceType());

	TorchImplementation *torchImp = new TorchImplementation();

	cout << DATA_DIRECTORY << endl;

	torch::Tensor tensor = torch::rand({2, 3});
	cout << tensor << endl;

	delete torchImp;
	return 0;
}
