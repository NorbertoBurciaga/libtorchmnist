#include <iostream>
#include <torch/torch.h>

#include "main.h"

using namespace std;

torch::Device IdentifyDeviceType() {
	if (torch::cuda::is_available()) {
	    cout << "CUDA available! Training on GPU." << endl;
	    return torch::kCUDA;
	  } else {
	    cout << "Training on CPU." << endl;
	    return torch::kCPU;
	  }
}

int main(int argc, char* argv[]) {
	torch::Device device(IdentifyDeviceType());

	cout << DATA_DIRECTORY << endl;

	torch::Tensor tensor = torch::rand({2, 3});
	cout << tensor << endl;
	return 0;
}
