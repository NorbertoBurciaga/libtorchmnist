#include "TorchImplementation.h"

torch::Device TorchImplementation::IdentifyDeviceType() {
	if (torch::cuda::is_available()) {
		cout << "CUDA available! Training on GPU." << endl;
		return torch::kCUDA;
	} else {
		cout << "Training on CPU." << endl;
		return torch::kCPU;
	}
}

TorchImplementation::TorchImplementation() {
	this->device = new torch::Device(this->IdentifyDeviceType());

}

TorchImplementation::~TorchImplementation() {
	delete this->device;
}

