#include "TorchNetwork.h"

TorchNetwork::TorchNetwork() :  conv1(torch::nn::Conv2dOptions(1, 10, /*kernel_size=*/5)),
								conv2(torch::nn::Conv2dOptions(10, 20, /*kernel_size=*/5)),
								fc1(320, 50),
								fc2(50, 10) {
	register_module("conv1", conv1);
	register_module("conv2", conv2);
	register_module("conv2_drop", conv2_drop);
	register_module("fc1", fc1);
	register_module("fc2", fc2);
}

/*void TorchNetwork::to(torch::Device device, bool non_blocking) {
	torch::nn::Module::to(device, non_blocking);
}

void TorchNetwork::train(bool on) {
	torch::nn::Module::train(on);
}

void TorchNetwork::eval() {
	torch::nn::Module::eval();
}

std::vector<torch::Tensor> TorchNetwork::parameters(bool recurse) const {
	torch::nn::Module::parameters(recurse);
}*/


torch::Tensor TorchNetwork::forward(torch::Tensor x) {
	x = torch::relu(torch::max_pool2d(conv1->forward(x),2));
	x = torch::relu(torch::max_pool2d(conv2_drop->forward(conv2->forward(x)), 2));
	x = x.view({-1,320});
	x = torch::relu(fc1->forward(x));
	x = torch::dropout(x, /*p=*/0.5, /*training=*/is_training());
	x = fc2->forward(x);
	return torch::log_softmax(x, /*dim=*/1);
}

TorchNetwork::~TorchNetwork() {
}
