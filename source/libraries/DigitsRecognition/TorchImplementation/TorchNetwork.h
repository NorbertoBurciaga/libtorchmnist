#ifndef SOURCE_LIBRARIES_DIGITSRECOGNITION_TORCHIMPLEMENTATION_TORCHNETWORK_H_
#define SOURCE_LIBRARIES_DIGITSRECOGNITION_TORCHIMPLEMENTATION_TORCHNETWORK_H_

#include <vector>
#include <torch/torch.h>

struct TorchNetwork : torch::nn::Module {
	torch::nn::Conv2d conv1;
	torch::nn::Conv2d conv2;
	torch::nn::Dropout2d conv2_drop;
	torch::nn::Linear fc1;
	torch::nn::Linear fc2;
//public:
	TorchNetwork();
/*	void to(torch::Device device, bool non_blocking = false);
	void train(bool on = true);
	void eval();
	std::vector<torch::Tensor> parameters(bool recurse = true) const;*/
	torch::Tensor forward(torch::Tensor x);
	virtual ~TorchNetwork();
};

#endif /* SOURCE_LIBRARIES_DIGITSRECOGNITION_TORCHIMPLEMENTATION_TORCHNETWORK_H_ */
