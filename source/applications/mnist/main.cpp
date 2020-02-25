#include <iostream>
#include <torch/torch.h>

#include "main.h"
#include "TorchNetwork.h"

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

template<typename DataLoader>
void train(size_t epoch, TorchNetwork &model, torch::Device &device, DataLoader &dataLoader,
		   torch::optim::Optimizer &optimizer, size_t datasetSize) {
	// After how many batches to log a new update with the loss value.
	const int64_t kLogInterval = 10;

	model.train();
	size_t batch_idx = 0;
	for (auto& batch : dataLoader) {
		auto data = batch.data.to(device);
		auto targets = batch.target.to(device);
		optimizer.zero_grad();
		auto output = model.forward(data);
		auto loss = torch::nll_loss(output, targets);
		AT_ASSERT(!std::isnan(loss.template item<float>()));
		loss.backward();
		optimizer.step();

		if (batch_idx++ % kLogInterval == 0) {
			std::printf("\rTrain Epoch: %ld [%5ld/%5ld] Loss: %.4f", epoch, batch_idx * batch.data.size(0),
					datasetSize,loss.template item<float>());
		}
	}
}

template<typename DataLoader>
void test(TorchNetwork &model, torch::Device &device, DataLoader &dataLoader, size_t datasetSize) {
	torch::NoGradGuard noGrad;
	model.eval();
	double test_loss = 0;
	int32_t correct = 0;
	for (const auto& batch : dataLoader) {
		auto data = batch.data.to(device);
		auto targets = batch.target.to(device);
		auto output = model.forward(data);
		test_loss += torch::nll_loss(output, targets, /*weight=*/{}, torch::Reduction::Sum).template item<float>();
		auto pred = output.argmax(1);
		correct += pred.eq(targets).sum().template item<int64_t>();
	}
	test_loss /= datasetSize;
	std::printf("\nTest set: Average loss: %.4f | Accuracy: %.3f\n", test_loss, static_cast<double>(correct) / datasetSize);
}

int main(int argc, char* argv[]) {
	const int64_t kTrainBatchSize = 64;
	const int64_t kTestBatchSize = 1000;
	const int64_t kNumberOfEpochs = 10;

	torch::manual_seed(1);

	cout << DATA_DIRECTORY << endl;

	auto train_dataset = torch::data::datasets::MNIST(DATA_DIRECTORY)
							.map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
							.map(torch::data::transforms::Stack<>());
	const size_t train_dataset_size = train_dataset.size().value();
	auto train_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(train_dataset), kTrainBatchSize);

	auto test_dataset = torch::data::datasets::MNIST(DATA_DIRECTORY)
							.map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
							.map(torch::data::transforms::Stack<>());
	const size_t test_dataset_size = test_dataset.size().value();
	auto test_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(test_dataset), kTestBatchSize);

	torch::Device device(IdentifyDeviceType());

	TorchNetwork model;
	model.to(device);

	torch::optim::SGD optimizer(model.parameters(), torch::optim::SGDOptions(0.01).momentum(0.5));

	for (int epoch = 1; epoch < kNumberOfEpochs; ++epoch) {
		train(epoch, model, device, *train_loader, optimizer, train_dataset_size);
		test(model, device, *test_loader, test_dataset_size);
	}

	return 0;
}
