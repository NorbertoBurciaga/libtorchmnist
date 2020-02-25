#include "TorchImplementation.h"

TorchImplementation::TorchImplementation(torch::Device & device, TorchNetwork& model) : device(device), model(model) {
}

TorchImplementation::~TorchImplementation() {
}

