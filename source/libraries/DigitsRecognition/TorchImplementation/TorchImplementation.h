#ifndef SOURCE_LIBRARIES_DIGITSRECOGNITION_TORCHIMPLEMENTATION_TORCHIMPLEMENTATION_H_
#define SOURCE_LIBRARIES_DIGITSRECOGNITION_TORCHIMPLEMENTATION_TORCHIMPLEMENTATION_H_
#include <iostream>
#include <torch/torch.h>

#include "TorchNetwork.h"

using namespace std;

class TorchImplementation {
public:
	TorchImplementation();
	virtual ~TorchImplementation();
};

#endif /* SOURCE_LIBRARIES_DIGITSRECOGNITION_TORCHIMPLEMENTATION_TORCHIMPLEMENTATION_H_ */
