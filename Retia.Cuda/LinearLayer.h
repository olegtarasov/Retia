#pragma once

#include <memory>
#include "LayerBase.h"
#include "NeuroWeight.h"

class LinearLayer : public LayerBase
{
public:
	LinearLayer(int inputSize, int outSize, int batchSize, int seqLen);


	void TransferStatesToDevice(std::vector<WeightSyncContainer*>& states) override;
	void TransferStatesToHost(std::vector<WeightSyncContainer*>& states) override;
	void ForwardSequence(DeviceMatrix& input) override;
	void BackpropSequence(DeviceMatrix& input, DeviceMatrix& outSens) override;
	void Optimize(OptimizerBase& optimizer) override;
	void ResetMemory() override;
	void ResetOptimizerCache() override;
private:
	std::unique_ptr<NeuroWeight> _w, _b;
	std::unique_ptr<DeviceMatrix> _identity;
};
