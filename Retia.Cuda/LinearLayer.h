#pragma once

#include <memory>
#include "LayerBase.h"
#include "NeuroWeight.h"
#include "RawMatrixPtr.h"

class LinearLayer : public LayerBase
{
public:
	LinearLayer(int inputSize, int outSize, int batchSize, int seqLen);


	void TransferStatesFromHost(std::vector<HostMatrixPtr*>& states) override;
	void TransferStatesToHost(std::vector<HostMatrixPtr*>& states) override;
	void ForwardSequence(DeviceMatrix& input) override;
	void BackpropSequence(DeviceMatrix& input, DeviceMatrix& outSens) override;
	void Optimize(OptimizerBase& optimizer) override;
	void ResetMemory() override;
	void ResetOptimizerCache() override;
private:
	std::unique_ptr<NeuroWeight> _w, _b;
	std::unique_ptr<DeviceMatrix> _identity;
};
