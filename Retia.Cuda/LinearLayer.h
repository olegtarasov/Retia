#pragma once

#include <memory>
#include "NeuroLayer.h"
#include "NeuroWeigth.h"
#include "RawMatrixPtr.h"

class LinearLayer : public NeuroLayer
{
public:
	LinearLayer(int inputSize, int outSize, int batchSize, int seqLen);


	void TransferStatesFromHost(std::vector<RawMatrixPtr*>& states) override;
	void TransferStatesToHost(std::vector<RawMatrixPtr*>& states) override;
	void ForwardSequence(DeviceMatrix& input) override;
	void BackpropSequence(DeviceMatrix& input, DeviceMatrix& outSens) override;
	void Optimize(OptimizerBase& optimizer) override;
	void ResetMemory() override;
	void ResetOptimizerCache() override;
private:
	std::unique_ptr<NeuroWeigth> _w, _b;
	std::unique_ptr<DeviceMatrix> _identity;
};
