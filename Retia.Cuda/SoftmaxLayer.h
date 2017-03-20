#pragma once
#include "LayerBase.h"
#include "CuDnnTensor.h"
#include "RawMatrixPtr.h"

class SoftmaxLayer : public LayerBase
{
public:
	SoftmaxLayer(int inSize, int batchSize, int seqLen);


	void TransferStatesFromHost(std::vector<RawMatrixPtr*>& states) override;
	void TransferStatesToHost(std::vector<RawMatrixPtr*>& states) override;
	void ForwardSequence(DeviceMatrix& input) override;
	void BackpropSequence(DeviceMatrix& input, DeviceMatrix& outSens) override;
	void Optimize(OptimizerBase& optimizer) override;
	void ErrorPropagate(DeviceMatrix& target) override;
	double LayerError(DeviceMatrix& target) override;
	void ResetMemory() override;
	void ResetOptimizerCache() override;
private:
	std::unique_ptr<CuDnnNdTensor>	_xTensor;
};
