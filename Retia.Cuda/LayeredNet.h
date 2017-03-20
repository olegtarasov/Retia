#pragma once

#include <memory>
#include <vector>
#include "LayerBase.h"
#include "LayeredNetBase.h"

class LayeredNet : public LayeredNetBase
{
public:
	LayeredNet(int inputSize, int outputSize, int batchSize, int seqLen)
		: LayeredNetBase(inputSize, outputSize, batchSize, seqLen)
	{
	}


	void UpdateLearningRate(float learningRate) override;
	void TransferStatesToHost(int layer, std::vector<RawMatrixPtr*>& states) override;
	double TrainSequence(std::vector<RawMatrixPtr*>& inputs, std::vector<RawMatrixPtr*>& targets) override;
	void AddLayer(LayerBase* layer);
	double TrainSequence(DeviceMatrix& input, DeviceMatrix& target);
	void Opimize() override;
	void ResetMemory() override;
	void ResetOptimizerCache() override;
	
	OptimizerBase& optimizer() const
	{
		return *_optimizer;
	}

	void setOptimizer(OptimizerBase* optimizer)
	{
		_optimizer = std::unique_ptr<OptimizerBase>(optimizer);
	}

private:
	std::vector<std::unique_ptr<LayerBase>> _layers;
	std::unique_ptr<OptimizerBase> _optimizer;
};
