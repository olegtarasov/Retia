#pragma once

#include <memory>
#include <vector>
#include "LayerBase.h"

class LayeredNet
{
public:
	LayeredNet(int inputSize, int outputSize, int batchSize, int seqLen)
		: _inputSize(inputSize),
		_outputSize(outputSize),
		_batchSize(batchSize),
		_seqLen(seqLen)
	{
	}


	void UpdateLearningRate(float learningRate);
	double TrainSequence(std::vector<HostMatrixPtr*>& inputs, std::vector<HostMatrixPtr*>& targets);
	void AddLayer(LayerBase* layer);
	double TrainSequence(DeviceMatrix& input, DeviceMatrix& target);
	void Opimize();
	void ResetMemory();
	void ResetOptimizerCache();
	
	OptimizerBase& optimizer() const
	{
		return *_optimizer;
	}

	void setOptimizer(OptimizerBase* optimizer)
	{
		_optimizer = std::unique_ptr<OptimizerBase>(optimizer);
	}

private:
	int	_inputSize, _outputSize, _batchSize, _seqLen;

	std::vector<std::unique_ptr<LayerBase>> _layers;
	std::unique_ptr<OptimizerBase> _optimizer;
};
