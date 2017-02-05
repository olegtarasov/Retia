#pragma once

#include <memory>
#include <vector>
#include "NeuroLayer.h"
#include "LayeredNetBase.h"

class LayeredNet : public LayeredNetBase
{
public:
	LayeredNet(int input_size, int output_size, int batch_size, int seqLen)
		: LayeredNetBase(input_size, output_size, batch_size, seqLen)
	{
	}


	double TrainSequence(std::vector<RawMatrixPtr*>& inputs, std::vector<RawMatrixPtr*>& targets) override;
	void AddLayer(NeuroLayer* layer);
	double TrainSequence(DeviceMatrix& input, DeviceMatrix& target);
	void Opimize() override;
	void ResetMemory() override;
	void ResetOptimizerCache() override;
	
	OptimizerBase& optimizer() const
	{
		return *_optimizer;
	}

	void set_optimizer(OptimizerBase* optimizer)
	{
		_optimizer = std::unique_ptr<OptimizerBase>(optimizer);
	}

private:
	std::vector<std::unique_ptr<NeuroLayer>> _layers;
	std::unique_ptr<OptimizerBase> _optimizer;
};
