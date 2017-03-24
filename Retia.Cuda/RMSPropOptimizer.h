#pragma once
#include "OptimizerBase.h"

class RMSPropOptimizer : public OptimizerBase
{
public:

	RMSPropOptimizer(float learningRate, float momentum, float decayRate, float weightDecay)
		: OptimizerBase(learningRate),
		_momentum(momentum),
		_decayRate(decayRate),
		_weightDecay(weightDecay)
	{
	}

	void Optimize(NeuroWeight& weigth) override;
private:
	float _momentum, _decayRate, _weightDecay;
};
