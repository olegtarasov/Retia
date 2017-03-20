#pragma once
#include "NeuroWeigth.h"

class OptimizerBase
{
public:

	explicit OptimizerBase(float learningRate)
		: _learningRate(learningRate)
	{
	}

	virtual ~OptimizerBase() = default;


	float learningRate() const
	{
		return _learningRate;
	}

	void setLearningRate(float learningRate)
	{
		_learningRate = learningRate;
	}

	virtual void Optimize(NeuroWeigth& weigth) = 0;

protected:
	float	_learningRate;
};
