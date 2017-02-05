#pragma once

#include <vector>
#include "LayeredNetBase.h"
#include "RawMatrixPtr.h"
#include <memory>

class LayeredNetFactoryBase
{
public:

	explicit LayeredNetFactoryBase(LayeredNetBase* net)
		: _net(net)
	{
	}

	virtual ~LayeredNetFactoryBase() = default;

	virtual void WithRMSPropOptimizer(float learningRate, float momentum, float decayRate, float weightDecay) = 0;
	virtual void WithGruLayers(int layers, int inSize, int hSize, std::vector<RawMatrixPtr*>& states) = 0;
	virtual void WithLinearLayer(int inSize, int outSize, std::vector<RawMatrixPtr*>& states) = 0;
	virtual void WithSoftMaxLayer(int size) = 0;

	LayeredNetBase* GetLayeredNet() const { return _net; };

	static std::unique_ptr<LayeredNetFactoryBase> Create(int inputSize, int outputSize, int batchSize, int seqLen);

protected:
	LayeredNetBase* _net;
};


