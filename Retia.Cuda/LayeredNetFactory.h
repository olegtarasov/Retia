#pragma once
#include <memory>
#include "LayeredNetFactoryBase.h"
#include "LayeredNet.h"

class LayeredNetFactory : public LayeredNetFactoryBase
{
public:
	LayeredNetFactory(int inputSize, int outputSize, int batchSize, int seqLen);


	void WithRMSPropOptimizer(float learningRate, float momentum, float decayRate, float weightDecay) override;
	void WithGruLayers(int layers, int inSize, int hSize, std::vector<RawMatrixPtr*>& states) override;
	void WithLinearLayer(int inSize, int outSize, std::vector<RawMatrixPtr*>& states) override;
	void WithSoftMaxLayer(int size) override;

private:
	LayeredNet* _fullNet;
};
