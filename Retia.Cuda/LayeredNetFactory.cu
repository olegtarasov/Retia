#include "LayeredNetFactory.h"
#include "RMSPropOptimizer.h"
#include "GruLayer.h"
#include "LinearLayer.h"
#include "SoftmaxLayer.h"

LayeredNetFactory::LayeredNetFactory(int inputSize, int outputSize, int batchSize, int seqLen)
	: LayeredNetFactoryBase(new LayeredNet(inputSize, outputSize, batchSize, seqLen))
{
	_fullNet = static_cast<LayeredNet*>(_net);
}

void LayeredNetFactory::WithRMSPropOptimizer(float learningRate, float momentum, float decayRate, float weightDecay)
{
	_fullNet->set_optimizer(new RMSPropOptimizer(learningRate, momentum, decayRate, weightDecay));
}

void LayeredNetFactory::WithGruLayers(int layers, int inSize, int hSize, std::vector<RawMatrixPtr*>& states)
{
	auto layer = new GruLayer(inSize, hSize, layers, _net->batchSize(), _net->seqLen());
	layer->TransferStatesFromHost(states);
	_fullNet->AddLayer(layer);
}

void LayeredNetFactory::WithLinearLayer(int inSize, int outSize, std::vector<RawMatrixPtr*>& states)
{
	auto layer = new LinearLayer(inSize, outSize, _net->batchSize(), _net->seqLen());
	layer->TransferStatesFromHost(states);
	_fullNet->AddLayer(layer);
}

void LayeredNetFactory::WithSoftMaxLayer(int size)
{
	auto layer = new SoftmaxLayer(size, _net->batchSize(), _net->seqLen());
	_fullNet->AddLayer(layer);
}

