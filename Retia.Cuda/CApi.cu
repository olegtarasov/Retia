#include "CApi.h"

RMSPropOptimizer* CreateRMSPropOptimizer(float learningRate, float momentum, float decayRate, float weightDecay)
{
	return new RMSPropOptimizer(learningRate, momentum, decayRate, weightDecay);
}

void DestroyOptimizer(OptimizerBase* optimizer)
{
	delete optimizer;
}

void SetLearningRate(OptimizerBase* optimizer, float learningRate)
{
	optimizer->setLearningRate(learningRate);
}

LayeredNet* CreateLayeredNetwork(int inputSize, int outputSize, int batchSize, int seqLen)
{
	return new LayeredNet(inputSize, outputSize, batchSize, seqLen);
}

void DestroyLayeredNetwork(LayeredNet* network)
{
	delete network;
}

void SetNetworkOptimizer(LayeredNet* network, OptimizerBase* optimizer)
{
	network->setOptimizer(optimizer);
}

void AddNetworkLayer(LayeredNet* network, LayerBase* layer)
{
	network->AddLayer(layer);
}

LinearLayer* CreateLinearLayer(int inputSize, int outSize, int batchSize, int seqLen)
{
	return new LinearLayer(inputSize, outSize, batchSize, seqLen);
}

GruLayer* CreateGruLayer(int inputSize, int hSize, int layers, int batchSize, int seqLen)
{
	return new GruLayer(inputSize, hSize, layers, batchSize, seqLen);
}

SoftmaxLayer* CreateSoftmaxLayer(int inSize, int batchSize, int seqLen)
{
	return new SoftmaxLayer(inSize, batchSize, seqLen);
}
