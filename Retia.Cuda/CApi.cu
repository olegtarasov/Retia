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

void TransferLayerStatesToDevice(LayerBase* layer, WeightDefinition *weights, int count)
{
	auto states = GetWeightSyncContainers(weights, count);

	layer->TransferStatesToDevice(states);

	DestroyWeightSyncContainers(states);
}

void TransferLayerStatesToHost(LayerBase* layer, WeightDefinition *weights, int count)
{
	auto states = GetWeightSyncContainers(weights, count);

	layer->TransferStatesToHost(states);

	DestroyWeightSyncContainers(states);
}

double TrainSequence(LayeredNet* net, MatrixDefinition* inputs, MatrixDefinition* targets, int count)
{
	auto in = GetMatrixPointers(inputs, count);
	auto targ = GetMatrixPointers(targets, count);

	double result = net->TrainSequence(in, targ);

	DestroyMatrixPointers(in);
	DestroyMatrixPointers(targ);

	return result;
}

std::vector<WeightSyncContainer*> GetWeightSyncContainers(WeightDefinition* weights, int count)
{
	std::vector<WeightSyncContainer*> result;

	for (int i = 0; i < count; ++i)
	{
		auto cur = weights[i];
		result.push_back(new WeightSyncContainer(cur.Rows, cur.Columns, cur.SeqLength, cur.WeightPtr, cur.GradPtr, cur.Cache1Ptr, cur.Cache2Ptr, cur.CacheMPtr));
	}

	return result;
}

void DestroyWeightSyncContainers(std::vector<WeightSyncContainer*>& containers)
{
	for (int i = 0; i < containers.size(); ++i)
	{
		delete containers[i];
	}
}

std::vector<HostMatrixPtr*> GetMatrixPointers(MatrixDefinition* matrices, int matrixCount)
{
	std::vector<HostMatrixPtr*> result;

	for (int i = 0; i < matrixCount; ++i)
	{
		auto cur = matrices[i];
		result.push_back(new HostMatrixPtr(cur.Rows, cur.Columns, cur.SeqLength, cur.Pointer));
	}

	return result;
}

void DestroyMatrixPointers(std::vector<HostMatrixPtr*>& ptrs)
{
	for (int i = 0; i < ptrs.size(); ++i)
	{
		delete ptrs[i];
	}
}


