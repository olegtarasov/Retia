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

void TransferLayerStatesFromHost(LayerBase* layer, MatrixDefinition* matrices, int matrixCount)
{
	auto states = GetMatrixPointers(matrices, matrixCount);

	layer->TransferStatesFromHost(states);

	DestroyMatrixPointers(states);
}

void TransferLayerStatesToHost(LayerBase* layer, MatrixDefinition* matrices, int matrixCount)
{
	auto states = GetMatrixPointers(matrices, matrixCount);

	layer->TransferStatesToHost(states);

	DestroyMatrixPointers(states);
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


