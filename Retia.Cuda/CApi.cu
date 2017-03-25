#include "CApi.h"
#include "Algorithms.h"

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

double TestCrossEntropyErrorCpu(MatrixDefinition m1, MatrixDefinition m2)
{
	auto mat1 = std::make_unique<HostMatrixPtr>(m1.Rows, m1.Columns, m1.SeqLength, m1.Pointer);
	auto mat2 = std::make_unique<HostMatrixPtr>(m2.Rows, m2.Columns, m2.SeqLength, m2.Pointer);

	return Algorithms::CrossEntropyError(*mat1, *mat2);
}

double TestCrossEntropyErrorGpu(MatrixDefinition m1, MatrixDefinition m2)
{
	auto mat1 = std::make_unique<HostMatrixPtr>(m1.Rows, m1.Columns, m1.SeqLength, m1.Pointer);
	auto mat2 = std::make_unique<HostMatrixPtr>(m2.Rows, m2.Columns, m2.SeqLength, m2.Pointer);

	auto gpum1 = std::make_unique<DeviceMatrix>(m1.Rows, m1.Columns, m1.SeqLength);
	auto gpum2 = std::make_unique<DeviceMatrix>(m2.Rows, m2.Columns, m2.SeqLength);

	mat1->CopyTo(*gpum1);
	mat2->CopyTo(*gpum2);

	return Algorithms::CrossEntropyError(*gpum1, *gpum2);
}

void TestCrossEntropyBackpropCpu(MatrixDefinition m1, MatrixDefinition m2, MatrixDefinition result)
{
	auto mat1 = std::make_unique<HostMatrixPtr>(m1.Rows, m1.Columns, m1.SeqLength, m1.Pointer);
	auto mat2 = std::make_unique<HostMatrixPtr>(m2.Rows, m2.Columns, m2.SeqLength, m2.Pointer);
	auto mResult = std::make_unique<HostMatrixPtr>(result.Rows, result.Columns, result.SeqLength, result.Pointer);

	Algorithms::BackpropagateCrossEntropyError(*mat1, *mat2, *mResult);
}

void TestCrossEntropyBackpropGpu(MatrixDefinition m1, MatrixDefinition m2, MatrixDefinition result)
{
	auto mat1 = std::make_unique<HostMatrixPtr>(m1.Rows, m1.Columns, m1.SeqLength, m1.Pointer);
	auto mat2 = std::make_unique<HostMatrixPtr>(m2.Rows, m2.Columns, m2.SeqLength, m2.Pointer);
	auto mResult = std::make_unique<HostMatrixPtr>(result.Rows, result.Columns, result.SeqLength, result.Pointer);

	auto gpum1 = std::make_unique<DeviceMatrix>(m1.Rows, m1.Columns, m1.SeqLength);
	auto gpum2 = std::make_unique<DeviceMatrix>(m2.Rows, m2.Columns, m2.SeqLength);
	auto gpumRes = std::make_unique<DeviceMatrix>(result.Rows, result.Columns, result.SeqLength);

	mat1->CopyTo(*gpum1);
	mat2->CopyTo(*gpum2);
	mResult->CopyTo(*gpumRes);

	Algorithms::BackpropagateCrossEntropyError(*gpum1, *gpum2, *gpumRes);

	mResult->CopyFrom(*gpumRes);
}

void TestRMSPropUpdateCpu(MatrixDefinition weight, MatrixDefinition grad, MatrixDefinition cache1, MatrixDefinition cache2, MatrixDefinition cacheM, float learningRate, float decayRate, float momentum, float weightDecay)
{
	auto mWeight = std::make_unique<HostMatrixPtr>(weight.Rows, weight.Columns, weight.SeqLength, weight.Pointer);
	auto mGrad = std::make_unique<HostMatrixPtr>(grad.Rows, grad.Columns, grad.SeqLength, grad.Pointer);
	auto mCache1 = std::make_unique<HostMatrixPtr>(cache1.Rows, cache1.Columns, cache1.SeqLength, cache1.Pointer);
	auto mCache2 = std::make_unique<HostMatrixPtr>(cache2.Rows, cache2.Columns, cache2.SeqLength, cache2.Pointer);
	auto mCacheM = std::make_unique<HostMatrixPtr>(cacheM.Rows, cacheM.Columns, cacheM.SeqLength, cacheM.Pointer);

	Algorithms::RMSPropOptimize(*mWeight, *mGrad, *mCache1, *mCache2, *mCacheM, learningRate, decayRate, momentum, weightDecay);
}

void TestRMSPropUpdateGpu(MatrixDefinition weight, MatrixDefinition grad, MatrixDefinition cache1, MatrixDefinition cache2, MatrixDefinition cacheM, float learningRate, float decayRate, float momentum, float weightDecay)
{
	auto mWeight = std::make_unique<HostMatrixPtr>(weight.Rows, weight.Columns, weight.SeqLength, weight.Pointer);
	auto mGrad = std::make_unique<HostMatrixPtr>(grad.Rows, grad.Columns, grad.SeqLength, grad.Pointer);
	auto mCache1 = std::make_unique<HostMatrixPtr>(cache1.Rows, cache1.Columns, cache1.SeqLength, cache1.Pointer);
	auto mCache2 = std::make_unique<HostMatrixPtr>(cache2.Rows, cache2.Columns, cache2.SeqLength, cache2.Pointer);
	auto mCacheM = std::make_unique<HostMatrixPtr>(cacheM.Rows, cacheM.Columns, cacheM.SeqLength, cacheM.Pointer);

	auto gmWeight = std::make_unique<DeviceMatrix>(weight.Rows, weight.Columns, weight.SeqLength);
	auto gmGrad = std::make_unique<DeviceMatrix>(grad.Rows, grad.Columns, grad.SeqLength);
	auto gmCache1 = std::make_unique<DeviceMatrix>(cache1.Rows, cache1.Columns, cache1.SeqLength);
	auto gmCache2 = std::make_unique<DeviceMatrix>(cache2.Rows, cache2.Columns, cache2.SeqLength);
	auto gmCacheM = std::make_unique<DeviceMatrix>(cacheM.Rows, cacheM.Columns, cacheM.SeqLength);

	mWeight->CopyTo(*gmWeight);
	mGrad->CopyTo(*gmGrad);
	mCache1->CopyTo(*gmCache1);
	mCache2->CopyTo(*gmCache2);
	mCacheM->CopyTo(*gmCacheM);

	Algorithms::RMSPropOptimize(*gmWeight, *gmGrad, *gmCache1, *gmCache2, *gmCacheM, learningRate, decayRate, momentum, weightDecay);

	mWeight->CopyFrom(*gmWeight);
	mCache1->CopyFrom(*gmCache1);
	mCache2->CopyFrom(*gmCache2);
	mCacheM->CopyFrom(*gmCacheM);
}

void TestClampMatrixCpu(MatrixDefinition matrix, float threshold)
{
	auto mat = std::make_unique<HostMatrixPtr>(matrix.Rows, matrix.Columns, matrix.SeqLength, matrix.Pointer);

	Algorithms::Clamp(*mat, threshold);
}

void TestClampMatrixGpu(MatrixDefinition matrix, float threshold)
{
	auto mat = std::make_unique<HostMatrixPtr>(matrix.Rows, matrix.Columns, matrix.SeqLength, matrix.Pointer);
	auto gMat = std::make_unique<DeviceMatrix>(matrix.Rows, matrix.Columns, matrix.SeqLength);

	mat->CopyTo(*gMat);

	Algorithms::Clamp(*gMat, threshold);

	mat->CopyFrom(*gMat);
}


