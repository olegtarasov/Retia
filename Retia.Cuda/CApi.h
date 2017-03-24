#pragma once

#include "LayeredNet.h"
#include "RMSPropOptimizer.h"
#include "LinearLayer.h"
#include "GruLayer.h"
#include "SoftmaxLayer.h"

// TODO: _Crossplatform exports
#define GPUAPI extern "C" __declspec(dllexport)
#define _VOID void _cdecl

/*
 * Optimizers
 */
GPUAPI RMSPropOptimizer* _cdecl CreateRMSPropOptimizer(float learningRate, float momentum, float decayRate, float weightDecay);
GPUAPI _VOID DestroyOptimizer(OptimizerBase *optimizer);
GPUAPI _VOID SetLearningRate(OptimizerBase *optimizer, float learningRate);

/*
 * Network
 */
GPUAPI LayeredNet* _cdecl CreateLayeredNetwork(int inputSize, int outputSize, int batchSize, int seqLen);
GPUAPI _VOID DestroyLayeredNetwork(LayeredNet *network);
GPUAPI _VOID SetNetworkOptimizer(LayeredNet *network, OptimizerBase* optimizer);
GPUAPI _VOID AddNetworkLayer(LayeredNet *network, LayerBase *layer);

/*
 * Layers
 */
GPUAPI LinearLayer* _cdecl CreateLinearLayer(int inputSize, int outSize, int batchSize, int seqLen);
GPUAPI GruLayer* _cdecl CreateGruLayer(int inputSize, int hSize, int layers, int batchSize, int seqLen);
GPUAPI SoftmaxLayer* _cdecl CreateSoftmaxLayer(int inSize, int batchSize, int seqLen);

/*
 * State transfer
 */
struct MatrixDefinition
{
	int Rows;
	int Columns;
	int SeqLength;
	float *Pointer;
};

struct WeightDefinition
{
	int Rows;
	int Columns;
	int SeqLength;
	float *WeightPtr, *GradPtr, *Cache1Ptr, *Cache2Ptr, *CacheMPtr;
};

GPUAPI _VOID TransferLayerStatesToDevice(LayerBase *layer, WeightDefinition *weights, int count);
GPUAPI _VOID TransferLayerStatesToHost(LayerBase *layer, WeightDefinition *weights, int count);

/*
 * Training
 */
GPUAPI double _cdecl TrainSequence(LayeredNet *net, MatrixDefinition *inputs, MatrixDefinition *targets, int count);

/*
 * Helpers
 */
std::vector<WeightSyncContainer*> GetWeightSyncContainers(WeightDefinition* weights, int count);
void DestroyWeightSyncContainers(std::vector<WeightSyncContainer*>& containers);
std::vector<HostMatrixPtr*> GetMatrixPointers(MatrixDefinition *matrices, int matrixCount);
void DestroyMatrixPointers(std::vector<HostMatrixPtr*>& ptrs);