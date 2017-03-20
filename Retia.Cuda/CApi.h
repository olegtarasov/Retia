#pragma once

#include <windows.h>
#include "LayeredNet.h"
#include "RMSPropOptimizer.h"
#include "LinearLayer.h"
#include "GruLayer.h"
#include "SoftmaxLayer.h"

#define GPUAPI EXTERN_C __declspec(dllexport)
#define _VOID void _cdecl

/*
 * Optimizers
 */
GPUAPI RMSPropOptimizer* _cdecl CreateRMSPropOptimizer(float learningRate, float momentum, float decayRate, float weightDecay);
GPUAPI _VOID DestroyOptimizer(OptimizerBase* optimizer);

/*
 * Network
 */
GPUAPI LayeredNet* _cdecl CreateLayeredNetwork(int input_size, int output_size, int batch_size, int seqLen);
GPUAPI _VOID DestroyLayeredNetwork(LayeredNet *network);
GPUAPI _VOID SetNetworkOptimizer(LayeredNet *network, OptimizerBase* optimizer);
GPUAPI _VOID AddNetworkLayer(LayeredNet *network, LayerBase *layer);

/*
 * Layers
 */
GPUAPI LinearLayer* _cdecl CreateLinearLayer(int inputSize, int outSize, int batchSize, int seqLen);
GPUAPI GruLayer* _cdecl CreateGruLayer(int inputSize, int hSize, int layers, int batchSize, int seqLen);
GPUAPI SoftmaxLayer* _cdecl CreateSoftmaxLayer(int inSize, int batchSize, int seqLen);
