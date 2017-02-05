#include "LayerContainerBase.h"
#include "GruLayer.h"
#include "LayerContainer.h"
#include "LinearLayer.h"
#include "SoftmaxLayer.h"

LayerContainerBase* LayerContainerBase::GruLayersContainer(int layers, int inSize, int hSize, int batchSize, int seqLen, std::vector<RawMatrixPtr*>& states)
{
	auto layer = new GruLayer(inSize, hSize, layers, batchSize, seqLen);
	layer->TransferStatesFromHost(states);
	return new LayerContainer(layer);
}

LayerContainerBase* LayerContainerBase::LinearLayerContainer(int inSize, int outSize, int batchSize, int seqLen, std::vector<RawMatrixPtr*>& states)
{
	auto layer = new LinearLayer(inSize, outSize, batchSize, seqLen);
	layer->TransferStatesFromHost(states);
	return new LayerContainer(layer);
}

LayerContainerBase* LayerContainerBase::SoftmaxLayerContainer(int inSize, int batchSize, int seqLen)
{
	return new LayerContainer(new SoftmaxLayer(inSize, batchSize, seqLen));
}
