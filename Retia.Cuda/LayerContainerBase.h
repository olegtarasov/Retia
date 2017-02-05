#pragma once
#include <vector>
#include "RawMatrixPtr.h"

class LayerContainerBase
{
public:
	virtual ~LayerContainerBase() = default;
	
	static LayerContainerBase* GruLayersContainer(int layers, int inSize, int hSize, int batchSize, int seqLen, std::vector<RawMatrixPtr*>& states);
	static LayerContainerBase* LinearLayerContainer(int inSize, int outSize, int batchSize, int seqLen, std::vector<RawMatrixPtr*>& states);
	static LayerContainerBase* SoftmaxLayerContainer(int inSize, int batchSize, int seqLen);

	virtual void ForwardSequence(std::vector<RawMatrixPtr*>& input) = 0;
	virtual void TransferOutputToHost(std::vector<RawMatrixPtr*>& output) = 0;
};