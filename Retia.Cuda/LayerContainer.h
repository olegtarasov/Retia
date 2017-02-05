#pragma once
#include "RawMatrixPtr.h"
#include <memory>
#include <vector>
#include "LayerContainerBase.h"
#include "NeuroLayer.h"

class LayerContainer : public LayerContainerBase
{
public:

	explicit LayerContainer(NeuroLayer* neuroLayer)
		: _layer(neuroLayer)
	{
	}

	void ForwardSequence(std::vector<RawMatrixPtr*>& input) override;
	void TransferOutputToHost(std::vector<RawMatrixPtr*>& output) override;
private:
	std::unique_ptr<NeuroLayer> _layer;
};
