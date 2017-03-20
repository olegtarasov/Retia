#pragma once
#include "RawMatrixPtr.h"
#include <memory>
#include <vector>
#include "LayerContainerBase.h"
#include "LayerBase.h"

class LayerContainer : public LayerContainerBase
{
public:

	explicit LayerContainer(LayerBase* neuroLayer)
		: _layer(neuroLayer)
	{
	}

	void ForwardSequence(std::vector<RawMatrixPtr*>& input) override;
	void TransferOutputToHost(std::vector<RawMatrixPtr*>& output) override;
private:
	std::unique_ptr<LayerBase> _layer;
};
