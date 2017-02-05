#pragma once

#include "ManagedMatrixContainer.h"
#include "LayerStateFactory.h"
#include "LayerContainerBase.h"

using namespace System;
using namespace System::Collections::Generic;
using namespace Retia::Contracts;
using namespace MathNet::Numerics::LinearAlgebra::Single;

public ref class LayerTester
{
public:

	LayerTester(LayerSpecBase^ layerSpec)
	{
		auto states = LayerStateFactory::GetLayerState(layerSpec);

		if (layerSpec->LayerType == LayerType::Linear)
		{
			auto ls = (LinearLayerSpec^)layerSpec;

			_container = LayerContainerBase::LinearLayerContainer(ls->InputSize, ls->OutSize, ls->BatchSize, ls->SeqLen, states->matrices());
		}
		else if (layerSpec->LayerType == LayerType::Gru)
		{
			auto gs = (GruLayerSpec^)layerSpec;

			_container = LayerContainerBase::GruLayersContainer(gs->Layers, gs->InputSize, gs->HSize, gs->BatchSize, gs->SeqLen, states->matrices());
		}
		else if (layerSpec->LayerType == LayerType::Softmax)
		{
			_container = LayerContainerBase::SoftmaxLayerContainer(layerSpec->InputSize, layerSpec->BatchSize, layerSpec->SeqLen);
		}
		else
		{
			throw gcnew InvalidOperationException();
		}

		delete states;
	}

	~LayerTester()
	{
		delete _container;
	}

	void TestForward(List<Matrix^>^ inputs, List<Matrix^>^ outputs)
	{
		auto inMatrices = gcnew ManagedMatrixContainer();
		auto outMatrices = gcnew ManagedMatrixContainer();

		for (int i = 0; i < inputs->Count; i++)
		{
			inMatrices->AddMatrix(inputs[i]);
			outMatrices->AddMatrix(outputs[i]);
		}

		_container->ForwardSequence(inMatrices->matrices());
		_container->TransferOutputToHost(outMatrices->matrices());

		delete inMatrices;
		delete outMatrices;
	}
private:
	LayerContainerBase* _container;
};
