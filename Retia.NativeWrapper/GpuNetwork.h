// Retia.NativeWrapper.h

#pragma once

#include <memory>
#include <vector>
#include "LayeredNetBase.h"
#include "RawMatrixPtr.h"
#include "LayeredNetFactoryBase.h"
#include "LayerStateFactory.h"
#include "ManagedMatrixContainer.h"


using namespace System;
using namespace System::Collections::Generic;
using namespace Retia::Contracts;
using namespace MathNet::Numerics::LinearAlgebra;

namespace Retia::NativeWrapper {

	public ref class GpuNetwork : public IGpuOptimizerProxy, public IDisposable
	{
	public:
		GpuNetwork(LayeredNetSpec^ spec)
		{
			auto factory = LayeredNetFactoryBase::Create(spec->InputSize, spec->OutputSize, spec->BatchSize, spec->SeqLen);
			CreateOptimizer(*factory, spec->Optimizer);

			for (int i = 0; i < spec->Layers->Count; i++)
			{
				CreateLayer(*factory, spec->Layers[i]);
			}

			_network = factory->GetLayeredNet();
		}

		~GpuNetwork()
		{
			delete _network;
		}

		double TrainSequence(List<Matrix<float>^>^ inputs, List<Matrix<float>^>^ targets)
		{
			if (inputs->Count != targets->Count)
				throw gcnew InvalidOperationException("Input and target sequences don't have the same length!");

			auto inMatrices = gcnew ManagedMatrixContainer();
			auto targMatrices = gcnew ManagedMatrixContainer();
			
			for (int i = 0; i < inputs->Count; i++)
			{
				inMatrices->AddMatrix(inputs[i]);
				targMatrices->AddMatrix(targets[i]);
			}

			double result = _network->TrainSequence(inMatrices->matrices(), targMatrices->matrices());

			delete inMatrices;
			delete targMatrices;

			return result;
		}

		void TransferStatesToHost(LayeredNetSpec^ spec)
		{
			for (int i = 0; i < spec->Layers->Count; i++)
			{
				auto state = LayerStateFactory::GetLayerState(spec->Layers[i]);
				_network->TransferStatesToHost(i, state->matrices());
				state->SyncToMatrix();
			}
		}

		void Optimize()
		{
			_network->Opimize();
		}

		void ResetMemory()
		{
			_network->ResetMemory();
		}

		void ResetOptimizerCache()
		{
			_network->ResetOptimizerCache();
		}

		virtual void SetLearningRate(float learningRate)
		{
			_network->UpdateLearningRate(learningRate);
		}
		

	private:
		LayeredNetBase* _network;
		
		void CreateOptimizer(LayeredNetFactoryBase& factory, OptimizerSpecBase^ spec)
		{
			if (spec->OptimizerType == OptimizerType::RMSProp)
			{
				auto rs = (RMSPropSpec^)spec;
				factory.WithRMSPropOptimizer(rs->LearningRate, rs->Momentum, rs->DecayRate, rs->WeigthDecay);
				return;
			}
			
			throw gcnew InvalidOperationException();
		}

		void CreateLayer(LayeredNetFactoryBase& factory, LayerSpecBase^ spec)
		{
			auto layerState = LayerStateFactory::GetLayerState(spec);

			if (spec->LayerType == LayerType::Linear)
			{
				auto ls = (LinearLayerSpec^)spec;
				
				factory.WithLinearLayer(ls->InputSize, ls->OutSize, layerState->matrices());
			}
			else if (spec->LayerType == LayerType::Gru)
			{
				auto gs = (GruLayerSpec^)spec;
				
				factory.WithGruLayers(gs->Layers, gs->InputSize, gs->HSize, layerState->matrices());
			}
			else if (spec->LayerType == LayerType::Softmax)
			{
				factory.WithSoftMaxLayer(spec->InputSize);
			}

			delete layerState;
		}
	};
}
