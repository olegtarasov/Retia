#pragma once

#include "ManagedMatrixContainer.h"

using namespace Retia::Contracts;

public ref class LayerStateFactory
{
public:
	static ManagedMatrixContainer^ GetLayerState(LayerSpecBase^ spec)
	{
		if (spec->LayerType == LayerType::Linear)
		{
			auto result = gcnew ManagedMatrixContainer();
			auto ls = (LinearLayerSpec^)spec;

			result->AddMatrix(ls->W);
			result->AddMatrix(ls->b);

			return result;
		}
		
		if (spec->LayerType == LayerType::Gru)
		{
			auto result = gcnew ManagedMatrixContainer(true);
			auto gs = (GruLayerSpec^)spec;

			for (int i = 0; i < gs->Layers; i++)
			{
				result->AddMatrix(gs->Weights[i]->Wxr);
				result->AddMatrix(gs->Weights[i]->Wxz);
				result->AddMatrix(gs->Weights[i]->Wxh);
				
				result->AddMatrix(gs->Weights[i]->Whr);
				result->AddMatrix(gs->Weights[i]->Whz);
				result->AddMatrix(gs->Weights[i]->Whh);
				
				result->AddMatrix(gs->Weights[i]->bxr);
				result->AddMatrix(gs->Weights[i]->bxz);
				result->AddMatrix(gs->Weights[i]->bxh);
				
				result->AddMatrix(gs->Weights[i]->bhr);
				result->AddMatrix(gs->Weights[i]->bhz);
				result->AddMatrix(gs->Weights[i]->bhh);
			}

			return result;
		}
		
		//valid layer, but no state to return, returning empy containter, not nullptr!
		if(spec->LayerType == LayerType::Softmax)
			return gcnew ManagedMatrixContainer(true);

		return nullptr;
	}
};
