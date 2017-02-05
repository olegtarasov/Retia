#pragma once

#include "ManagedMatrixContainer.h"

using namespace Retia::Contracts;

public ref class LayerStateFactory
{
public:
	static ManagedMatrixContainer^ GetLayerState(LayerSpecBase^ spec)
	{
		auto result = gcnew ManagedMatrixContainer();

		if (spec->LayerType == LayerType::Linear)
		{
			auto ls = (LinearLayerSpec^)spec;

			result->AddMatrix(ls->W);
			result->AddMatrix(ls->b);
		}
		else if (spec->LayerType == LayerType::Gru)
		{
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
		}
		
		return result;
	}
};
