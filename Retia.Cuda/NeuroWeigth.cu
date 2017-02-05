#include "NeuroWeigth.h"

void NeuroWeigth::ClearCache()
{
	_cache1->ZeroMemory();
	_cache2->ZeroMemory();
	_cacheM->ZeroMemory();
}

void NeuroWeigth::ClearGradient()
{
	_gradient->ZeroMemory();
}
