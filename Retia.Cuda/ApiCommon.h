#pragma once

#include "LayeredNet.h"
#include "RMSPropOptimizer.h"
#include "LinearLayer.h"
#include "GruLayer.h"
#include "SoftmaxLayer.h"

// TODO: _Crossplatform exports
#define GPUAPI extern "C" __declspec(dllexport)
#define _VOID void _cdecl

struct MatrixDefinition
{
	int Rows;
	int Columns;
	int SeqLength;
	float *Pointer;
};

struct WeightDefinition
{
	int Rows;
	int Columns;
	int SeqLength;
	float *WeightPtr, *GradPtr, *Cache1Ptr, *Cache2Ptr, *CacheMPtr;
};