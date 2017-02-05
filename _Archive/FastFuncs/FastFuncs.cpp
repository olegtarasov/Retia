// FastFuncs.cpp : Defines the exported functions for the DLL application.
//

#include "stdafx.h"
#include <ppl.h>
using namespace concurrency;

EXTERN_C
 __declspec(dllexport) void __cdecl FastAdagradUpdate(double learningRate, double *weightMatrix, double *mem, double *gradient, int n)
{
	for (int i = 0; i < n; i++)
	{
		double grad = gradient[i];
		double grad2 = grad*grad;
		mem[i] += grad2;
		double k = learningRate / (sqrt(mem[i]) + 1e-8);
		weightMatrix[i] += -k*grad;
	}
}

EXTERN_C
__declspec(dllexport) void __cdecl FastRMSPropUpdate(double learningRate, double decayRate, double *weightMatrix, double *grad2_cache, double *gradient, int n)
{
	const double e = 1e-4;
	for (int i = 0; i < n; i++)
	{
		double grad = gradient[i];
		double grad2 = grad*grad;
		grad2_cache[i] = decayRate*grad2_cache[i] + (1-decayRate)*grad2;
		double k = sqrt(grad2_cache[i] + e);
		weightMatrix[i] += -learningRate*grad/k;
	}
}

//See http://arxiv.org/pdf/1308.0850v5.pdf, page 23
EXTERN_C
__declspec(dllexport) void __cdecl FastGravesRMSPropUpdate(double weightDecay, double learningRate, double decayRate, double momentum, double *weightMatrix, double *grad1_cache, double *grad2_cache, double *momentum_cache, double *gradient, int n)
{
	const double e = 1e-4;

	for (int i = 0; i < n; i++)
	{
		double grad = gradient[i];
		double grad2 = grad*grad;
		
		grad2_cache[i] = decayRate*grad2_cache[i] + (1 - decayRate)*grad2;
		grad1_cache[i] = decayRate*grad1_cache[i] + (1 - decayRate)*grad;

		double k = sqrt(grad2_cache[i] - grad1_cache[i] * grad1_cache[i] + e);

		momentum_cache[i] = momentum*momentum_cache[i] - learningRate*grad / k;
		
		weightMatrix[i] = weightMatrix[i] + momentum_cache[i] - learningRate*weightDecay*weightMatrix[i];
	}
}

inline double Sigmoid(double x)
{
	return 1.0 / (1 + exp(-x));
}

inline void ApplySigmoidElement(double *r, double *z, int i)
{
	r[i] = Sigmoid(r[i]);
	z[i] = Sigmoid(z[i]);
}

EXTERN_C
__declspec(dllexport) void __cdecl FastApplySigmoidRZ(double *r, double *z, int n)
{
	parallel_for(0, n, [&](int i)	
	{
		ApplySigmoidElement(r, z, i);
	});
}

inline void ApplyTanhElement(double *matrix, int i)
{
	matrix[i] = tanh(matrix[i]);
}

EXTERN_C
__declspec(dllexport) void __cdecl FastApplyTanh(double *matrix, int n)
{
	parallel_for(0, n, [&](int i)
	{
		ApplyTanhElement(matrix, i);
	});
}

inline void BackHadamardElement(double* sH, double* sR, double* sZ, double* sHprop, double* hNextComponent, double* z, double* r, double* h,
	double* newH,
	double* prevH, double* sO, double* propH, int i)
{
	double derH = 1 - (newH[i] * newH[i]);
	sH[i] = derH*z[i] * sO[i];
	sHprop[i] = sH[i] * r[i];
	double derR = r[i] * (1.0 - r[i]);
	sR[i] = derR*propH[i] * sH[i];
	double derZ = z[i] * (1.0 - z[i]);
	sZ[i] = derZ*(newH[i] - prevH[i])*sO[i];
	hNextComponent[i] = (1 - z[i])*sO[i];
}

__declspec(dllexport) void __cdecl  FastBackHadamards(double* sH, double* sR, double* sZ, double* sHprop, double* hNextComponent, double* z, double* r, double* h,
	double* newH,
	double* prevH, double* sO, double* propH, int n)
{
	/*
	var derH = hiddenOnes - (newH ^ newH);
	var sH = derH ^ z ^ sO;
	var sHprop = sH ^ r;

	var derR = r ^ (hiddenOnes - r);
	var sR = derR ^ propH ^ sH;

	var derZ = z ^ (hiddenOnes - z);
	var sZ = derZ ^ (newH - prevH) ^ sO;

	//The way prevH influence current state
	var hNextComponent = ((hiddenOnes - z) ^ sO);
	*/

	parallel_for(0, n, [&](int i)
	{
		BackHadamardElement(sH, sR, sZ, sHprop, hNextComponent, z, r, h, newH, prevH, sO, propH, i);
	});

	/*
	for (int i = 0; i < n; i++)
	{
		double derH = 1 - (newH[i] * newH[i]);
		sH[i] = derH*z[i] * sO[i];
		sHprop[i] = sH[i] * r[i];
		double derR = r[i] * (1.0 - r[i]);
		sR[i] = derR*propH[i] * sH[i];
		double derZ = z[i] * (1.0 - z[i]);
		sZ[i] = derZ*(newH[i] - prevH[i])*sO[i];
		hNextComponent[i] = (1 - z[i])*sO[i];
	}*/
}

void CalculateHElements(double* H, double* hCandidate, double* z, double* lastH, int i)
{
	H[i] = z[i] * hCandidate[i] + (1 - z[i]) *lastH[i];
}

EXTERN_C
__declspec(dllexport) void __cdecl FastCalculateH(double* H, double* hCandidate, double* z, double* lastH, int n)
{
	//var H = (z ^ hNew) + ((_hiddenOnes - z) ^ _lastH);

	parallel_for(0, n, [&](int i)
	{
		CalculateHElements(H, hCandidate, z, lastH, i);
	});
}