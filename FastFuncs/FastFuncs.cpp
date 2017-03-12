#include "stdafx.h"
#include <ppl.h>

#define FASTFUNC EXTERN_C __declspec(dllexport) void __cdecl

using namespace concurrency;

template <class T>
void AdamUpdate(T learningRate, T b1, T b2, int t, T *weights, T *cache1, T *cache2, T *grad, int n)
{
	const T e = 1e-8f;

	parallel_for(0, n, [&](int i)
	{
		T g = grad[i];

		cache1[i] = b1 * cache1[i] + (1 - b1) * g;
		cache2[i] = b2 * cache2[i] + (1 - b2) * g * g;

		T a = learningRate * sqrt(1 - pow(b2, t)) / (1 - pow(b1, t));
		weights[i] = weights[i] - a * cache1[i] / (sqrt(cache2[i]) + e);
	});
}

FASTFUNC AdamUpdateS(float learningRate, float b1, float b2, int t, float *weights, float *cache1, float *cache2, float *grad, int n)
{
	AdamUpdate(learningRate, b1, b2, t, weights, cache1, cache2, grad, n);
}

FASTFUNC AdamUpdateD(double learningRate, double b1, double b2, int t, double *weights, double *cache1, double *cache2, double *grad, int n)
{
	AdamUpdate(learningRate, b1, b2, t, weights, cache1, cache2, grad, n);
}

template <class T>
void AdagradUpdate(T learningRate, T *weightMatrix, T *mem, T *gradient, int n)
{
	for (int i = 0; i < n; i++)
	{
		T grad = gradient[i];
		T grad2 = grad*grad;
		mem[i] += grad2;
		T k = learningRate / (sqrt(mem[i]) + 1e-8f);
		weightMatrix[i] += -k*grad;
	}
}

FASTFUNC AdagradUpdateS(float learningRate, float *weightMatrix, float *mem, float *gradient, int n)
{
	AdagradUpdate(learningRate, weightMatrix, mem, gradient, n);
}

FASTFUNC AdagradUpdateD(double learningRate, double *weightMatrix, double *mem, double *gradient, int n)
{
	AdagradUpdate(learningRate, weightMatrix, mem, gradient, n);
}

//See http://arxiv.org/pdf/1308.0850v5.pdf, page 23
template <class T>
void GravesRMSPropUpdate(T weightDecay, T learningRate, T decayRate, T momentum, T *weightMatrix, T *grad1_cache, T *grad2_cache, T *momentum_cache, T *gradient, int n)
{
	const T e = 1e-4f;

	for (int i = 0; i < n; i++)
	{
		T grad = gradient[i];
		T grad2 = grad*grad;
		
		grad2_cache[i] = decayRate*grad2_cache[i] + (1 - decayRate)*grad2;
		grad1_cache[i] = decayRate*grad1_cache[i] + (1 - decayRate)*grad;

		T k = sqrt(grad2_cache[i] - grad1_cache[i] * grad1_cache[i] + e);

		momentum_cache[i] = momentum*momentum_cache[i] - learningRate*grad / k;
		
		weightMatrix[i] = weightMatrix[i] + momentum_cache[i] - learningRate*weightDecay*weightMatrix[i];
	}
}

FASTFUNC GravesRMSPropUpdateD(double weightDecay, double learningRate, double decayRate, double momentum, double *weightMatrix, double *grad1_cache, double *grad2_cache, double *momentum_cache, double *gradient, int n)
{
	GravesRMSPropUpdate(weightDecay, learningRate, decayRate, momentum, weightMatrix, grad1_cache, grad2_cache, momentum_cache, gradient, n);
}

FASTFUNC GravesRMSPropUpdateS(float weightDecay, float learningRate, float decayRate, float momentum, float *weightMatrix, float *grad1_cache, float *grad2_cache, float *momentum_cache, float *gradient, int n)
{
	GravesRMSPropUpdate(weightDecay, learningRate, decayRate, momentum, weightMatrix, grad1_cache, grad2_cache, momentum_cache, gradient, n);
}

template <class T>
T Sigmoid(T x)
{
	return (T)(1.0 / (1 + exp(-x)));
}

template <class T>
void ApplySigmoid2(T *a, T *b, int n)
{
	parallel_for(0, n, [&](int i)
	{
		a[i] = Sigmoid(a[i]);
		b[i] = Sigmoid(b[i]);
	});
}

template <class T>
void ApplySigmoid(T *matrix, int n)
{
	parallel_for(0, n, [&](int i)
	{
		matrix[i] = Sigmoid(matrix[i]);
	});
}

FASTFUNC ApplySigmoid2D(double *a, double *b, int n)
{
	ApplySigmoid2(a, b, n);
}

FASTFUNC ApplySigmoid2S(float *a, float *b, int n)
{
	ApplySigmoid2(a, b, n);
}

FASTFUNC ApplySigmoidD(double *matrix, int n)
{
	ApplySigmoid(matrix, n);
}

FASTFUNC ApplySigmoidS(float *matrix, int n)
{
	ApplySigmoid(matrix, n);
}

template <class T>
void ApplyTanh(T *matrix, int n)
{
	parallel_for(0, n, [&](int i)
	{
		matrix[i] = tanh(matrix[i]);
	});
}

FASTFUNC ApplyTanhD(double *matrix, int n)
{
	ApplyTanh(matrix, n);
}

FASTFUNC ApplyTanhS(float *matrix, int n)
{
	ApplyTanh(matrix, n);
}

template <class T>
void CalculateHElements(T* H, T* hCandidate, T* z, T* lastH, int i)
{
	H[i] = (1 - z[i]) * hCandidate[i] + z[i] *lastH[i];
}

FASTFUNC CalculateHD(double* H, double* hCandidate, double* z, double* lastH, int n)
{
	parallel_for(0, n, [&](int i)
	{
		CalculateHElements(H, hCandidate, z, lastH, i);
	});
}

FASTFUNC CalculateHS(float* H, float* hCandidate, float* z, float* lastH, int n)
{
	parallel_for(0, n, [&](int i)
	{
		CalculateHElements(H, hCandidate, z, lastH, i);
	});
}

// Old stuff

//inline void BackHadamardElement(double* sH, double* sR, double* sZ, double* sHprop, double* hNextComponent, double* z, double* r, double* h,
//	double* newH,
//	double* prevH, double* sO, double* propH, int i)
//{
//	double derH = 1 - (newH[i] * newH[i]);
//	sH[i] = derH*z[i] * sO[i];
//	sHprop[i] = sH[i] * r[i];
//	double derR = r[i] * (1.0 - r[i]);
//	sR[i] = derR*propH[i] * sH[i];
//	double derZ = z[i] * (1.0 - z[i]);
//	sZ[i] = derZ*(newH[i] - prevH[i])*sO[i];
//	hNextComponent[i] = (1 - z[i])*sO[i];
//}
//
//FASTFUNC  FastBackHadamards(double* sH, double* sR, double* sZ, double* sHprop, double* hNextComponent, double* z, double* r, double* h,
//	double* newH,
//	double* prevH, double* sO, double* propH, int n)
//{
//	/*
//	var derH = hiddenOnes - (newH ^ newH);
//	var sH = derH ^ z ^ sO;
//	var sHprop = sH ^ r;
//
//	var derR = r ^ (hiddenOnes - r);
//	var sR = derR ^ propH ^ sH;
//
//	var derZ = z ^ (hiddenOnes - z);
//	var sZ = derZ ^ (newH - prevH) ^ sO;
//
//	//The way prevH influence current state
//	var hNextComponent = ((hiddenOnes - z) ^ sO);
//	*/
//
//	parallel_for(0, n, [&](int i)
//	{
//		BackHadamardElement(sH, sR, sZ, sHprop, hNextComponent, z, r, h, newH, prevH, sO, propH, i);
//	});
//
//	/*
//	for (int i = 0; i < n; i++)
//	{
//	double derH = 1 - (newH[i] * newH[i]);
//	sH[i] = derH*z[i] * sO[i];
//	sHprop[i] = sH[i] * r[i];
//	double derR = r[i] * (1.0 - r[i]);
//	sR[i] = derR*propH[i] * sH[i];
//	double derZ = z[i] * (1.0 - z[i]);
//	sZ[i] = derZ*(newH[i] - prevH[i])*sO[i];
//	hNextComponent[i] = (1 - z[i])*sO[i];
//	}*/
//}

//FASTFUNC FastRMSPropUpdate(double learningRate, double decayRate, double *weightMatrix, double *grad2_cache, double *gradient, int n)
//{
//	const double e = 1e-4;
//	for (int i = 0; i < n; i++)
//	{
//		double grad = gradient[i];
//		double grad2 = grad*grad;
//		grad2_cache[i] = decayRate*grad2_cache[i] + (1 - decayRate)*grad2;
//		double k = sqrt(grad2_cache[i] + e);
//		weightMatrix[i] += -learningRate*grad / k;
//	}
//}
