#pragma once

#include "ApiCommon.h"
/*
* Tests
*/
GPUAPI double _cdecl TestCrossEntropyErrorCpu(MatrixDefinition m1, MatrixDefinition m2);
GPUAPI double _cdecl TestCrossEntropyErrorGpu(MatrixDefinition m1, MatrixDefinition m2);
GPUAPI _VOID TestCrossEntropyBackpropCpu(MatrixDefinition m1, MatrixDefinition m2, MatrixDefinition result);
GPUAPI _VOID TestCrossEntropyBackpropGpu(MatrixDefinition m1, MatrixDefinition m2, MatrixDefinition result);
GPUAPI _VOID TestRMSPropUpdateCpu(MatrixDefinition weight, MatrixDefinition grad, MatrixDefinition cache1,
	MatrixDefinition cache2, MatrixDefinition cacheM, float learningRate, float decayRate, float momentum, float weightDecay);
GPUAPI _VOID TestRMSPropUpdateGpu(MatrixDefinition weight, MatrixDefinition grad, MatrixDefinition cache1,
	MatrixDefinition cache2, MatrixDefinition cacheM, float learningRate, float decayRate, float momentum, float weightDecay);
GPUAPI _VOID TestClampMatrixCpu(MatrixDefinition matrix, float threshold);
GPUAPI _VOID TestClampMatrixGpu(MatrixDefinition matrix, float threshold);
GPUAPI _VOID TestMatrixTransferCpu(MatrixDefinition matrix);
GPUAPI _VOID TestMatrixTransferGpu(MatrixDefinition matrix);
GPUAPI _VOID TestMatrixTransferRowMajorCpu(MatrixDefinition matrix);
GPUAPI _VOID TestMatrixTransferRowMajorGpu(MatrixDefinition matrix);
GPUAPI _VOID TestWeightTransferCpu(WeightDefinition weight);
GPUAPI _VOID TestWeightTransferGpu(WeightDefinition weight);
GPUAPI _VOID TestWeightTransferRowMajorCpu(WeightDefinition weight);
GPUAPI _VOID TestWeightTransferRowMajorGpu(WeightDefinition weight);
GPUAPI _VOID TestComplexWeightTransfer(WeightDefinition weight);
GPUAPI _VOID TestComplexWeightTransferRowMajor(WeightDefinition weight);