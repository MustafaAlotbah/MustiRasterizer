#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "rasterizer.h"




namespace mge {


	class GPURasterizer : public Rasterizer
	{
	public:
		GPURasterizer(VideoBuffer* buffer);
		~GPURasterizer();

	public:
		bool initSession();		// Allocate Buffer to GPU Memory
		bool finishSession();		// Copy GPU Buffer to CPU Buffer and Free GPU Memory

	public:
		virtual bool drawPixel( int x, int y, Pixel p);

		/*	__device__ bool drawVerticalLine(int x, int y1, int y2, Pixel p);
			__device__ bool drawHorizontalLine(int y, int x1, int x2, Pixel p);
			__device__ bool drawFallRightLine(int x1, int y1, int x2, int y2, Pixel p);
			__device__ bool drawFallLeftLine(int x1, int y1, int x2, int y2, Pixel p);
			__device__ bool drawLine(int x1, int y1, int x2, int y2, Pixel p);*/

	public:
		// buffer inherited from Rasterizer
		void* gpuBuffer;
		cudaError_t cudaStatus;
		int lastAllocatedSize = 0;
	};

}