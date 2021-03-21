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
		bool initSession(Pixel p = Pixel(0));		// Allocate Buffer to GPU Memory
		bool finishSession();		// Copy GPU Buffer to CPU Buffer and Free GPU Memory

	public:
		virtual bool drawPixel( int x, int y, Pixel p);
		virtual bool clearBuffer(Pixel p);

			virtual bool drawVerticalLine(int x, int y1, int y2, Pixel p);
			virtual bool drawHorizontalLine(int y, int x1, int x2, Pixel p);
		/*	virtual bool drawFallRightLine(int x1, int y1, int x2, int y2, Pixel p);
			virtual bool drawFallLeftLine(int x1, int y1, int x2, int y2, Pixel p);
			virtual bool drawLine(int x1, int y1, int x2, int y2, Pixel p);*/

	public:
		// buffer inherited from Rasterizer
		void* gpuBuffer;
		cudaError_t cudaStatus;
		unsigned int lastAllocatedSize = 0;
	};

}