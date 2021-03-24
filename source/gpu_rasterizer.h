#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "rasterizer.h"
#include "algebra.h"
#include "geometry_processor.h"
#include <vector>



namespace mge {


	class cudaRasterizer : public Rasterizer
	{
	public:
		enum PolygonMode{  Disconnected, Connected, Filled};
	public:
		cudaRasterizer(VideoBuffer* buffer);
		~cudaRasterizer();

	public:

		virtual bool initSession(Pixel p = Pixel(0));		// Allocate Buffer to GPU Memory
		virtual bool finishSession();		// Copy GPU Buffer to CPU Buffer and Free GPU Memory

	public:
		virtual bool clearBuffer(Pixel p);
		// overriding methods
		virtual bool drawPixel(int x, int y, Pixel p);
		virtual bool drawVerticalLine(int x, int y1, int y2, Pixel p);
		virtual bool drawHorizontalLine(int y, int x1, int x2, Pixel p);
		virtual bool drawFallingLine(int x1, int y1, int x2, int y2, Pixel p);
		virtual bool drawRisingLine(int x1, int y1, int x2, int y2, Pixel p);

		virtual bool drawLine(vector2d a, vector2d b, Pixel p);
		virtual bool drawPolygon(int count, vector2d points[], PolygonMode mode, Pixel p);
		virtual bool drawPolygon(std::vector<vector2d> points, PolygonMode mode, Pixel p);


		virtual bool drawTriangle(std::vector<vector2d> points, Pixel p);


		virtual bool drawMesh(Mesh m, Pixel p);

		virtual bool FillTriangle(vector2d points[3], Pixel p);


	public:
		// buffer inherited from Rasterizer
		void* gpuBuffer;
		void* depthAddr;
		cudaError_t cudaStatus;
		unsigned int lastAllocatedSize = 0;
	};

}