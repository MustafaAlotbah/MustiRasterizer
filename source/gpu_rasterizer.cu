/*
	Rasterizer in Cuda C

	Use this way:
	rasterizer.initSession();			<- always include these
	...
	rasterizer.drawLine or whatever ();
	...
	rasterizer.finishSession();			<- always include these

*/

#define DEBUG

#include "gpu_rasterizer.h"




namespace mge {


	/*---------------------CUDA Functions-------------------------*/
	__device__ void gpuDrawPixel(void* gpuBuffer, int width, int height, int x, int y, Pixel p) {
		y = height - y - 1;	// correct the orientation
		*((uint32_t*)gpuBuffer + y * width + x) = p.color.value;
	}

	__global__ void gpuClearBufferKernel(void* gpuBuffer, int width, int height, Pixel p) {
		//int i = threadIdx.x; 
		//*((uint32_t*)gpuBuffer + i) = 0xFF00FF;//p.color.value;

		gpuDrawPixel(gpuBuffer, width, height, threadIdx.x + 1000*blockIdx.x % (width * height), height-1, p);
	}

	__global__ void gpuDrawPixelKernel(void* gpuBuffer, int width, int height, int x, int y, Pixel p) {
		gpuDrawPixel(gpuBuffer, width, height, x, y, p);
	}





	/*---------------------Rasterizer Interface-------------------------*/

	GPURasterizer::GPURasterizer(VideoBuffer* buffer) :
	Rasterizer(buffer)
	{
		cudaStatus = cudaSetDevice(0);
	}


	GPURasterizer::~GPURasterizer(){
	}

	bool GPURasterizer::clearBuffer(Pixel p) {

		dim3 thrds(1000, 1);
		dim3 blcks(lastAllocatedSize / 1000 + 1, 1);
		gpuClearBufferKernel <<<blcks, thrds >>> (gpuBuffer, buffer->width, buffer->height, p);
		return true;

	}
	bool GPURasterizer::initSession(Pixel p) {
		lastAllocatedSize = buffer->height * buffer->width;
		// allocate GPU Memory (will be disposed of by finishing the session
		cudaMalloc((void**)&gpuBuffer, sizeof(uint32_t) * lastAllocatedSize) ;
		clearBuffer(p);
		return true;
	}


	bool GPURasterizer::finishSession() {
		// copy GPU buffer into CPU buffer and free GPU buffer
		cudaMemcpy(buffer->addr, gpuBuffer, lastAllocatedSize * sizeof(uint32_t), cudaMemcpyDeviceToHost);
		return cudaFree(gpuBuffer) == cudaSuccess;
	}



	/*
	Pixel has its x, y, z and depth information
	Note the the pixel has to be in the screen space already!
	!!! not suited for a single used!!!
	*/

	bool GPURasterizer::drawPixel(int x, int y, Pixel p) {
		gpuDrawPixelKernel<<<1, 1>>>(gpuBuffer, buffer->width, buffer->height, x, y, p);
		return true;
	}


	// This line looks like this ( | )
	/*bool Rasterizer::drawVerticalLine(int x, int y1, int y2, Pixel p) {
		int startY = min(min(y1, y2), buffer->height);
		int endY = max(max(y1, y2), 0);
		for (int y = startY; y <= endY; y++)
		{
			drawPixel(x, y, p);
		}
		return true;
	}*/


	// This line looks like this ( - )
	/*bool Rasterizer::drawHorizontalLine(int y, int x1, int x2, Pixel p) {
		int startX = min(min(x1, x2), buffer->width);
		int endX = max(max(x1, x2), 0);
		for (int x = startX; x <= endX; x++)
		{
			drawPixel(x, y, p);
		}
		return true;
	}*/


	// this line looks like ( \ )
	/*bool Rasterizer::drawFallRightLine(int x1, int y1, int x2, int y2, Pixel p) {

		int startX = min(min(x1, x2), buffer->width);
		int endX = max(max(x1, x2), 0);

		int startY = min(min(y1, y2), buffer->height);
		int endY = max(max(y1, y2), 0);

		int currY = startY;
		int prevY = startY;
		for (int x = startX; x <= endX; x++)
		{
			currY = startY + (x - startX) * (float(endY - startY) / float(endX - startX));
			drawVerticalLine(x, prevY, currY, p);
			prevY = currY;
		}

		return true;
	}*/


	// this line looks like ( / )
	/*bool Rasterizer::drawFallLeftLine(int x1, int y1, int x2, int y2, Pixel p) {

		int xMin = min(min(x1, x2), buffer->width);
		int xMax = max(max(x1, x2), 0);

		int yMax = max(max(y1, y2), 0);
		int yMin = min(min(y1, y2), buffer->height);

		int currY = yMax;
		int prevY = yMax;
		int f;
		for (int x = xMin; x <= xMax; x++)
		{
			currY = yMax - (x - xMin) * (float(yMax - yMin) / float(xMax - xMin));
			drawVerticalLine(x, prevY, currY, p);
			prevY = currY;
		}
		return true;
	}*/



	/*bool Rasterizer::drawLine(int x1, int y1, int x2, int y2, Pixel p) {

		if (x1 == x2)
		{
			return drawVerticalLine(x1, y1, y2, p);
		}
		else if (y1 == y2)
		{
			return drawHorizontalLine(y1, x1, x2, p);
		}
		else
		{
			int dx = x2 - x1;
			int dy = y2 - y1;

			if (dx > 0 && dy > 0)
			{
				return drawFallRightLine(x1, y1, x2, y2, p);
			}
			else if (dx < 0 && dy < 0) {
				return drawFallRightLine(x2, y2, x1, y1, p);
			}
			else if (dx > 0 && dy < 0) {
				return drawFallLeftLine(x1, y1, x2, y2, p);
			}
			else {
				return drawFallLeftLine(x2, y2, x1, y1, p);
			}

		}




		return true;
	}*/








}