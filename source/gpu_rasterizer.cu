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

	/*---------------------CUDA Globals-------------------------*/
	__device__ void* __gpu_buffer = 0;
	__device__ int __gpu_width = 0;
	__device__ int __gpu_height = 0;


	/*---------------------CUDA Functions-------------------------*/
	__global__ void gpuSetBuffer(void* gpuBuffer, int width, int height) {
		__gpu_buffer = gpuBuffer;
		__gpu_width = width;
		__gpu_height = height;
	}

	__device__ void gpuDrawPixel(int x, int y, Pixel p) {
		y = __gpu_height - y - 1;	// correct the orientation
		*((uint32_t*)__gpu_buffer + y * __gpu_width + x) = p.color.value;
	}

	__global__ void gpuClearBufferKernel(Pixel p) {
		gpuDrawPixel(threadIdx.x + blockDim.x * blockIdx.x % (__gpu_width * __gpu_height), __gpu_height -1, p);
	}

	__global__ void gpuDrawPixelKernel(int x, int y, Pixel p) {
		gpuDrawPixel(x, y, p);
	}

	// draws a vertical line
	__global__ void gpuDrawVerticalLine(int x, int yMin, Pixel p) {
		gpuDrawPixel(x, yMin + blockIdx.x * blockDim.x + threadIdx.x, p);
	}

	// draws a horizontal line
	__global__ void gpuDrawHorizontalLine(int y, int xMin, Pixel p) {
		gpuDrawPixel(xMin + blockIdx.x * blockDim.x + threadIdx.x, y, p);
	}

	// draws a line like this ( \ )
	__global__ void gpuDrawFallingRightLine(int y, int xMin, Pixel p) {
		gpuDrawPixel(xMin + blockIdx.x * blockDim.x + threadIdx.x, y, p);
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
		gpuClearBufferKernel <<<blcks, thrds >>> (p);
		return true;
	}


	bool GPURasterizer::initSession(Pixel p) {
		lastAllocatedSize = buffer->height * buffer->width;
		// allocate GPU Memory (will be disposed of by finishing the session
		cudaMalloc((void**)&gpuBuffer, sizeof(uint32_t) * lastAllocatedSize) ;
		gpuSetBuffer << <1, 1 >> > (gpuBuffer, buffer->width, buffer->height);
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
		gpuDrawPixelKernel<<<1, 1>>>(x, y, p);
		return true;
	}


	// This line looks like this ( | )
	bool GPURasterizer::drawVerticalLine(int x, int y1, int y2, Pixel p) {
		int startY = min(min(y1, y2), buffer->height);
		int endY = max(max(y1, y2), 0);
		uint32_t numPixels = endY + startY;
		gpuDrawVerticalLine<<<1, numPixels>>>(x, startY, p);

		return true;
	}


	// This line looks like this ( - )
	bool GPURasterizer::drawHorizontalLine(int y, int x1, int x2, Pixel p) {
		int startX = min(min(x1, x2), buffer->width);
		int endX = max(max(x1, x2), 0);
		uint32_t numPixels = endX + startX;
		gpuDrawHorizontalLine << <1, numPixels >> > (y, startX, p);
		return true;
	}


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