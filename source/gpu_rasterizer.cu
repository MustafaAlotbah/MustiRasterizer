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

	__device__ __host__ void gpuDrawPixel(int x, int y, Pixel p) {
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

	// with dx > dy (operating on x-axis)
	__global__ void gpuDrawFallingRightLineDx(int xMin, int yMin, float dx , float dy, Pixel p) {
		float y = yMin + (threadIdx.x) * (dy / dx);
		gpuDrawPixel(threadIdx.x + xMin, y, p);
	}

	// with dx < dy (operating on y-axis)
	__global__ void gpuDrawFallingRightLineDy(int xMin, int yMin, float dx, float dy, Pixel p) {
		float x = xMin + (threadIdx.x) * (dx / dy);
		gpuDrawPixel(x, threadIdx.x + yMin, p);
	}


	// with dx > dy (operating on x-axis)
	__global__ void gpuDrawFallingLeftLineDx(int xMin, int yMax, float dx, float dy, Pixel p) {
		float y = yMax - (threadIdx.x) * (dy / dx);
		gpuDrawPixel(threadIdx.x + xMin, y, p);
	}


	// with dx > dy (operating on x-axis)
	__global__ void gpuDrawFallingLeftLineDy(int xMax, int yMin, float dx, float dy, Pixel p) {
		float x = xMax - (threadIdx.x) * (dx / dy);
		gpuDrawPixel(x, threadIdx.x + yMin, p);
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
		uint32_t numPixels = endY - startY;
		gpuDrawVerticalLine<<<1, numPixels>>>(x, startY, p);

		return true;
	}


	// This line looks like this ( - )
	bool GPURasterizer::drawHorizontalLine(int y, int x1, int x2, Pixel p) {
		int startX = min(min(x1, x2), buffer->width);
		int endX = max(max(x1, x2), 0);
		uint32_t numPixels = endX - startX;
		gpuDrawHorizontalLine <<<1, numPixels >>> (y, startX, p);
		return true;
	}


	// this line looks like ( \ )
	bool GPURasterizer::drawFallingRightLine(int x1, int y1, int x2, int y2, Pixel p) {

		int minX = min(min(x1, x2), buffer->width);
		int maxX = max(max(x1, x2), 0);

		int minY = min(min(y1, y2), buffer->height);
		int maxY = max(max(y1, y2), 0);

		float dx = maxX - minX;
		float dy = maxY - minY;

		if (dx >= dy) 
			gpuDrawFallingRightLineDx <<<1, dx>>> (minX, minY, dx, dy, p);
		else 
			gpuDrawFallingRightLineDy <<<1, dy >>> (minX, minY, dx, dy, p);


		return true;
	}


	// this line looks like ( / )
	bool GPURasterizer::drawFallingLeftLine(int x1, int y1, int x2, int y2, Pixel p) {

		int xMin = min(min(x1, x2), buffer->width);
		int xMax = max(max(x1, x2), 0);

		int yMin = min(min(y1, y2), buffer->height);
		int yMax = max(max(y1, y2), 0);

		float dx = xMax - xMin;
		float dy = yMax - yMin;


		if (dx >= dy)
			gpuDrawFallingLeftLineDx <<<1, dx >>> (xMin, yMax, dx, dy, p);
		else
			gpuDrawFallingLeftLineDy <<<1, dy >>> (xMax, yMin, dx, dy, p);

		return true;


	}


	bool GPURasterizer::drawLine(vector2d a, vector2d b, Pixel p) {
		return Rasterizer::drawLine((int)a.x, (int)a.y, (int)b.x, (int)b.y, p);
	}



	bool GPURasterizer::drawPolygon(int count, vector2d points[], PolygonMode mode, Pixel p) {
		for (int i = 0; i < count-1; i++)
		{
			drawLine(points[i], points[i+1], p);
		}
		if (mode == Connected)
		{
			drawLine(points[count - 1], points[0], p);
		}
		return true;
	}




	struct line
	{
		vector2d p1;
		vector2d p2;
	};

	__device__ __host__ int getLineBound(int x, int y, line _line) {
		float y1 = _line.p1.y, y2 = _line.p2.y;
		float x1 = _line.p1.x, x2 = _line.p2.x;
		if (y1 == y2)
		{
			return x;
		}
		return x1 + ((y - y1) * (x2 - x1)) / (y2 - y1 + 0.0001f);
		return 1;
	}

	__global__ void FillTri(int leftX, int rightX, int upperY, int lowerY,
		line _leftLine, line _rightLine, line _leftLine2, line _rightLine2,
		Pixel p) {
		int width = rightX - leftX;
		int height = lowerY - upperY;
		int i = (blockIdx.x * blockDim.x + threadIdx.x) % (width * height);
		int x = leftX + (i % width);
		int y = upperY + (i / width);
		if (
			x >= getLineBound(x, y, _leftLine)
			&& x >= getLineBound(x, y, _leftLine2)
			&& x <= getLineBound(x, y, _rightLine)
			&& x <= getLineBound(x, y, _rightLine2)
			)
		{
			gpuDrawPixel(x, y, p);
		}
	}


	bool GPURasterizer::FillTriangle(vector2d points[3], Pixel p) {

		int upperY = min(points[0].y, min(points[1].y, points[2].y));
		int lowerY = max(points[0].y, max(points[1].y, points[2].y));
		int leftX =  min(points[0].x, min(points[1].x, points[2].x));
		int rightX = max(points[0].x, max(points[1].x, points[2].x));


		// draw outlines?
		//drawLine(leftX, upperY, rightX, upperY, Pixel(0xf0f0f0));
		//drawLine(leftX, lowerY, rightX, lowerY, Pixel(0xf0f0f0));
		//drawLine(leftX, upperY, leftX, lowerY, Pixel(0xf0f0f0));
		//drawLine(rightX, upperY, rightX, lowerY, Pixel(0xf0f0f0));

		vector2d upperPoint(points[0].x, points[0].y);
		vector2d lowerPoint(points[0].x, points[0].y );
		vector2d leftPoint(points[0].x, points[0].y);
		vector2d rightPoint(points[0].x, points[0].y);



		for (int i = 0; i < 3; i++)
		{
			if (points[i].y < upperPoint.y)
			{
				upperPoint = points[i];
			}
			if (points[i].y > lowerPoint.y)
			{
				lowerPoint = points[i];
			}
			if (points[i].x > rightPoint.x)
			{
				rightPoint = points[i];
			}
			if (points[i].x < leftPoint.x)
			{
				leftPoint = points[i];
			}
		}

		vector2d _up	(upperPoint.x,	upperPoint.y );
		vector2d _left	(leftPoint.x,	leftPoint.y  );
		vector2d _down	(lowerPoint.x,	lowerPoint.y );
		vector2d _right (rightPoint.x,	rightPoint.y );

		if (_left.x == _up.x && _left.y == _up.y)
		{
			for (volatile int i = 0; i < 3; i++)
			{
				if (/*points[i][0] <= _left.x &&*/ points[i].y >= _left.y)
				{
					_left = points[i];
				}
			}
		}


		line _leftLine = { _up, _left };
		line _rightLine = { _up, _right };
		line _leftLine2 = { _down, _left };
		line _rightLine2 = { _down, _right };


		int _area = (rightX - leftX) * (lowerY - upperY);
		int blcks = _area / 1024;
		FillTri<<<blcks+1, 1024>>>(leftX, rightX, upperY, lowerY, _leftLine, _rightLine, _leftLine2, _rightLine2, p);
	

		drawLine(_leftLine.p1, _leftLine.p2, Pixel(0xff0000));


		return true;




	}



}