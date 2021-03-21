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





	bool GPURasterizer::drawPolygon(int count, int points[][2], PolygonMode mode, Pixel p) {
		for (int i = 0; i < count-1; i++)
		{
			drawLine(points[i][0], points[i][1], points[i+1][0], points[i + 1][1], p);
		}
		if (mode == Connected)
		{
			drawLine(points[count - 1][0], points[count - 1][1], points[0][0], points[0][1], p);
		}
		return true;
	}




	struct point
	{
		int x=0;
		int y=0;
	};

	struct line
	{
		point p1;
		point p2;
	};

	__device__ __host__ int getLineBound(int y, line _line) {
		float y1 = _line.p1.y, y2 = _line.p2.y;
		float x1 = _line.p1.x, x2 = _line.p2.x;
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
		//gpuDrawPixel(x, y, p);
		if (
			x > getLineBound(y, _leftLine)
			&& x > getLineBound(y, _leftLine2)
			&& x < getLineBound(y, _rightLine)
			&& x < getLineBound(y, _rightLine2)
			)
		{
			gpuDrawPixel(x, y, p);
		}
	}


	bool GPURasterizer::FillTriangle(int points[3][2], Pixel p) {

		int upperY = min(points[0][1], min(points[1][1], points[2][1]));
		int lowerY = max(points[0][1], max(points[1][1], points[2][1]));
		int leftX =  min(points[0][0], min(points[1][0], points[2][0]));
		int rightX = max(points[0][0], max(points[1][0], points[2][0]));



		drawLine(leftX, upperY, rightX, upperY, Pixel(0xf0f0f0));
		drawLine(leftX, lowerY, rightX, lowerY, Pixel(0xf0f0f0));
		drawLine(leftX, upperY, leftX, lowerY, Pixel(0xf0f0f0));
		drawLine(rightX, upperY, rightX, lowerY, Pixel(0xf0f0f0));

		int upperPoint[2] = { points[0][0], points[0][1] };
		int lowerPoint[2] = { points[0][0], points[0][1] };
		int leftPoint[2] = { points[0][0], points[0][1] };
		int rightPoint[2] = { points[0][0], points[0][1] };

		for (int i = 0; i < 3; i++)
		{
			if (points[i][1] < upperPoint[1])
			{
				upperPoint[0] = points[i][0];
				upperPoint[1] = points[i][1];
			}
			if (points[i][1] > lowerPoint[1])
			{
				lowerPoint[0] = points[i][0];
				lowerPoint[1] = points[i][1];
			}
			if (points[i][0] > rightPoint[0])
			{
				rightPoint[0] = points[i][0];
				rightPoint[1] = points[i][1];
			}
			if (points[i][0] < leftPoint[0])
			{
				leftPoint[0] = points[i][0];
				leftPoint[1] = points[i][1];
			}
		}

		/*int leftLine[2][2] = {
			{upperPoint[0],upperPoint[1] },
			{leftPoint[0],leftPoint[1] } };
		int rightLine[2][2] = {
			{upperPoint[0],upperPoint[1] },
			{rightPoint[0],rightPoint[1] } };
		int leftLine2[2][2] = {
			{lowerPoint[0],lowerPoint[1] },
			{leftPoint[0],leftPoint[1] } };
		int rightLine2[2][2] = {
			{lowerPoint[0],lowerPoint[1] },
			{rightPoint[0],rightPoint[1] } };*/

		point _up = { upperPoint[0], upperPoint[1] };
		point _left = { leftPoint[0], leftPoint[1] };
		point _down = { lowerPoint[0], lowerPoint[1] };
		point _right = { rightPoint[0], rightPoint[1] };
		line _leftLine = { _up, _left };
		line _rightLine = { _up, _right };
		line _leftLine2 = { _down, _left };
		line _rightLine2 = { _down, _right };



		int _area = (rightX - leftX) * (lowerY - upperY);
		int blcks = _area / 1024;
		FillTri<<<blcks+1, 1024>>>(leftX, rightX, upperY, lowerY, _leftLine, _rightLine, _leftLine2, _rightLine2, p);
		//for (int x = leftX; x < rightX; x++)
		//{
		//	for (int y = upperY; y < lowerY; y++)
		//	{
		//		if (
		//			x > getLineBound(y, leftLine) 
		//			//&& x < getlinebound(y, rightline)
		//			)
		//		{
		//			drawPixel(x, y, Pixel(0xff00ff));
		//		}
		//	}
		//}


		return true;




	}



}