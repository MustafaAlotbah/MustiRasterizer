/*
	Rasterizer in Cuda C

	Use this way:
	rasterizer.initSession();			<- always include these
	...
	rasterizer.drawLine or whatever ();
	...
	rasterizer.finishSession();			<- always include these

*/

/*
	Functions implemented here on this pattern (this allows more dynamic use of all GPU functions
	__device__	gpuDoSomething			does everything done on the GPU
	__kernel__	gpuDoSomethingKernel	xs and ys and calls device
	__host__	doSomething				calls kernel's instances

*/

#define DEBUG

#include "gpu_rasterizer.h"
#include "geometry_processor.h"


namespace mge {

	/*---------------------CUDA Globals-------------------------*/
	__device__ void* __gpu_buffer = 0;
	__device__ int __gpu_width = 0;
	__device__ int __gpu_height = 0;


	/*---------------------CUDA Functions-------------------------*/
	__global__ void gpuSetBufferKernel(void* gpuBuffer, int width, int height) {
		__gpu_buffer = gpuBuffer;
		__gpu_width = width;
		__gpu_height = height;
	}



	__global__ void gpuClearBufferKernel(Pixel p) {
		int i = threadIdx.x + blockDim.x * blockIdx.x % (__gpu_width * __gpu_height);
		*((uint32_t*)__gpu_buffer + i) = p.color.value;
	}







	/*---------------------Rasterizer Interface-------------------------*/

	cudaRasterizer::cudaRasterizer(VideoBuffer* buffer) :
	Rasterizer(buffer)
	{
		cudaStatus = cudaSetDevice(0);
	}


	cudaRasterizer::~cudaRasterizer(){
	}


	bool cudaRasterizer::clearBuffer(Pixel p) {

		dim3 thrds(1000, 1);
		dim3 blcks(lastAllocatedSize / 1000 + 1, 1);
		gpuClearBufferKernel <<<blcks, thrds >>> (p);
		return true;
	}


	bool cudaRasterizer::initSession(Pixel p) {
		lastAllocatedSize = buffer->height * buffer->width;
		// allocate GPU Memory (will be disposed of by finishing the session
		cudaMalloc((void**)&gpuBuffer, sizeof(uint32_t) * lastAllocatedSize) ;
		gpuSetBufferKernel << <1, 1 >> > (gpuBuffer, buffer->width, buffer->height);
		clearBuffer(p);
		return true;
	}


	bool cudaRasterizer::finishSession() {
		// copy GPU buffer into CPU buffer and free GPU buffer
		cudaMemcpy(buffer->addr, gpuBuffer, lastAllocatedSize * sizeof(uint32_t), cudaMemcpyDeviceToHost);
		return cudaFree(gpuBuffer) == cudaSuccess;
	}



	/*
	Pixel has its x, y, z and depth information
	Note the the pixel has to be in the screen space already!
	!!! not suited for a single used!!!
	*/


	/*START*/		


	/*######################### Public: Draw Pixel ##############################*/
	// only on device since the device memory will be copied in the end
	__device__ void gpuDrawPixel(int x, int y, Pixel p) {
		y = __gpu_height - y - 1;	// correct the orientation
		if (y < __gpu_height && y >= 0 && x >= 0 && x <= __gpu_width)
			*((uint32_t*)__gpu_buffer + y * __gpu_width + x) = p.color.value;
	}


	__global__ void gpuDrawPixelKernel(int x, int y, Pixel p) {
		gpuDrawPixel(x, y, p);
	}

	bool cudaRasterizer::drawPixel(int x, int y, Pixel p) {
		gpuDrawPixelKernel<<<1, 1>>>(x, y, p);
		return true;
	}

	/*######################### Public: Draw Vertical Line ##############################*/
	// This line looks like this ( | )

	// !Only use when kernel can't be used (this has a loop)
	__device__ void gpuDrawVerticalLine(int x, int yMin, int yMax, Pixel p) {
		for (int y = yMin; y < yMax; y++)
		{
			gpuDrawPixel(x, y, p);
		}
	}
	
	__global__ void gpuDrawVerticalLineKernel(int x, int yMin, Pixel p) {
		gpuDrawPixel(x, yMin + blockIdx.x * blockDim.x + threadIdx.x, p);
	}

	bool cudaRasterizer::drawVerticalLine(int x, int y1, int y2, Pixel p) {
		int startY = min(min(y1, y2), buffer->height);
		int endY = max(max(y1, y2), 0);
		uint32_t numPixels = endY - startY;
		gpuDrawVerticalLineKernel << <1, numPixels >> > (x, startY, p);

		return true;
	}

	/*######################### Public: Draw Horizontal Line ##############################*/
	// This line looks like this ( - )

	// !Only use when kernel can't be used (this has a loop)
	__global__ void gpuDrawHorizontalLine(int y, int xMin, int xMax, Pixel p) {
		for (int x = xMin; x < xMax; x++)
		{
			gpuDrawPixel(xMin + x, y, p);
		}
	}
	// draws a horizontal line
	__global__ void gpuDrawHorizontalLineKernel(int y, int xMin, Pixel p) {
		gpuDrawPixel(xMin + blockIdx.x * blockDim.x + threadIdx.x, y, p);
	}

	bool cudaRasterizer::drawHorizontalLine(int y, int x1, int x2, Pixel p) {
		int startX = min(min(x1, x2), buffer->width);
		int endX = max(max(x1, x2), 0);
		uint32_t numPixels = endX - startX;
		gpuDrawHorizontalLineKernel << <1, numPixels >> > (y, startX, p);
		return true;
	}


	/*######################### Public: Draw Falling Line (\) ##############################*/
	// this line looks like ( \ )

	// TODO: __device__

	// with dx > dy (operating on x-axis)
	__global__ void gpuDrawFallingLineDxKernel(int xMin, int yMin, float dx, float dy, Pixel p) {
		float y = yMin + (threadIdx.x) * (dy / dx);
		gpuDrawPixel(threadIdx.x + xMin, y, p);
	}


	// with dx < dy (operating on y-axis)
	__global__ void gpuDrawFallingLineDyKernel(int xMin, int yMin, float dx, float dy, Pixel p) {
		float x = xMin + (threadIdx.x) * (dx / dy);
		gpuDrawPixel(x, threadIdx.x + yMin, p);
	}


	bool cudaRasterizer::drawFallingLine(int x1, int y1, int x2, int y2, Pixel p) {

		int minX = min(min(x1, x2), buffer->width);
		int maxX = max(max(x1, x2), 0);

		int minY = min(min(y1, y2), buffer->height);
		int maxY = max(max(y1, y2), 0);

		float dx = maxX - minX;
		float dy = maxY - minY;

		if (dx >= dy)
			gpuDrawFallingLineDxKernel << <1, dx >> > (minX, minY, dx, dy, p);
		else
			gpuDrawFallingLineDyKernel << <1, dy >> > (minX, minY, dx, dy, p);


		return true;
	}


	/*######################### Public: Draw Rising Line (/) ##############################*/
	// this line looks like ( / )

	// TODO: __device__


	// with dx > dy (operating on x-axis)
	__global__ void gpuDrawFallingLineDx(int xMin, int yMax, float dx, float dy, Pixel p) {
		float y = yMax - (threadIdx.x) * (dy / dx);
		gpuDrawPixel(threadIdx.x + xMin, y, p);
	}


	// with dx > dy (operating on x-axis)
	__global__ void gpuDrawFallingLineDy(int xMax, int yMin, float dx, float dy, Pixel p) {
		float x = xMax - (threadIdx.x) * (dx / dy);
		gpuDrawPixel(x, threadIdx.x + yMin, p);
	}


	bool cudaRasterizer::drawFallingLeftLine(int x1, int y1, int x2, int y2, Pixel p) {

		int xMin = min(min(x1, x2), buffer->width);
		int xMax = max(max(x1, x2), 0);

		int yMin = min(min(y1, y2), buffer->height);
		int yMax = max(max(y1, y2), 0);

		float dx = xMax - xMin;
		float dy = yMax - yMin;


		if (dx >= dy)
			gpuDrawFallingLineDx << <1, dx >> > (xMin, yMax, dx, dy, p);
		else
			gpuDrawFallingLineDy << <1, dy >> > (xMax, yMin, dx, dy, p);

		return true;


	}


	bool cudaRasterizer::drawLine(vector2d a, vector2d b, Pixel p) {
		return Rasterizer::drawLine((int)a.x, (int)a.y, (int)b.x, (int)b.y, p);
	}


	/*######################### Public: Draw a Polygon ##############################*/
	// shouldn't be really used for shapes but good for plotting paths and characters

	bool cudaRasterizer::drawPolygon(int count, vector2d points[], PolygonMode mode, Pixel p) {
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

	bool cudaRasterizer::drawPolygon(std::vector<vector2d> points, PolygonMode mode, Pixel p) {
		for (int i = 0; i < points.size() - 1; i++)
		{
			drawLine(points[i], points[i + 1], p);
		}
		if (mode == Connected)
		{
			drawLine(points[points.size() - 1], points[0], p);
		}
		return true;
	}

	/*######################### Public: Draw a Triangle ##############################*/


	/*END*/


	__device__ void FillTriangleDevice(int leftX, int rightX, int upperY, int lowerY, vector2d v1, vector2d v2, vector2d base, Pixel p) {

		int width = rightX - leftX;
		int height = lowerY - upperY;

		int i = (blockIdx.x * blockDim.x + threadIdx.x) % (width * height);
		int x = leftX + (i % width);
		int y = upperY + (i / width);

		float det = v1.x * v2.y - v2.x * v1.y;

		float a = (v2.y * (x - base.x) - v2.x * (y - base.y)) / det;
		float b = (-v1.y * (x - base.x) + v1.x * (y - base.y)) / det;

		if (a >= 0 && b >= 0 && a + b <= 1)
		{
			gpuDrawPixel(x, y, p);
		}
	}

	__global__ void FillTriangleKernel(int leftX, int rightX, int upperY, int lowerY, vector2d v1, vector2d v2, vector2d base, Pixel p) {

		FillTriangleDevice(leftX, rightX, upperY, lowerY, v1, v2, base, p);
	}

	//  Thanks to Raman for help
	// Fills a triangle with a solid color
	bool cudaRasterizer::FillTriangle(vector2d points[3], Pixel p) {


		int upperY = min(points[0].y, min(points[1].y, points[2].y));
		int lowerY = max(points[0].y, max(points[1].y, points[2].y));
		int leftX = min(points[0].x, min(points[1].x, points[2].x));
		int rightX = max(points[0].x, max(points[1].x, points[2].x));


		vector2d v1 = points[0] - points[1];
		vector2d v2 = points[2] - points[1];

		int _area = (rightX - leftX) * (lowerY - upperY);
		int blcks = _area / 1024;
		FillTriangleKernel <<<blcks + 1, 1024 >>> (leftX, rightX, upperY, lowerY, points[0] - points[1], points[2] - points[1], points[1], p);
		return true;
	}






	__global__ void gpuDrawMeshKernel(std::vector<path4d>* MeshPtr, Pixel p) {
		gpuDrawPixel(20, 20, p);
	}


	// THIS BELONGS TO GEOMETRY!
	bool cudaRasterizer::drawMesh(Mesh m, Pixel p) {
		// copy mesh to GPU
		// copy triangles to GPU
		void* triags;
		int size = m.triags.size() * sizeof(m.triags[0]);

		//cudaMalloc(triags, )
		

		// perform Draw Mesh
		gpuDrawMeshKernel <<<1, 1 >>> (0, p);

		// wait

		// free mesh

		return true;
	}

}