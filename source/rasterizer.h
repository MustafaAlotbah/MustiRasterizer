#pragma once



#include"main.h"
#include"pixel_processor.h"


namespace mge {


	class Rasterizer
	{
	public:
		Rasterizer(VideoBuffer* buffer);
		~Rasterizer();

	public:
		bool drawPixel(int x, int y, Pixel p);
		bool drawVerticalLine(int x, int y1, int y2, Pixel p);
		bool drawHorizontalLine(int y, int x1, int x2, Pixel p);
		bool drawFallRightLine(int x1, int y1, int x2, int y2, Pixel p);
		bool drawFallLeftLine(int x1, int y1, int x2, int y2, Pixel p);
		bool drawLine(int x1, int y1, int x2, int y2, Pixel p);

	private:
		VideoBuffer* buffer;
	};



}