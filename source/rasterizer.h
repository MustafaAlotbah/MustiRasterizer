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
		virtual bool drawPixel(int x, int y, Pixel p);
		virtual bool drawVerticalLine(int x, int y1, int y2, Pixel p);
		virtual bool drawHorizontalLine(int y, int x1, int x2, Pixel p);
		virtual bool drawFallingRightLine(int x1, int y1, int x2, int y2, Pixel p);
		virtual bool drawFallingLeftLine(int x1, int y1, int x2, int y2, Pixel p);
		virtual bool drawLine(int x1, int y1, int x2, int y2, Pixel p);

	public:
		VideoBuffer* buffer;
	};



}