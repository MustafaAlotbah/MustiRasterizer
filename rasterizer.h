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

	private:
		VideoBuffer* buffer;
	};



}