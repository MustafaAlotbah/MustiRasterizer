

/*
Rasterization
1- triangle setup
2- triangle traversal

*/


#include"rasterizer.h"






namespace mge {





	Rasterizer::Rasterizer(VideoBuffer* buffer) {
		this->buffer = buffer;
	}
	Rasterizer::~Rasterizer() {

	}



	/*
	Pixel has its x, y, z and depth information
	Note the the pixel has to be in the screen space already!
	*/
	bool Rasterizer::drawPixel(int x, int y, Pixel p) {
		*((uint32_t*)buffer->addr + y * buffer->width + x) = p.color.value;
		return true;
	}


}