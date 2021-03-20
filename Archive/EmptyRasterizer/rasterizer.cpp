

#include"main.h"



namespace mge
{
	Rasterizer::Rasterizer(VideoBuffer buffer) {
		this->buffer = buffer;
	}
	
	Rasterizer::~Rasterizer() {

	}

	bool Rasterizer::OnLoad() {
		return true;
	}












	bool Rasterizer::OnUpdate(float deltaTime) {
		for (int y = 0; y < this->buffer.height; y++)
		{
			for (int x = 0; x < this->buffer.width; x++)
			{
				*((unsigned int*)this->buffer.addr + y * this->buffer.width + x) = x;
			}
		}
		return true;
	}


}


