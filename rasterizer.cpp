

#include"rasterizer.h"



namespace mge
{
	/*  (De)Constructing the rasterizer  */
	Rasterizer::Rasterizer(VideoBuffer buffer) {
		this->buffer = buffer;
	}
	
	Rasterizer::~Rasterizer() {

	}








	/*  The work of the rasterizer  */

	bool Rasterizer::OnLoad() {
		return true;
	}


	float time = 0;

	bool Rasterizer::OnUpdate(float deltaTime) {


		/* dummy animation */
		time = time + deltaTime;
		if (time > 5)
		{
			time = 0;
		}
		for (int y = 0; y < this->buffer.height; y++)
		{
			for (int x = 0; x < this->buffer.width; x++)
			{
				*((unsigned int*)this->buffer.addr + y * this->buffer.width + x) = x * time;
			}
		}






		/* successfully computed */
		return true;
	}






}


