

#include"rasterizer.h"




/*

Rasterizer job is to convert 2D vetices in screen space with a z-value
						into pixels on the screen

*/







namespace mge
{
	/*  (De)Constructing the rasterizer  */
	Application::Application(VideoBuffer* buffer) {
		this->buffer = buffer;
	}

	Application::~Application() {

	}








	/*  The work of the rasterizer  */

	bool Application::OnLoad() {
		return true;
	}


	float time = 0;

	bool Application::OnUpdate(float deltaTime) {


		/* dummy animation */
		time = time + deltaTime;
		if (time > 5)
		{
			time = 0;
		}
		for (int y = 0; y < this->buffer->height; y++)
		{
			for (int x = 0; x < this->buffer->width; x++)
			{
				*((unsigned int*)this->buffer->addr + y * this->buffer->width + x) = x * time;
			}
		}






		/* successfully computed */
		return true;
	}






}


