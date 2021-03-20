



/*
Rasterization
1- triangle setup
2- triangle traversal

*/







#include"application.h"
#include"pixel_processor.h"


namespace mge
{
	/*  (De)Constructing the rasterizer  */
	Application::Application(VideoBuffer* buffer) {
		this->videoBuffer = buffer;
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
		for (int y = 0; y < this->videoBuffer->height; y++)
		{
			for (int x = 0; x < this->videoBuffer->width; x++)
			{
				*((unsigned int*)this->videoBuffer->addr + y * this->videoBuffer->width + x) = x * time;
			}
		}






		/* successfully computed */
		return true;
	}






}


