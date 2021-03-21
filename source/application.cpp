
/*
Application Space
*/








#include"application.h"
#include"pixel_processor.h"
#include"gpu_rasterizer.h"


namespace mge
{

	/*  Globals  */
	GPURasterizer rasterizer = 0;
	float time = 0;



	/*  The work of the rasterizer  */
	bool Application::OnLoad() {
		// initialize the rasterizer
		rasterizer = GPURasterizer(this->videoBuffer);


		return true;
	}














	bool Application::OnUpdate(float deltaTime) {


		/* dummy animation */
		time = time + deltaTime;
		
		if (time > 5)
		{
			time = 0; 
		}




		rasterizer.initSession();


		rasterizer.drawPixel(10, 10 * time, Pixel(0xFFFFFF));

		rasterizer.finishSession();




		/* successfully computed */
		return true;
	}










}


