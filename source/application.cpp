
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
		rasterizer.initSession(Pixel(0x300300));
		/*  Start Drawing scope */





		rasterizer.drawPixel(10, 10 * time, Pixel(0xFFFFFF));
		rasterizer.drawPixel(30, 10 * time, Pixel(0xFFFFFF));
		rasterizer.drawVerticalLine(30, 30, 60, Pixel(0xF00F00));
		rasterizer.drawHorizontalLine(30, 30, 60, Pixel(0xF00F00));




		/*  End Drawing Scope */





		rasterizer.finishSession();
		return true;
	}










}


