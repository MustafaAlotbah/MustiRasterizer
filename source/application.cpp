
/*
Application Space
*/








#include"application.h"
#include"pixel_processor.h"
#include"rasterizer.h"


namespace mge
{

	/*  Globals  */
	Rasterizer rasterizer = 0;
	float time = 0;



	/*  The work of the rasterizer  */
	bool Application::OnLoad() {
		// initialize the rasterizer
		rasterizer = Rasterizer(this->videoBuffer);


		return true;
	}














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
				rasterizer.drawPixel(x, y, Pixel(x * time));
			}
			//rasterizer.drawVerticalLine(20, 20, 120, Pixel(0xFFFFFF));
			//rasterizer.drawHorizontalLine(20, 20, 120, Pixel(0xFFFFFF));

			rasterizer.drawLine( 100, 50, 150, 100, Pixel(0xFFFFFF));
			rasterizer.drawLine(150, 100, 100, 150, Pixel(0xFFFFFF));
			rasterizer.drawLine(100, 150, 50, 100, Pixel(0xFFFFFF));
			rasterizer.drawLine(50, 100, 100, 50, Pixel(0xFFFFFF));



		}






		/* successfully computed */
		return true;
	}










}


