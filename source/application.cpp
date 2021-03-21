
/*
Application Space
*/








#include"application.h"
#include"pixel_processor.h"
#include"gpu_rasterizer.h"
#include<cmath>

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












	float angle = 0;

	bool Application::OnUpdate(float deltaTime) {


		/* dummy animation */
		time = time + deltaTime;
		
		if (time > 5)
		{
			time = 0; 
		}
		rasterizer.initSession(Pixel(0x300300));
		/*  Start Drawing scope */






		//rasterizer.drawVerticalLine(30, 30, 60, Pixel(0xF00F00));
		//rasterizer.drawHorizontalLine(30, 30, 60, Pixel(0xF00F00));


		//rasterizer.drawFallingRightLine(30, 30, 60, 60, Pixel(0xF00F00));
		//rasterizer.drawFallingRightLine(30, 30, 100, 60, Pixel(0xF00F00));
		//rasterizer.drawFallingRightLine(30, 30, 60, 100, Pixel(0xF00F00));

		//rasterizer.drawFallingLeftLine(130, 160, 160, 130, Pixel(0xF00F00));
		//rasterizer.drawFallingLeftLine(130, 160, 260, 130, Pixel(0xF00F00));
		//rasterizer.drawFallingLeftLine(130, 160, 160, 60, Pixel(0xF00F00));


		int points[3][2] = { {200-100, 200}, {200+100, 200+100}, {200-100, 200+100} };

		int center[2] = { 200, 200 };

		angle +=  deltaTime;

		for (int i = 0; i < 3; i++)
		{
			int x = points[i][0] - center[0];
			int y = points[i][1] - center[1];
			points[i][0] = x * cos(angle) - y * sin(angle);
			points[i][1] = x * sin(angle) + y * cos(angle);
			points[i][0] += center[0];
			points[i][1] += center[1];
		}

		rasterizer.FillTriangle(points, Pixel(0xf00f00));
		rasterizer.drawPolygon(3, points, GPURasterizer::PolygonMode::Connected ,Pixel(0xf00f00));

		for (int i = 0; i < 3; i++)
		{
			rasterizer.drawPixel(points[i][0], points[i][1], Pixel(0xaaaaFF));
		}


		/*  End Drawing Scope */





		rasterizer.finishSession();
		return true;
	}










}


