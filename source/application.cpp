
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
		angle += 2 * deltaTime;
		
		if (time > 5)
		{
			time = 0; 
		}
		rasterizer.initSession(Pixel(0x300300));
		/*  Start Drawing scope */





		// set three points
		int points[3][2] = { {200-100, 200}, {200+100, 200+100}, {200-100, 200+100} };
		// set some center point
		int center[2] = { 200, 200 };

		// perform rotation
		for (int i = 0; i < 3; i++)
		{
			int x = points[i][0] - center[0];
			int y = points[i][1] - center[1];
			points[i][0] = x * cos(angle) - y * sin(angle);
			points[i][1] = x * sin(angle) + y * cos(angle);
			points[i][0] += center[0];
			points[i][1] += center[1];
		}

		// fill in the triangle
		rasterizer.FillTriangle(points, Pixel(0xf00f00));
		rasterizer.drawPolygon(3, points, GPURasterizer::PolygonMode::Connected ,Pixel(0xffffff));

		for (int i = 0; i < 3; i++)
		{
			rasterizer.drawPixel(points[i][0], points[i][1], Pixel(0xaaaaFF));
		}


		/*  End Drawing Scope */





		rasterizer.finishSession();
		return true;
	}










}


