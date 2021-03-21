
/*
Application Space
*/








#include"application.h"
#include"pixel_processor.h"
#include"gpu_rasterizer.h"
#include"algebra.h"
#include<cmath>

namespace mge
{

	/*  The work of the rasterizer  */
	bool Application::OnLoad() {
		// initialize the rasterizer
		rasterizer = GPURasterizer(this->videoBuffer);
		return true;
	}




	/*  Globals  */
	GPURasterizer rasterizer = 0;
	float time = 0;
	float angle = 0;
















	bool Application::OnUpdate(float deltaTime) {


		/* dummy animation */
		time = time + deltaTime;
		angle += 2 * deltaTime;
		matrix2d rotationMatrix(cos(angle), -sin(angle), sin(angle), cos(angle));
		
		if (time > 5)
		{
			time = 0; 
		}
		rasterizer.initSession(Pixel(0x300300));
		/*  Start Drawing scope */





		// set three points
		// set some center point

		vector2d vectors[3] = {
			vector2d(200 - 100, 200), 
			vector2d(200 + 100, 200 + 100),
			vector2d(200 - 100, 200 + 100) 
		};

		vector2d center(200, 200);

		// perform rotation
		for (int i = 0; i < 3; i++)
		{
			vectors[i] -= center;
			vectors[i] = rotationMatrix * vectors[i];
			vectors[i] += center;

		}


		// fill in the triangle

		rasterizer.FillTriangle(vectors, Pixel(0xf00f00));
		rasterizer.drawPolygon(3, vectors, GPURasterizer::PolygonMode::Connected, Pixel(0xffffff));
		for (int i = 0; i < 3; i++)
		{
			rasterizer.drawPixel(vectors[i].x, vectors[i].y, Pixel(0xaaaaFF));
		}


		/*  End Drawing Scope */





		rasterizer.finishSession();
		return true;
	}










}


