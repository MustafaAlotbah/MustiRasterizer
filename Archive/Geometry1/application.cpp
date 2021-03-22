
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


	/*  Global Graphics Classes  */
	GPURasterizer rasterizer = 0;


	/*  The work of the rasterizer  */
	bool Application::OnLoad() {
		// initialize the rasterizer
		rasterizer = GPURasterizer(this->videoBuffer);
		return true;
	}



	/*  Global Variables  */
	float time = 0;
	float angle = 0;









	float scale = 100;






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


		// cube 
		vector4d cube_vecs[8] = {
			vector4d(-1,	1,		0, 1) * 30.0f,
			vector4d(1,		1,		0, 1) * 30.0f,
			vector4d(1,		-1,		0, 1) * 30.0f,
			vector4d(-1,	-1,		0, 1) * 30.0f,

			vector4d(1,		1,		1, 1) * 30.0f,
			vector4d(-1,	1,		1, 1) * 30.0f,
			vector4d(-1,	-1,		1, 1) * 30.0f,
			vector4d(1,		-1,		1, 1) * 30.0f
		};



		/*vector4d vectors[3] = {
			vector4d(1,		1,		1, 1)*50.0f,
			vector4d(1,		-1,		1, 1)*50.0f,
			vector4d(-1,	1,		1, 1)*50.0f
		};
		

		float minX = min(vectors[0].x, min(vectors[1].x, vectors[2].x));
		float maxX = max(vectors[0].x, max(vectors[1].x, vectors[2].x));
		float midX = minX + (maxX - minX)/2;


		float minY = min(vectors[0].y, min(vectors[1].y, vectors[2].y));
		float maxY = max(vectors[0].y, max(vectors[1].y, vectors[2].y));
		float midY = minY + (maxY - minY) / 2;*/








		matrix2d rotationMatrix(cos(angle), -sin(angle), sin(angle), cos(angle));


		float _translate[4][4] = {
			{1, 0, 0, 5},
			{0, 1, 0, 0.001 * time},
			{0, 0, 1, 5},
			{0, 0, 0, 1}
		};
		matrix4d translate(_translate);

		float _rotate[4][4] = {
			{cos(angle),	-sin(angle),	0, 0},
			{sin(angle),	cos(angle),		0, 0},
			{0,				0,				1, 0},
			{0,				0,				0, 1}
		};
		matrix4d rotate(_rotate);


		float _shear[4][4] = {
			{1,	-0.5* time, 0, 0},
			{0,	1, 0, 0},
			{0, 0, 1, 0},
			{0, 0, 0, 1}
		};
		matrix4d shear(_shear);


		float _orthoPorj1[4][4] = {
			{1,	0,	-0.4, 0},
			{0,	1,	-0.4, 0},
			{0,	0,	0, 0},
			{0,	0,	0, 1}
		};
		matrix4d orthoPorj1(_orthoPorj1);



		// perform rotation
		for (int i = 0; i < 8; i++)
		{
			//vectors[i] -= center;
			//vectors[i] =  translate  * vectors[i];
			cube_vecs[i] = rotate * cube_vecs[i];
			//vectors[i] = shear * vectors[i];
			cube_vecs[i] = orthoPorj1 * cube_vecs[i];

			//vectors[i] += center;




			// translate to screen world
			 cube_vecs[i] = cube_vecs[i] *
				 vector4d(videoBuffer->width / 2 / scale, videoBuffer->height / 2 / scale, 1, 1) +
				 vector4d(videoBuffer->width / 2, videoBuffer->height / 2, 0, 0);

			/*vectors[i] = vectors[i] * 
				vector4d(videoBuffer->width / 2 / scale, videoBuffer->height / 2 / scale, 1, 1) + 
				vector4d(videoBuffer->width / 2 , videoBuffer->height / 2 , 0, 0);*/
		}



		vector4d cube_faces[6][4] = {
			{cube_vecs[0],	cube_vecs[1],	cube_vecs[2],	cube_vecs[3]},		//near
			{cube_vecs[4],	cube_vecs[5],	cube_vecs[6],	cube_vecs[7]},		//far
			{cube_vecs[0],	cube_vecs[1],	cube_vecs[4],	cube_vecs[5]},		//bottom
			{cube_vecs[2],	cube_vecs[3],	cube_vecs[6],	cube_vecs[7]},		//top
			{cube_vecs[3],	cube_vecs[0],	cube_vecs[5],	cube_vecs[6]},	//left
			{cube_vecs[1],	cube_vecs[2],	cube_vecs[7],	cube_vecs[4]},	//right
		};


		// fill in the triangle
		//rasterizer.FillTriangle(_vectors, Pixel(0xf00f00));

		for (int i = 0; i < 6; i++)
		{
			vector2d _vectors[4] = {
				vector2d(cube_faces[i][0]),
				vector2d(cube_faces[i][1]),
				vector2d(cube_faces[i][2]),
				vector2d(cube_faces[i][3])
			};
			rasterizer.drawPolygon(4, _vectors, GPURasterizer::PolygonMode::Connected, Pixel(0xffffff));
		}


		/*for (int i = 0; i < 3; i++)
		{
			rasterizer.drawPixel(_vectors[i].x,_vectors[i].y, Pixel(0xaaaaFF));
		}*/















		/*  End Drawing Scope */
		rasterizer.finishSession();
		return true;
	}










}


