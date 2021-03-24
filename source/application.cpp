
/*
Application Space
*/







#include"application.h"
#include"pixel_processor.h"
#include"gpu_rasterizer.h"
#include"algebra.h"
#include"print.h"
#include<cmath>

namespace mge
{


	/*  Global Graphics Classes  */
	cudaRasterizer rasterizer = 0;


	/*  Global Variables  */
	float time = 0;
	float angle = 0;
	Mesh m;
	dyn_Mesh dyn_m;
	Mesh _m;

	/*  Load Graphics classes  */
	bool Application::OnLoad() {
		// initialize the rasterizer
		rasterizer = cudaRasterizer(this->videoBuffer);



		m.loadFromFile("./teapot.obj");
		_m = m;
		//dyn_m.loadFromFile("./huisje.obj");
		return true;
	}


	/* Destroy Graphics classes */
	bool Application::OnDestroy() {

		return true;
	}










	float scale = 100;


	float _angle = 0;
	bool tdone = false;


	bool Application::OnUpdate(float deltaTime) {


		/* dummy animation */
		time = time + deltaTime;
		//angle = angle + 2 * deltaTime;
		angle = 3.14 *  sin( time );
		/*if (time > 5)
		{
			time = 0; 
		}*/
		rasterizer.initSession(Pixel(0x202020));
		/*  Start Drawing scope */


		// cube 
		vector4d cube_vecs[8] = {
			vector4d(1,		1,		0, 1) * 1.0f,
			vector4d(1,		-1,		0, 1) * 1.0f,
			vector4d(-1,	-1,		0, 1) * 1.0f,
			vector4d(-1,	1,		0, 1) * 1.0f,


			vector4d(1,		1,		1, 1) * 1.0f,
			vector4d(1,		-1,		1, 1) * 1.0f,
			vector4d(-1,	-1,		1, 1) * 1.0f,
			vector4d(-1,	1,		1, 1) * 1.0f

		};

		//vector4d cube_faces[6][4] = {
		//	{cube_vecs[0],	cube_vecs[1],	cube_vecs[2],	cube_vecs[3]},		//near
		//	{cube_vecs[4],	cube_vecs[5],	cube_vecs[6],	cube_vecs[7]},		//far
		//	{cube_vecs[0],	cube_vecs[1],	cube_vecs[4],	cube_vecs[5]},		//bottom
		//	{cube_vecs[2],	cube_vecs[3],	cube_vecs[6],	cube_vecs[7]},		//top
		//	{cube_vecs[3],	cube_vecs[0],	cube_vecs[5],	cube_vecs[6]},	//left
		//	{cube_vecs[1],	cube_vecs[2],	cube_vecs[7],	cube_vecs[4]},	//right
		//};


		std::vector<path4d> cube_faces;

		// pointing to screen (clockwise)
		cube_faces.push_back(path4d(
			cube_vecs[0], cube_vecs[1], cube_vecs[2]
		));
		cube_faces.push_back(path4d(
			cube_vecs[0], cube_vecs[2], cube_vecs[3]
		));

		// pointing away from the screen (counter-clockwise from out perspective)
		cube_faces.push_back(path4d(
			cube_vecs[4], cube_vecs[6], cube_vecs[5]
		));
		cube_faces.push_back(path4d(
			cube_vecs[4], cube_vecs[7], cube_vecs[6]
		));
		// pointing right  
		cube_faces.push_back(path4d(
			cube_vecs[1], cube_vecs[0], cube_vecs[4]
		));
		cube_faces.push_back(path4d(
			cube_vecs[4], cube_vecs[5], cube_vecs[1]
		));
		//
		// pointing left
		cube_faces.push_back(path4d(
			cube_vecs[2], cube_vecs[6], cube_vecs[3]
		));
		cube_faces.push_back(path4d(
			cube_vecs[3], cube_vecs[6], cube_vecs[7]
		));

		//
		// top
		cube_faces.push_back(path4d(
			cube_vecs[0], cube_vecs[3], cube_vecs[4]
		));
		cube_faces.push_back(path4d(
			cube_vecs[3], cube_vecs[7], cube_vecs[4]
		));

		//
		// bottom
		cube_faces.push_back(path4d(
			cube_vecs[5], cube_vecs[2], cube_vecs[1]
		));
		cube_faces.push_back(path4d(
			cube_vecs[2], cube_vecs[5], cube_vecs[6]
		));








		
		/*
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

		float _rotatez[4][4] = {
			{cos(angle/2),	0,	-sin(angle/2), 0},
			{0,				1,				0, 0},
			{sin(angle/2),	0,		cos(angle/2), 0},
			{0,				0,				0, 1}
		};
		matrix4d rotatez(_rotatez);

		float _rotatey[4][4] = {
			{1,					0,				0,					0},
			{0,					cos(angle / 2),	-sin(angle / 2),	0},
			{0,					sin(angle / 2),	cos(angle / 2),		0},
			{0,					0,				0,					1}
		};
		matrix4d rotatey(_rotatey);


		float _shear[4][4] = {
			{1,	-0.5* time, 0, 0},
			{0,	1, 0, 0},
			{0, 0, 1, 0},
			{0, 0, 0, 1}
		};
		matrix4d shear(_shear);


		float _orthoPorj1[4][4] = {
			{1,	0,	0.3, 0},
			{0,	1,	-0.2, 0},
			{0,	0,	1, 0},
			{0,	0,	0, 1}
		};
		matrix4d orthoPorj1(_orthoPorj1);


		float _perspectiveProj1[4][4] = {
			{1,	0,	-0.3, 0},
			{0,	1,	-0.2, 0},
			{0,	0,	0, 0},
			{0,	0,	0, 1}
		};
		matrix4d perspectiveProj1(_perspectiveProj1);

		// perform rotation
		for (int i = 0; i < 8; i++)
		{
			//vectors[i] -= center;
			//vectors[i] =  translate  * vectors[i];
			//vectors[i] = shear * vectors[i];

			//cube_vecs[i] = rotatez * cube_vecs[i];

			//cube_vecs[i] = orthoPorj1 * cube_vecs[i];

			//vectors[i] += center;




			// translate to screen world
			 /*cube_vecs[i] = cube_vecs[i] *
				 vector4d(videoBuffer->width / 2 / scale, videoBuffer->height / 2 / scale, 1, 1) +
				 vector4d(videoBuffer->width / 2, videoBuffer->height / 2, 0, 0);*/

			/*vectors[i] = vectors[i] * 
				vector4d(videoBuffer->width / 2 / scale, videoBuffer->height / 2 / scale, 1, 1) + 
				vector4d(videoBuffer->width / 2 , videoBuffer->height / 2 , 0, 0);*/
		}



		//vector4d cube_faces[6][4] = {
		//	{cube_vecs[0],	cube_vecs[1],	cube_vecs[2],	cube_vecs[3]},		//near
		//	{cube_vecs[4],	cube_vecs[5],	cube_vecs[6],	cube_vecs[7]},		//far
		//	{cube_vecs[0],	cube_vecs[1],	cube_vecs[4],	cube_vecs[5]},		//bottom
		//	{cube_vecs[2],	cube_vecs[3],	cube_vecs[6],	cube_vecs[7]},		//top
		//	{cube_vecs[3],	cube_vecs[0],	cube_vecs[5],	cube_vecs[6]},	//left
		//	{cube_vecs[1],	cube_vecs[2],	cube_vecs[7],	cube_vecs[4]},	//right
		//};







		// fill in the triangle
		//rasterizer.FillTriangle(_vectors, Pixel(0xf00f00));

		/*for (int i = 0; i < 6; i++)
		{
			vector2d _vectors[4] = {
				vector2d(cube_faces[i][0]),
				vector2d(cube_faces[i][1]),
				vector2d(cube_faces[i][2]),
				vector2d(cube_faces[i][3])
			};
			rasterizer.drawPolygon(4, _vectors, GPURasterizer::PolygonMode::Connected, Pixel(0xffffff));
		}*/

		
		////////////////////////////// ROTATING TRIANGLE //////////////////////////////////
		//vector4d vectors[3] = {
		//	vector4d(1,	0,		1, 1) * 50.0f,
		//	vector4d(0,	0.8,		1, 1) * 50.0f,
		//	vector4d(0,	0,		1, 1) * 50.0f
		//};
		//	 _angle += deltaTime;
		///*float _rotate2[4][4] = {
		//	{cos(90 * 3.1415 / 180),	-sin(90 * 3.1415 / 180),	0, 0},
		//	{sin(90 * 3.1415 / 180),	 cos(90 * 3.1415 / 180 ),		0, 0},
		//	{0,				0,				1, 0},
		//	{0,				0,				0, 1}
		//};*/
		//float _rotate2[4][4] = {
		//	{cos(_angle),	-sin(_angle),	0, 0},
		//	{sin(_angle),	 cos(_angle),		0, 0},
		//	{0,				0,				1, 0},
		//	{0,				0,				0, 1}
		//};
		//matrix4d rotate2(_rotate2);


		//for (int i = 0; i < 3; i++)
		//{
		//	vectors[i] = rotate2 * vectors[i];
		//	vectors[i] = vectors[i] *
		//		vector4d(videoBuffer->width / 2 / scale, videoBuffer->height / 2 / scale, 1, 1) +
		//		vector4d(videoBuffer->width / 2, videoBuffer->height / 2, 0, 0);

		//}
		//	vector2d _vectors[3] = {
		//		vector2d(vectors[0]),
		//		vector2d(vectors[1]),
		//		vector2d(vectors[2])
		//	};
		//	std::vector<vector2d> vecs = {
		//		vector2d(vectors[0]),
		//		vector2d(vectors[1]),
		//		vector2d(vectors[2])
		//	};
		//	rasterizer.drawTriangle(vecs, Pixel(0xffffff)  );

		//	rasterizer.FillTriangle(_vectors, Pixel(0xf00f00));

			 

		////////////////////////////// DrawMesh Function //////////////////////////////////
		/*_m = m;
		rasterizer.drawMesh(_m, Pixel(0xFFFFFF));*/
		

		////////////////////////////// HardCoding Draw Mesh //////////////////////////////////
		// Draw the mesh using the CPU loop (lines still through GPU!)
		
		//_m = m;
		_m.triags = {};
		for (int i = 0; i < cube_faces.size(); i++)
		{
			_m.triags.push_back(cube_faces[i]);
		}
		
		vector4d vCamera = vector4d(0, 0, 0, 0);
		int count = _m.triags.size(); 

		float _scale[4] = { videoBuffer->height / 2 / scale, videoBuffer->height / 2 / -scale, -1, 1 };
		float _trans[4] = { videoBuffer->width / 2, videoBuffer->height/2, 50, 0 };
		for (int i = 0; i < count; i++)
		{


			for (int v = 0; v < 3; v++)
			{
				// transitions in world space
				_m.triags[i].vectors[v] *= 25;
				_m.triags[i].vectors[v] = rotatez * _m.triags[i].vectors[v];
				//_m.triags[i].vectors[v] = rotate * _m.triags[i].vectors[v];

				// projection
				_m.triags[i].vectors[v] = orthoPorj1 * _m.triags[i].vectors[v];

				// world to screen
				_m.triags[i].vectors[v] = _m.triags[i].vectors[v] *
					vector4d(_scale[0], _scale[1], _scale[2], _scale[3]) +
					vector4d(_trans[0], _trans[1], _trans[2], _trans[3]);
			}

			vector3d A = _m.triags[i].vectors[1] - _m.triags[i].vectors[0];
			vector3d B = _m.triags[i].vectors[2] - _m.triags[i].vectors[0];
			vector3d N(
				A.y * B.z - A.z * B.y,
				A.z * B.x - A.x * B.z,
				A.x * B.y - A.y * B.x
			);
			//float absN = sqrt(N.x * N.x + N.y * N.y + N.z + N.z);
			//N /= absN;

			//if (N.z < 0 )
			if (
				N.x * (_m.triags[i].vectors[0].x - vCamera.x) +
				N.y * (_m.triags[i].vectors[0].y - vCamera.y) +
				N.z * (_m.triags[i].vectors[0].z - vCamera.z) < .0f
				)
			{

			vector2d _vectors[3] = {
				vector2d(_m.triags[i].vectors[0]),
				vector2d(_m.triags[i].vectors[1]),
				vector2d(_m.triags[i].vectors[2])
			};
			std::vector<vector2d> vecs = {
				vector2d(_vectors[0]),
				vector2d(_vectors[1]),
				vector2d(_vectors[2])
			};
			if (_m.triags[i].vectors[2].z != 0)
			{
					
				int oo = 0;
			}
			float minZ = -_m.triags[i].vectors[0].z ;
			//rasterizer.FillTriangle(_vectors, Pixel(0xf77f00, minZ));
			rasterizer.drawTriangle(vecs, Pixel(0xffffff, minZ));

			}
			
		}
		


		////////////////////////////// HardCoding Draw Mesh //////////////////////////////////
		// Draw the mesh using the CPU loop (lines still through GPU!)

		//_m = dyn_m;
		//int count = dyn_m.triags.size();
		//
		//for (int i = 0; i < count; i++)
		//{
		//	for (int v = 0; v < 3; v++)
		//	{
		//		if (!tdone)
		//		{
		//			float _scale[4] = { videoBuffer->height / 2 / scale, videoBuffer->height / 2 / scale, 1, 1 };
		//			float _trans[4] = { videoBuffer->width / 2, videoBuffer->height / 2, 0, 0 };
		//			dyn_m.triags[i].vectors[v] *= 30.0f;
		//			dyn_m.triags[i].vectors[v] = rotatez * dyn_m.triags[i].vectors[v];
		//			dyn_m.triags[i].vectors[v] = orthoPorj1 * dyn_m.triags[i].vectors[v];
		//			dyn_m.triags[i].vectors[v] = dyn_m.triags[i].vectors[v] *
		//				cvector4d(_scale) +
		//				cvector4d(_trans);
		//			tdone = true;
		//		}
		//	}
		//	float* v1 = dyn_m.triags[i].vectors[0].read();
		//	float* v2 = dyn_m.triags[i].vectors[1].read();
		//	float* v3 = dyn_m.triags[i].vectors[2].read();
		//
		//	/*vector2d _vectors[3] = {
		//		vector2d(_m.triags[i].vectors[0]),
		//		vector2d(_m.triags[i].vectors[1]),
		//		vector2d(_m.triags[i].vectors[2])
		//	};*/
		//	vector2d _vectors[3] = {
		//		vector2d(v1[0], v1[1]),
		//		vector2d(v2[0], v2[1]),
		//		vector2d(v3[0], v3[1])
		//	};
		//	std::vector<vector2d> vecs = {
		//		vector2d(_vectors[0]),
		//		vector2d(_vectors[1]),
		//		vector2d(_vectors[2])
		//	};
		//
		//	rasterizer.FillTriangle(_vectors, Pixel(0xf00f00));
		//	//rasterizer.drawTriangle(vecs, Pixel(0xffffff));
		//}
		//















		// print FPS
		int val = 1.0f/deltaTime;
		int scale = 16;
		int offset = 0;
		rasterizer.drawPolygon(
			translateScale(chars[(val/100) % 10], 20, vector2d(scale + offset, 40)).vectors,
			cudaRasterizer::PolygonMode::Disconnected, Pixel(0xffffff, 100)
		);
		offset += scale + scale/2;
		rasterizer.drawPolygon(
			translateScale(chars[(val / 10) % 10], 20, vector2d(scale + offset, 40)).vectors,
			cudaRasterizer::PolygonMode::Disconnected, Pixel(0xffffff, 100)
		);
		offset += scale + scale / 2;
		rasterizer.drawPolygon(
			translateScale(chars[val % 10], 20, vector2d(scale + offset, 40)).vectors,
			cudaRasterizer::PolygonMode::Disconnected, Pixel(0xffffff, 100)
		);
		offset += scale + scale / 2;



		/*  End Drawing Scope */
		rasterizer.finishSession();
		return true;
	}










}


