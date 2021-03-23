#include "geometry_processor.h"




namespace mge {

	path2d translateScale(path2d mesh, float scale, vector2d offset) {
		for (int i = 0; i < mesh.vectors.size(); i++)
		{
			mesh.vectors[i] *= scale;
			mesh.vectors[i] += offset;
		}
		return mesh;
	}



	__device__ __host__ path4d::path4d(vector4d a, vector4d b, vector4d c) {
		vectors.push_back(a);
		vectors.push_back(b);
		vectors.push_back(c);
	}


	// thanks to javidx9 (olc)
	// https://www.youtube.com/watch?v=XgMWc6LumG4
	bool Mesh::loadFromFile(std::string fileName) {
		std::ifstream file(fileName);
		if (!file.is_open())
		{
			return false;
		}
		
		std::vector<vector4d> vertices;


		while (!file.eof()) {
			char line[128];
			file.getline(line, 128);

			std::strstream stream;
			stream << line;
			char line_delimiter;

			if (line[0] == 'v')
			{
				vector4d v(1);
				stream >> line_delimiter >> v.x >> v.y >> v.z;
				v.y = -v.y;
				vertices.push_back(v);
			}


			if (line[0] == 'f')
			{
				int face[3];
				stream >> line_delimiter >> face[0] >> face[1] >> face[2];
				triags.push_back(path4d(
					vertices[face[0]- 1], vertices[face[1]- 1], vertices[face[2]- 1]
				));
			}



		}


		return true;
	}
















}