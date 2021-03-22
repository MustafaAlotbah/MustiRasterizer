#include "geometry_processor.h"




namespace mge {

	mesh2d translateScale(mesh2d mesh, float scale, vector2d offset) {
		for (int i = 0; i < mesh.vectors.size(); i++)
		{
			mesh.vectors[i] *= scale;
			mesh.vectors[i] += offset;
		}
		return mesh;
	}
}