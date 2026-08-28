#extension GL_EXT_mesh_shader : require

#include "common_model2.glsl"

struct Task {
    mat4 model;
    MeshDataModelMeshlets meshlets;
    MeshDataModelMeshletInfos meshletInfos;
    uint8_t meshletOffsets[MODEL_PART_SIZE];
    uint meshletBase;
};

#define TASK_VAR taskPayloadSharedEXT Task
