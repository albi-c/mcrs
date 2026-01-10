glslc -fshader-stage=vertex shaders/vert.glsl -o shaders/vert.spv
glslc -fshader-stage=fragment shaders/frag.glsl -o shaders/frag.spv

#glslc -fshader-stage=vertex shaders/vert_post.glsl -o shaders/vert_post.spv
glslc --target-env=vulkan1.3 -fshader-stage=mesh shaders/mesh_post.glsl -o shaders/mesh_post.spv
glslc -fshader-stage=fragment shaders/frag_post.glsl -o shaders/frag_post.spv
glslc -fshader-stage=compute shaders/comp_post.glsl -o shaders/comp_post.spv
