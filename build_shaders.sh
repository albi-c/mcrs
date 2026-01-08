glslc -fshader-stage=vertex shaders/vert.glsl -o shaders/vert.spv
glslc -fshader-stage=fragment shaders/frag.glsl -o shaders/frag.spv

glslc -fshader-stage=vertex shaders/vert_post.glsl -o shaders/vert_post.spv
glslc -fshader-stage=fragment shaders/frag_post.glsl -o shaders/frag_post.spv
