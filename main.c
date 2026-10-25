// render_gltf.c
// Compile (Linux): gcc -o render_gltf render_gltf.c $(pkg-config --cflags --libs sdl3) -lGL -lm
// (Make sure cgltf.h is in the same directory)

#include <SDL3/SDL.h>
#define GL_GLEXT_PROTOTYPES
#include <SDL3/SDL_opengl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define CGLTF_IMPLEMENTATION
#include "cgltf.h"

// ------------------------------------------------------------
// Minimal 4x4 matrix helpers (column-major, OpenGL style)
// ------------------------------------------------------------
typedef float mat4[16];

static void mat4_identity(mat4 m) {
    for (int i = 0; i < 16; i++) m[i] = (i % 5 == 0) ? 1.0f : 0.0f;
}

static void mat4_mul(mat4 out, const mat4 a, const mat4 b) {
    mat4 tmp = {0};
    for (int c = 0; c < 4; c++)
        for (int r = 0; r < 4; r++)
            for (int k = 0; k < 4; k++)
                tmp[c*4+r] += a[k*4+r] * b[c*4+k];
    for (int i = 0; i < 16; i++) out[i] = tmp[i];
}

static void mat4_perspective(mat4 m, float fov, float aspect, float znear, float zfar) {
    float f = 1.0f / tanf(fov * 0.5f);
    m[0] = f / aspect; m[1] = 0; m[2] = 0; m[3] = 0;
    m[4] = 0; m[5] = f; m[6] = 0; m[7] = 0;
    m[8] = 0; m[9] = 0; m[10] = (zfar + znear) / (znear - zfar); m[11] = -1;
    m[12] = 0; m[13] = 0; m[14] = (2 * zfar * znear) / (znear - zfar); m[15] = 0;
}

static void mat4_look_at(mat4 m, float eyeX, float eyeY, float eyeZ,
                         float cX, float cY, float cZ,
                         float upX, float upY, float upZ) {
    float fx = cX - eyeX, fy = cY - eyeY, fz = cZ - eyeZ;
    float len = sqrtf(fx*fx + fy*fy + fz*fz);
    fx /= len; fy /= len; fz /= len;
    float sx = fy * upZ - fz * upY;
    float sy = fz * upX - fx * upZ;
    float sz = fx * upY - fy * upX;
    len = sqrtf(sx*sx + sy*sy + sz*sz);
    sx /= len; sy /= len; sz /= len;
    float ux = sy * fz - sz * fy;
    float uy = sz * fx - sx * fz;
    float uz = sx * fy - sy * fx;
    m[0] = sx; m[1] = ux; m[2] = -fx; m[3] = 0;
    m[4] = sy; m[5] = uy; m[6] = -fy; m[7] = 0;
    m[8] = sz; m[9] = uz; m[10] = -fz; m[11] = 0;
    m[12] = -(sx*eyeX + sy*eyeY + sz*eyeZ);
    m[13] = -(ux*eyeX + uy*eyeY + uz*eyeZ);
    m[14] =  (fx*eyeX + fy*eyeY + fz*eyeZ);
    m[15] = 1;
}

static void mat4_translate(mat4 m, float x, float y, float z) {
    mat4_identity(m);
    m[12] = x; m[13] = y; m[14] = z;
}

// ------------------------------------------------------------
// Shader compilation helpers
// ------------------------------------------------------------
static GLuint compile_shader(GLenum type, const char *src) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &src, NULL);
    glCompileShader(shader);
    GLint ok;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[1024];
        glGetShaderInfoLog(shader, sizeof(log), NULL, log);
        fprintf(stderr, "Shader compile error: %s\n", log);
        exit(1);
    }
    return shader;
}

static GLuint create_program(const char *vs_src, const char *fs_src) {
    GLuint vs = compile_shader(GL_VERTEX_SHADER, vs_src);
    GLuint fs = compile_shader(GL_FRAGMENT_SHADER, fs_src);
    GLuint prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glLinkProgram(prog);
    GLint ok;
    glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[1024];
        glGetProgramInfoLog(prog, sizeof(log), NULL, log);
        fprintf(stderr, "Program link error: %s\n", log);
        exit(1);
    }
    glDeleteShader(vs);
    glDeleteShader(fs);
    return prog;
}

// ------------------------------------------------------------
// Main
// ------------------------------------------------------------
int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.gltf|model.glb>\n", argv[0]);
        return 1;
    }

    // ---------- 1. SDL3 initialisation ----------
    if (!SDL_Init(SDL_INIT_VIDEO)) {
        fprintf(stderr, "SDL_Init: %s\n", SDL_GetError());
        return 1;
    }

    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);

    SDL_Window *window = SDL_CreateWindow("glTF Viewer",
                                         800, 600,
                                         SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE);
    if (!window) {
        fprintf(stderr, "SDL_CreateWindow: %s\n", SDL_GetError());
        SDL_Quit();
        return 1;
    }

    SDL_GLContext gl_context = SDL_GL_CreateContext(window);
    if (!gl_context) {
        fprintf(stderr, "SDL_GL_CreateContext: %s\n", SDL_GetError());
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;
    }
    SDL_GL_SetSwapInterval(1); // vsync

    // ---------- 2. Load glTF / GLB with cgltf ----------
    cgltf_options options = {0};
    cgltf_data *data = NULL;
    cgltf_result res = cgltf_parse_file(&options, argv[1], &data);
    if (res != cgltf_result_success) {
        fprintf(stderr, "cgltf_parse_file failed: %d\n", res);
        goto cleanup;
    }
    res = cgltf_load_buffers(&options, data, argv[1]);
    if (res != cgltf_result_success) {
        fprintf(stderr, "cgltf_load_buffers failed: %d\n", res);
        goto cleanup;
    }
	fprintf(stderr, "meshes=%zu\n", data->meshes_count);
	for (cgltf_size m = 0; m < data->meshes_count; m++) {
	    cgltf_mesh *mesh = &data->meshes[m];
	    fprintf(stderr, "  mesh[%zu]: primitives=%zu\n", m, mesh->primitives_count);
	    for (cgltf_size p = 0; p < mesh->primitives_count; p++) {
	        cgltf_primitive *prim = &mesh->primitives[p];
	        cgltf_size vcount = 0;
	        for (cgltf_size a = 0; a < prim->attributes_count; a++)
	            if (prim->attributes[a].type == cgltf_attribute_type_position)
	                vcount = prim->attributes[a].data->count;
	        fprintf(stderr,
	                "    prim[%zu]: mode=%d verts=%zu idx=%s%zu\n",
	                p, (int)prim->type, vcount,
	                prim->indices ? "" : "none:",
	                prim->indices ? prim->indices->count : (cgltf_size)0);
	    }
	}
	fprintf(stderr, "nodes=%zu\n", data->nodes_count);
    // ---------- 3. Collect all vertex positions and indices ----------
    // For simplicity we flatten every primitive into one VBO/EBO.
    // (A production renderer would keep materials / primitives separate.)
    float *positions = NULL;
    size_t pos_count = 0;      // number of vertices (not floats)
    unsigned int *indices = NULL;
    size_t idx_count = 0;      // number of indices

    // First pass: count total vertices and indices.
    for (cgltf_size m = 0; m < data->meshes_count; m++) {
        cgltf_mesh *mesh = &data->meshes[m];
        for (cgltf_size p = 0; p < mesh->primitives_count; p++) {
            cgltf_primitive *prim = &mesh->primitives[p];
            for (cgltf_size a = 0; a < prim->attributes_count; a++) {
                if (prim->attributes[a].type == cgltf_attribute_type_position) {
                    pos_count += prim->attributes[a].data->count;
                    break;
                }
            }
            if (prim->indices) idx_count += prim->indices->count;
        }
    }

    if (pos_count == 0 || idx_count == 0) {
        fprintf(stderr, "No vertex positions or indices found.\n");
        goto cleanup;
    }

    positions = malloc(pos_count * 3 * sizeof(float));
    indices   = malloc(idx_count * sizeof(unsigned int));
    if (!positions || !indices) {
        fprintf(stderr, "Out of memory\n");
        goto cleanup;
    }

    size_t pos_offset = 0;  // vertex offset (in vertices)
    size_t idx_offset = 0;

    // Bounding-box min/max for centring the model
    float bmin[3] = { 1e30f,  1e30f,  1e30f};
    float bmax[3] = {-1e30f, -1e30f, -1e30f};

    // Second pass: copy data.
    for (cgltf_size m = 0; m < data->meshes_count; m++) {
        cgltf_mesh *mesh = &data->meshes[m];
        for (cgltf_size p = 0; p < mesh->primitives_count; p++) {
            cgltf_primitive *prim = &mesh->primitives[p];

            // --- positions ---
            cgltf_accessor *pos_acc = NULL;
            for (cgltf_size a = 0; a < prim->attributes_count; a++) {
                if (prim->attributes[a].type == cgltf_attribute_type_position) {
                    pos_acc = prim->attributes[a].data;
                    break;
                }
            }
            if (!pos_acc) continue;

            for (cgltf_size i = 0; i < pos_acc->count; i++) {
                float v[3];
                cgltf_accessor_read_float(pos_acc, i, v, 3);
                size_t vi = pos_offset + i;
                positions[vi*3+0] = v[0];
                positions[vi*3+1] = v[1];
                positions[vi*3+2] = v[2];
                for (int k = 0; k < 3; k++) {
                    if (v[k] < bmin[k]) bmin[k] = v[k];
                    if (v[k] > bmax[k]) bmax[k] = v[k];
                }
            }

            // --- indices ---
            if (prim->indices) {
                for (cgltf_size i = 0; i < prim->indices->count; i++) {
                    unsigned int idx = (unsigned int)cgltf_accessor_read_index(prim->indices, i);
                    // Adjust index to be relative to the whole flattened buffer
                    indices[idx_offset + i] = idx + (unsigned int)pos_offset;
                }
                idx_offset += prim->indices->count;
            }

            pos_offset += pos_acc->count;
        }
    }

    // Compute centre and radius of the bounding box.
    float center[3] = {
        (bmin[0] + bmax[0]) * 0.5f,
        (bmin[1] + bmax[1]) * 0.5f,
        (bmin[2] + bmax[2]) * 0.5f
    };
    float radius = 0.0f;
    for (int i = 0; i < 3; i++) {
        float d = (bmax[i] - bmin[i]) * 0.5f;
        if (d > radius) radius = d;
    }
    if (radius < 0.001f) radius = 1.0f;   // avoid zero

    // ---------- 4. Create OpenGL buffers ----------
    GLuint vao, vbo, ebo;
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    glGenBuffers(1, &ebo);

    glBindVertexArray(vao);

    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER,
                 pos_count * 3 * sizeof(float),
                 positions,
                 GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                 idx_count * sizeof(unsigned int),
                 indices,
                 GL_STATIC_DRAW);

    // attribute 0: position (vec3)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glBindVertexArray(0);

    // ---------- 5. Shaders ----------
    const char *vs_src =
        "#version 330 core\n"
        "layout(location = 0) in vec3 aPos;\n"
        "uniform mat4 uModel;\n"
        "uniform mat4 uView;\n"
        "uniform mat4 uProj;\n"
        "void main() {\n"
        "   gl_Position = uProj * uView * uModel * vec4(aPos, 1.0);\n"
        "}\n";

    const char *fs_src =
        "#version 330 core\n"
        "out vec4 FragColor;\n"
        "void main() {\n"
        "   FragColor = vec4(0.85, 0.55, 0.35, 1.0);\n" // warm orange
        "}\n";

    GLuint program = create_program(vs_src, fs_src);

    // ---------- 6. Camera setup ----------
    // Move the model so its centre is at the origin.
    mat4 model;
    mat4_translate(model, -center[0], -center[1], -center[2]);

    // Place the camera on the +Z axis, looking at the origin.
    float cam_dist = radius * 2.5f;
    mat4 view;
    mat4_look_at(view,
                 0.0f, 0.0f, cam_dist,   // eye
                 0.0f, 0.0f, 0.0f,      // centre
                 0.0f, 1.0f, 0.0f);     // up

    mat4 proj;
    mat4_perspective(proj, 45.0f * (float)M_PI / 180.0f,
                     800.0f / 600.0f, 0.1f, cam_dist + radius * 2.0f);

    // ---------- 7. Render loop ----------
    int running = 1;
    SDL_Event e;

    glEnable(GL_DEPTH_TEST);
    glClearColor(0.12f, 0.12f, 0.15f, 1.0f);

    while (running) {
        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_EVENT_QUIT) {
                running = 0;
            } else if (e.type == SDL_EVENT_WINDOW_RESIZED) {
                int w = e.window.data1;
                int h = e.window.data2;
                glViewport(0, 0, w, h);
                mat4_perspective(proj, 45.0f * (float)M_PI / 180.0f,
                                 (float)w / (float)h, 0.1f, cam_dist + radius * 2.0f);
            }
        }

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glUseProgram(program);
        glUniformMatrix4fv(glGetUniformLocation(program, "uModel"), 1, GL_FALSE, model);
        glUniformMatrix4fv(glGetUniformLocation(program, "uView"),  1, GL_FALSE, view);
        glUniformMatrix4fv(glGetUniformLocation(program, "uProj"),  1, GL_FALSE, proj);

        glBindVertexArray(vao);
        glDrawElements(GL_TRIANGLES, (GLsizei)idx_count, GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);

        SDL_GL_SwapWindow(window);
    }

    // ---------- 8. Cleanup ----------
    glDeleteProgram(program);
    glDeleteVertexArrays(1, &vao);
    glDeleteBuffers(1, &vbo);
    glDeleteBuffers(1, &ebo);

cleanup:
    free(positions);
    free(indices);
    if (data) cgltf_free(data);
    SDL_GL_DestroyContext(gl_context);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}

