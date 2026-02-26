#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define CUDA_CHECK(call)                                                                         \
    do {                                                                                         \
        cudaError_t err__ = (call);                                                              \
        if (err__ != cudaSuccess) {                                                              \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " -> "              \
                      << cudaGetErrorString(err__) << std::endl;                                 \
            std::exit(EXIT_FAILURE);                                                             \
        }                                                                                        \
    } while (0)

struct Camera {
    float3 eye;
    float3 right;
    float3 up;
    float3 forward;
    float tan_half_fov;
    float aspect;
};

struct Triangle {
    float3 a;
    float3 b;
    float3 c;
    float3 normal;
    float area;
};

struct Options {
    int width = 1280;
    int height = 720;
    int frames = 240;
    int particles = 160000;
    bool regroup = true;
    std::string obj_path = "stanford-bunny.obj";
    std::string out_dir = "frames";
};

__host__ __device__ inline float clampf(float x, float lo, float hi) {
    return fminf(hi, fmaxf(lo, x));
}

__host__ __device__ inline float lerpf(float a, float b, float t) {
    return a + (b - a) * t;
}

__host__ __device__ inline float smoothstepf(float e0, float e1, float x) {
    if (e0 == e1) {
        return x < e0 ? 0.0f : 1.0f;
    }
    float t = clampf((x - e0) / (e1 - e0), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

__host__ __device__ inline float3 operator+(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ __device__ inline float3 operator-(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__host__ __device__ inline float3 operator*(const float3& a, float s) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}

__host__ __device__ inline float3 operator*(float s, const float3& a) {
    return a * s;
}

__host__ __device__ inline float3 operator/(const float3& a, float s) {
    return make_float3(a.x / s, a.y / s, a.z / s);
}

__host__ __device__ inline float3& operator+=(float3& a, const float3& b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    return a;
}

__host__ __device__ inline float dot3(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ inline float3 cross3(const float3& a, const float3& b) {
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

__host__ __device__ inline float length3(const float3& v) {
    return sqrtf(dot3(v, v));
}

__host__ __device__ inline float3 normalize3(const float3& v) {
    float len = length3(v);
    if (len < 1e-8f) {
        return make_float3(0.0f, 0.0f, 0.0f);
    }
    return v / len;
}

__host__ __device__ inline float3 lerp3(const float3& a, const float3& b, float t) {
    return make_float3(lerpf(a.x, b.x, t), lerpf(a.y, b.y, t), lerpf(a.z, b.z, t));
}

__device__ inline uint32_t pcg_hash(uint32_t input) {
    uint32_t state = input * 747796405u + 2891336453u;
    uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

__device__ inline float hash3i(int x, int y, int z) {
    uint32_t ux = static_cast<uint32_t>(x);
    uint32_t uy = static_cast<uint32_t>(y);
    uint32_t uz = static_cast<uint32_t>(z);
    uint32_t mixed = (ux * 73856093u) ^ (uy * 19349663u) ^ (uz * 83492791u);
    return (pcg_hash(mixed) & 0x00ffffffu) * (1.0f / 16777215.0f);
}

__device__ inline float value_noise_3d(const float3& p) {
    int xi = static_cast<int>(floorf(p.x));
    int yi = static_cast<int>(floorf(p.y));
    int zi = static_cast<int>(floorf(p.z));

    float fx = p.x - floorf(p.x);
    float fy = p.y - floorf(p.y);
    float fz = p.z - floorf(p.z);

    float ux = fx * fx * (3.0f - 2.0f * fx);
    float uy = fy * fy * (3.0f - 2.0f * fy);
    float uz = fz * fz * (3.0f - 2.0f * fz);

    float c000 = hash3i(xi + 0, yi + 0, zi + 0);
    float c100 = hash3i(xi + 1, yi + 0, zi + 0);
    float c010 = hash3i(xi + 0, yi + 1, zi + 0);
    float c110 = hash3i(xi + 1, yi + 1, zi + 0);
    float c001 = hash3i(xi + 0, yi + 0, zi + 1);
    float c101 = hash3i(xi + 1, yi + 0, zi + 1);
    float c011 = hash3i(xi + 0, yi + 1, zi + 1);
    float c111 = hash3i(xi + 1, yi + 1, zi + 1);

    float x00 = lerpf(c000, c100, ux);
    float x10 = lerpf(c010, c110, ux);
    float x01 = lerpf(c001, c101, ux);
    float x11 = lerpf(c011, c111, ux);
    float y0 = lerpf(x00, x10, uy);
    float y1 = lerpf(x01, x11, uy);
    return lerpf(y0, y1, uz) * 2.0f - 1.0f;
}

__device__ inline float3 potential_field(const float3& p) {
    return make_float3(
        value_noise_3d(make_float3(p.x + 31.4f, p.y + 17.2f, p.z + 9.1f)),
        value_noise_3d(make_float3(p.x - 14.3f, p.y + 53.7f, p.z + 27.8f)),
        value_noise_3d(make_float3(p.x + 22.6f, p.y - 33.2f, p.z + 48.4f))
    );
}

__device__ inline float3 curl_noise(const float3& p) {
    const float e = 0.08f;
    const float inv = 1.0f / (2.0f * e);

    float3 px1 = potential_field(make_float3(p.x + e, p.y, p.z));
    float3 px0 = potential_field(make_float3(p.x - e, p.y, p.z));
    float3 py1 = potential_field(make_float3(p.x, p.y + e, p.z));
    float3 py0 = potential_field(make_float3(p.x, p.y - e, p.z));
    float3 pz1 = potential_field(make_float3(p.x, p.y, p.z + e));
    float3 pz0 = potential_field(make_float3(p.x, p.y, p.z - e));

    float dFz_dy = (py1.z - py0.z) * inv;
    float dFy_dz = (pz1.y - pz0.y) * inv;
    float dFx_dz = (pz1.x - pz0.x) * inv;
    float dFz_dx = (px1.z - px0.z) * inv;
    float dFy_dx = (px1.y - px0.y) * inv;
    float dFx_dy = (py1.x - py0.x) * inv;

    return make_float3(
        dFz_dy - dFy_dz,
        dFx_dz - dFz_dx,
        dFy_dx - dFx_dy
    );
}

__global__ void init_particles(const float4* p0, float4* pos, float4* vel, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) {
        return;
    }
    float4 base = p0[idx];
    pos[idx] = make_float4(base.x, base.y, base.z, 1.0f);
    vel[idx] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
}

__global__ void step_particles(
    const float4* p0,
    const float4* n0,
    float4* pos,
    float4* vel,
    int count,
    float dt,
    float time_s,
    float progress,
    int regroup_mode,
    float3 center
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) {
        return;
    }

    float4 b = p0[idx];
    float4 n = n0[idx];
    float4 cp = pos[idx];
    float4 cv = vel[idx];

    float3 base_pos = make_float3(b.x, b.y, b.z);
    float3 normal = normalize3(make_float3(n.x, n.y, n.z));
    float phase = n.w;

    float3 p = make_float3(cp.x, cp.y, cp.z);
    float3 v = make_float3(cv.x, cv.y, cv.z);

    // Single-point fracture seed at ear tip, then the crack front propagates over the body.
    const float3 ear_tip_source = make_float3(-0.01f, 0.90f, -0.22f);
    float dist_to_seed = length3(base_pos - ear_tip_source);

    float spread_delay = 0.34f * smoothstepf(0.0f, 1.85f, dist_to_seed);
    spread_delay += phase * 0.07f;
    float local_progress = clampf((progress - spread_delay) * 1.60f, 0.0f, 1.0f);

    float fracture = smoothstepf(0.02f + phase * 0.12f, 0.52f + phase * 0.15f, local_progress);
    float drift = smoothstepf(0.12f, 0.96f, local_progress);
    float regroup = regroup_mode ? smoothstepf(0.72f, 0.98f, progress) : 0.0f;
    float vanish = regroup_mode ? 0.0f : smoothstepf(0.66f + phase * 0.20f, 1.0f, local_progress);

    float d0 = local_progress - (0.06f + phase * 0.08f);
    float d1 = local_progress - (0.19f + phase * 0.05f);
    float pulse = expf(-(d0 * d0) / 0.008f) + 0.68f * expf(-(d1 * d1) / 0.018f);
    float source_gain = expf(-dist_to_seed * 3.4f);
    float wave_front = expf(-((dist_to_seed - local_progress * 1.40f) * (dist_to_seed - local_progress * 1.40f)) / 0.050f);

    float3 force = make_float3(0.0f, 0.0f, 0.0f);
    float blast = 7.0f + 8.2f * source_gain + 2.5f * wave_front;
    force += normal * (blast * pulse * fracture);
    float3 shock_dir = normalize3((base_pos - ear_tip_source) + make_float3(1e-4f, 2e-4f, 3e-4f));
    force += shock_dir * (2.8f * pulse * fracture * (0.45f + 0.55f * wave_front));

    float3 curl_p = p * 3.7f + make_float3(phase * 11.0f, time_s * 0.9f + phase * 5.0f, phase * 13.5f);
    float3 curl = curl_noise(curl_p);
    force += curl * (3.2f * drift * fracture);

    float3 out_dir = normalize3((p - center) + make_float3(1e-4f, 2e-4f, 3e-4f));
    force += out_dir * (1.2f * drift * fracture);

    float3 swirl_dir = normalize3(make_float3(-(p.z - center.z), 0.18f, p.x - center.x));
    force += swirl_dir * (0.65f * drift * (0.5f + 0.5f * b.w));

    if (regroup_mode) {
        float pull = 18.0f * regroup * regroup;
        force += (base_pos - p) * pull;
    } else {
        force += make_float3(0.0f, -3.6f * vanish, 0.0f);
    }

    if (regroup_mode) {
        float drag = 0.91f - 0.18f * regroup;
        drag = clampf(drag, 0.60f, 0.96f);
        v = v * drag + force * dt;
    } else {
        float drag = 0.96f - 0.09f * drift;
        drag = clampf(drag, 0.80f, 0.98f);
        v = v * drag + force * dt;
    }
    p += v * dt;

    if (regroup_mode) {
        // Force a deterministic reassembly near the ending so the final silhouette is stable.
        float snap = smoothstepf(0.82f, 1.0f, progress);
        p = lerp3(p, base_pos, snap);
        v = v * (1.0f - 0.92f * snap);
    }

    float alpha;
    if (regroup_mode) {
        float disperse = smoothstepf(0.22f, 0.72f, progress);
        alpha = clampf(1.0f - 0.4f * disperse + 0.85f * regroup, 0.30f, 1.0f);
    } else {
        float fade_start = 0.70f + phase * 0.16f;
        float fade = smoothstepf(fade_start, 1.0f, local_progress);
        alpha = clampf(1.0f - fade, 0.0f, 1.0f);
    }

    pos[idx] = make_float4(p.x, p.y, p.z, alpha);
    vel[idx] = make_float4(v.x, v.y, v.z, 0.0f);
}

__global__ void clear_accum(float4* accum, int pixels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pixels) {
        return;
    }
    accum[idx] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
}

__global__ void raster_particles(
    const float4* pos,
    const float4* vel,
    int count,
    float4* accum,
    int width,
    int height,
    Camera cam,
    float progress,
    int regroup_mode
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) {
        return;
    }

    float4 cp = pos[idx];
    float alpha = cp.w;
    if (alpha <= 1e-4f) {
        return;
    }

    float3 p = make_float3(cp.x, cp.y, cp.z);
    float3 rel = p - cam.eye;

    float cx = dot3(rel, cam.right);
    float cy = dot3(rel, cam.up);
    float cz = dot3(rel, cam.forward);
    if (cz <= 0.05f) {
        return;
    }

    float invz = 1.0f / cz;
    float ndc_x = cx * invz / (cam.tan_half_fov * cam.aspect);
    float ndc_y = cy * invz / cam.tan_half_fov;

    if (fabsf(ndc_x) > 1.25f || fabsf(ndc_y) > 1.25f) {
        return;
    }

    float sx = (ndc_x * 0.5f + 0.5f) * width;
    float sy = (1.0f - (ndc_y * 0.5f + 0.5f)) * height;

    float spread = smoothstepf(0.18f, 0.80f, progress);

    float4 cv = vel[idx];
    float3 v3 = make_float3(cv.x, cv.y, cv.z);
    float speed = length3(v3);
    float radius = 1.25f + 2.6f * invz + 1.2f * spread + 0.8f * clampf(speed * 0.22f, 0.0f, 1.0f);
    float heat = clampf(speed * 0.13f, 0.0f, 1.0f);

    float dissolve_mix = smoothstepf(0.16f, 0.78f, progress);
    float3 c_static = make_float3(0.99f, 0.78f, 0.92f);
    float3 c_mid = make_float3(0.97f, 0.44f, 0.95f);
    float3 c_deep = make_float3(0.65f, 0.16f, 0.92f);
    float3 c_tail = make_float3(0.93f, 0.24f, 0.98f);
    float3 color = lerp3(c_static, c_mid, dissolve_mix);
    color = lerp3(color, c_deep, heat * 0.92f);

    if (regroup_mode) {
        float regroup_mix = smoothstepf(0.74f, 1.0f, progress);
        color = lerp3(color, c_static, regroup_mix);
    } else {
        float fade_mix = smoothstepf(0.55f, 1.0f, progress);
        color = lerp3(color, c_deep, 0.35f * fade_mix);
    }

    float base_alpha = alpha * (0.16f + 0.62f * invz) * (0.95f + 0.35f * spread);
    base_alpha = clampf(base_alpha, 0.0f, 1.0f);

    float vx_cam = dot3(v3, cam.right);
    float vy_cam = dot3(v3, cam.up);
    float vx_px = vx_cam * invz / (cam.tan_half_fov * cam.aspect) * (0.5f * width);
    float vy_px = -vy_cam * invz / cam.tan_half_fov * (0.5f * height);
    float v2_px = vx_px * vx_px + vy_px * vy_px;

    float dir_x = 0.0f;
    float dir_y = 0.0f;
    float tail_len = 0.0f;
    if (v2_px > 1e-8f) {
        float inv_v = rsqrtf(v2_px);
        dir_x = vx_px * inv_v;
        dir_y = vy_px * inv_v;
        tail_len = clampf(speed * (3.2f + 6.0f * spread), 0.0f, 10.0f);
    }

    int trail_taps = 1;
    if (tail_len > 1.6f) {
        trail_taps = 2;
    }
    if (tail_len > 3.6f) {
        trail_taps = 3;
    }
    if (tail_len > 6.2f) {
        trail_taps = 4;
    }
    float tap_denom = static_cast<float>(trail_taps > 1 ? (trail_taps - 1) : 1);

    for (int tap = 0; tap < trail_taps; ++tap) {
        float t = static_cast<float>(tap) / tap_denom;
        float tap_x = sx - dir_x * tail_len * t;
        float tap_y = sy - dir_y * tail_len * t;
        float tap_radius = radius * (1.0f - 0.15f * t);
        int ir = static_cast<int>(ceilf(tap_radius));
        float tap_alpha = base_alpha * (tap == 0 ? 1.0f : (0.72f - 0.18f * t));
        float3 tap_color = lerp3(color, c_tail, 0.55f * t);
        int px = static_cast<int>(floorf(tap_x));
        int py = static_cast<int>(floorf(tap_y));

        for (int dy = -ir; dy <= ir; ++dy) {
            int yy = py + dy;
            if (yy < 0 || yy >= height) {
                continue;
            }
            for (int dx = -ir; dx <= ir; ++dx) {
                int xx = px + dx;
                if (xx < 0 || xx >= width) {
                    continue;
                }
                float nd = (dx * dx + dy * dy) / (tap_radius * tap_radius);
                if (nd > 1.0f) {
                    continue;
                }
                float w = expf(-nd * 2.7f);
                float a = tap_alpha * w;
                int pix = yy * width + xx;

                atomicAdd(&accum[pix].x, tap_color.x * a);
                atomicAdd(&accum[pix].y, tap_color.y * a);
                atomicAdd(&accum[pix].z, tap_color.z * a);
                atomicAdd(&accum[pix].w, a);
            }
        }
    }
}

__global__ void compose_frame(const float4* accum, unsigned char* rgb, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }

    int idx = y * width + x;
    float u = (x + 0.5f) / width;
    float v = (y + 0.5f) / height;

    float3 top = make_float3(0.18f, 0.10f, 0.25f);
    float3 bottom = make_float3(0.035f, 0.015f, 0.070f);
    float3 bg = lerp3(bottom, top, 1.0f - v);

    float dx = u - 0.52f;
    float dy = v - 0.48f;
    float halo = expf(-(dx * dx * 8.0f + dy * dy * 5.0f));
    bg += make_float3(0.34f, 0.10f, 0.30f) * (0.34f * halo);
    float hx = u - 0.34f;
    float hy = v - 0.30f;
    float halo2 = expf(-(hx * hx * 13.0f + hy * hy * 9.0f));
    bg += make_float3(0.11f, 0.07f, 0.22f) * (0.24f * halo2);

    float4 a = accum[idx];
    float density = a.w;
    float3 fg = density > 1e-6f
        ? make_float3(a.x / density, a.y / density, a.z / density)
        : make_float3(0.0f, 0.0f, 0.0f);

    float cover = clampf(density * 0.78f, 0.0f, 1.0f);
    float3 color = bg * (1.0f - cover) + fg * cover;

    float bloom = clampf(density * 0.20f, 0.0f, 1.0f);
    color += fg * (0.25f * bloom);

    float r2 = dx * dx + dy * dy;
    float vignette = clampf(1.0f - 0.65f * powf(r2, 0.7f), 0.35f, 1.0f);
    color = color * vignette;

    color.x = powf(clampf(color.x, 0.0f, 1.0f), 1.0f / 2.2f);
    color.y = powf(clampf(color.y, 0.0f, 1.0f), 1.0f / 2.2f);
    color.z = powf(clampf(color.z, 0.0f, 1.0f), 1.0f / 2.2f);

    rgb[idx * 3 + 0] = static_cast<unsigned char>(clampf(color.x, 0.0f, 1.0f) * 255.0f + 0.5f);
    rgb[idx * 3 + 1] = static_cast<unsigned char>(clampf(color.y, 0.0f, 1.0f) * 255.0f + 0.5f);
    rgb[idx * 3 + 2] = static_cast<unsigned char>(clampf(color.z, 0.0f, 1.0f) * 255.0f + 0.5f);
}

static int parse_obj_index(const std::string& token, int vertex_count) {
    std::string head = token;
    size_t slash = token.find('/');
    if (slash != std::string::npos) {
        head = token.substr(0, slash);
    }
    if (head.empty()) {
        return -1;
    }

    int idx = std::stoi(head);
    if (idx > 0) {
        return idx - 1;
    }
    if (idx < 0) {
        return vertex_count + idx;
    }
    return -1;
}

static bool load_obj(
    const std::string& path,
    std::vector<float3>& vertices,
    std::vector<int3>& faces
) {
    std::ifstream in(path);
    if (!in) {
        std::cerr << "Failed to open OBJ: " << path << std::endl;
        return false;
    }

    std::string line;
    vertices.clear();
    faces.clear();
    vertices.reserve(40000);
    faces.reserve(70000);

    while (std::getline(in, line)) {
        if (line.size() < 2) {
            continue;
        }

        if (line[0] == 'v' && (line[1] == ' ' || line[1] == '\t')) {
            std::istringstream iss(line.substr(1));
            float x, y, z;
            if (!(iss >> x >> y >> z)) {
                continue;
            }
            vertices.push_back(make_float3(x, y, z));
            continue;
        }

        if (line[0] == 'f' && (line[1] == ' ' || line[1] == '\t')) {
            std::istringstream iss(line.substr(1));
            std::vector<int> idxs;
            idxs.reserve(8);
            std::string token;
            while (iss >> token) {
                int idx = parse_obj_index(token, static_cast<int>(vertices.size()));
                if (idx >= 0) {
                    idxs.push_back(idx);
                }
            }

            if (idxs.size() < 3) {
                continue;
            }
            for (size_t i = 1; i + 1 < idxs.size(); ++i) {
                faces.push_back(make_int3(idxs[0], idxs[i], idxs[i + 1]));
            }
        }
    }

    return !vertices.empty() && !faces.empty();
}

static void normalize_vertices(std::vector<float3>& vertices) {
    float3 bmin = make_float3(
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max()
    );
    float3 bmax = make_float3(
        -std::numeric_limits<float>::max(),
        -std::numeric_limits<float>::max(),
        -std::numeric_limits<float>::max()
    );

    for (const float3& v : vertices) {
        bmin.x = std::min(bmin.x, v.x);
        bmin.y = std::min(bmin.y, v.y);
        bmin.z = std::min(bmin.z, v.z);
        bmax.x = std::max(bmax.x, v.x);
        bmax.y = std::max(bmax.y, v.y);
        bmax.z = std::max(bmax.z, v.z);
    }

    float3 center = (bmin + bmax) * 0.5f;
    float sx = bmax.x - bmin.x;
    float sy = bmax.y - bmin.y;
    float sz = bmax.z - bmin.z;
    float max_extent = std::max(sx, std::max(sy, sz));
    float scale = (max_extent > 1e-8f) ? (1.8f / max_extent) : 1.0f;

    for (float3& v : vertices) {
        v = (v - center) * scale;
    }
}

static std::vector<Triangle> build_triangles(
    const std::vector<float3>& vertices,
    const std::vector<int3>& faces
) {
    std::vector<Triangle> tris;
    tris.reserve(faces.size());
    for (const int3& f : faces) {
        if (f.x < 0 || f.y < 0 || f.z < 0) {
            continue;
        }
        if (f.x >= static_cast<int>(vertices.size()) ||
            f.y >= static_cast<int>(vertices.size()) ||
            f.z >= static_cast<int>(vertices.size())) {
            continue;
        }

        float3 a = vertices[f.x];
        float3 b = vertices[f.y];
        float3 c = vertices[f.z];
        float3 n = cross3(b - a, c - a);
        float nlen = length3(n);
        if (nlen < 1e-12f) {
            continue;
        }

        Triangle tri{};
        tri.a = a;
        tri.b = b;
        tri.c = c;
        tri.normal = n / nlen;
        tri.area = 0.5f * nlen;
        tris.push_back(tri);
    }
    return tris;
}

static void sample_particles(
    const std::vector<Triangle>& tris,
    int particle_count,
    std::vector<float4>& p0,
    std::vector<float4>& n0
) {
    p0.resize(particle_count);
    n0.resize(particle_count);

    std::vector<double> cdf(tris.size());
    double total_area = 0.0;
    for (size_t i = 0; i < tris.size(); ++i) {
        total_area += static_cast<double>(tris[i].area);
        cdf[i] = total_area;
    }

    std::mt19937 rng(42u);
    std::uniform_real_distribution<float> dist01(0.0f, 1.0f);
    std::uniform_real_distribution<double> dist_area(0.0, total_area);

    for (int i = 0; i < particle_count; ++i) {
        double r = dist_area(rng);
        auto it = std::lower_bound(cdf.begin(), cdf.end(), r);
        size_t tri_id = (it == cdf.end()) ? (cdf.size() - 1) : static_cast<size_t>(it - cdf.begin());
        const Triangle& t = tris[tri_id];

        float u = dist01(rng);
        float v = dist01(rng);
        float su = sqrtf(u);
        float b0 = 1.0f - su;
        float b1 = su * (1.0f - v);
        float b2 = su * v;

        float3 p = t.a * b0 + t.b * b1 + t.c * b2;
        float seed = dist01(rng);
        float phase = dist01(rng);

        p0[i] = make_float4(p.x, p.y, p.z, seed);
        n0[i] = make_float4(t.normal.x, t.normal.y, t.normal.z, phase);
    }
}

static Camera make_camera(int width, int height, float progress) {
    Camera cam{};
    const float pi = 3.1415926535f;
    float eased = smoothstepf(0.0f, 1.0f, progress);
    float orbit_angle = -0.55f + eased * (4.0f * pi);

    float p1 = progress - 0.30f;
    float p2 = progress - 0.76f;
    float close_1 = expf(-(p1 * p1) / 0.012f);
    float close_2 = expf(-(p2 * p2) / 0.016f);
    float close_mix = clampf(fmaxf(close_1, close_2), 0.0f, 1.0f);

    float radius = lerpf(3.30f, 1.72f, close_mix);
    radius += 0.20f * sinf(progress * 2.0f * pi * 2.1f + 0.35f);
    radius = clampf(radius, 1.60f, 3.50f);

    cam.eye = make_float3(
        sinf(orbit_angle) * radius,
        0.14f + 0.11f * sinf(progress * pi * 1.4f + 0.6f) + 0.06f * close_mix,
        cosf(orbit_angle) * radius
    );
    float3 target = make_float3(
        0.03f * sinf(progress * 2.0f * pi * 0.9f + 0.3f),
        0.02f + 0.10f * close_mix,
        -0.04f * close_mix
    );
    float3 world_up = make_float3(0.0f, 1.0f, 0.0f);
    cam.forward = normalize3(target - cam.eye);
    cam.right = normalize3(cross3(cam.forward, world_up));
    cam.up = cross3(cam.right, cam.forward);
    cam.tan_half_fov = tanf(35.0f * 0.5f * 3.1415926535f / 180.0f);
    cam.aspect = static_cast<float>(width) / static_cast<float>(height);
    return cam;
}

static void print_usage(const char* exe) {
    std::cout
        << "Usage: " << exe << " [options]\n"
        << "Options:\n"
        << "  --mode regroup|vanish    Final stage behavior (default: regroup)\n"
        << "  --frames N               Number of frames (default: 240)\n"
        << "  --particles N            Number of particles (default: 160000)\n"
        << "  --width N                Frame width (default: 1280)\n"
        << "  --height N               Frame height (default: 720)\n"
        << "  --obj PATH               OBJ path (default: stanford-bunny.obj)\n"
        << "  --out DIR                Output directory (default: frames)\n"
        << "  --help                   Show this help\n";
}

static Options parse_args(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        } else if (a == "--mode" && i + 1 < argc) {
            std::string m = argv[++i];
            if (m == "regroup") {
                opt.regroup = true;
            } else if (m == "vanish") {
                opt.regroup = false;
            } else {
                std::cerr << "Unknown mode: " << m << " (use regroup or vanish)\n";
                std::exit(EXIT_FAILURE);
            }
        } else if (a == "--frames" && i + 1 < argc) {
            opt.frames = std::max(1, std::stoi(argv[++i]));
        } else if (a == "--particles" && i + 1 < argc) {
            opt.particles = std::max(1000, std::stoi(argv[++i]));
        } else if (a == "--width" && i + 1 < argc) {
            opt.width = std::max(64, std::stoi(argv[++i]));
        } else if (a == "--height" && i + 1 < argc) {
            opt.height = std::max(64, std::stoi(argv[++i]));
        } else if (a == "--obj" && i + 1 < argc) {
            opt.obj_path = argv[++i];
        } else if (a == "--out" && i + 1 < argc) {
            opt.out_dir = argv[++i];
        } else {
            std::cerr << "Unknown option: " << a << "\n";
            print_usage(argv[0]);
            std::exit(EXIT_FAILURE);
        }
    }
    return opt;
}

int main(int argc, char** argv) {
    Options opt = parse_args(argc, argv);

    std::vector<float3> vertices;
    std::vector<int3> faces;
    if (!load_obj(opt.obj_path, vertices, faces)) {
        return EXIT_FAILURE;
    }

    normalize_vertices(vertices);
    std::vector<Triangle> tris = build_triangles(vertices, faces);
    if (tris.empty()) {
        std::cerr << "No valid triangles found in OBJ.\n";
        return EXIT_FAILURE;
    }

    std::vector<float4> h_p0;
    std::vector<float4> h_n0;
    sample_particles(tris, opt.particles, h_p0, h_n0);

    std::filesystem::create_directories(opt.out_dir);

    std::cout << "Loaded OBJ: " << opt.obj_path
              << " | verts=" << vertices.size()
              << " faces=" << faces.size()
              << " tris=" << tris.size()
              << " particles=" << opt.particles
              << " mode=" << (opt.regroup ? "regroup" : "vanish")
              << std::endl;

    const int count = opt.particles;
    const int pixels = opt.width * opt.height;
    const size_t particles_bytes = static_cast<size_t>(count) * sizeof(float4);
    const size_t accum_bytes = static_cast<size_t>(pixels) * sizeof(float4);
    const size_t rgb_bytes = static_cast<size_t>(pixels) * 3 * sizeof(unsigned char);

    float4* d_p0 = nullptr;
    float4* d_n0 = nullptr;
    float4* d_pos = nullptr;
    float4* d_vel = nullptr;
    float4* d_accum = nullptr;
    unsigned char* d_rgb = nullptr;

    CUDA_CHECK(cudaMalloc(&d_p0, particles_bytes));
    CUDA_CHECK(cudaMalloc(&d_n0, particles_bytes));
    CUDA_CHECK(cudaMalloc(&d_pos, particles_bytes));
    CUDA_CHECK(cudaMalloc(&d_vel, particles_bytes));
    CUDA_CHECK(cudaMalloc(&d_accum, accum_bytes));
    CUDA_CHECK(cudaMalloc(&d_rgb, rgb_bytes));

    CUDA_CHECK(cudaMemcpy(d_p0, h_p0.data(), particles_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_n0, h_n0.data(), particles_bytes, cudaMemcpyHostToDevice));

    const int tpb = 256;
    const int particle_blocks = (count + tpb - 1) / tpb;
    const int pixel_blocks = (pixels + tpb - 1) / tpb;

    init_particles<<<particle_blocks, tpb>>>(d_p0, d_pos, d_vel, count);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    dim3 block2d(16, 16);
    dim3 grid2d(
        static_cast<unsigned int>((opt.width + block2d.x - 1) / block2d.x),
        static_cast<unsigned int>((opt.height + block2d.y - 1) / block2d.y)
    );

    std::vector<unsigned char> h_rgb(rgb_bytes);

    // Keep dt fixed for stable animation pacing across different frame counts.
    const float dt = 1.0f / 60.0f;

    for (int frame = 0; frame < opt.frames; ++frame) {
        float progress = (opt.frames <= 1) ? 1.0f : static_cast<float>(frame) / static_cast<float>(opt.frames - 1);
        float time_s = progress * 6.0f;
        Camera cam = make_camera(opt.width, opt.height, progress);

        step_particles<<<particle_blocks, tpb>>>(
            d_p0,
            d_n0,
            d_pos,
            d_vel,
            count,
            dt,
            time_s,
            progress,
            opt.regroup ? 1 : 0,
            make_float3(0.0f, 0.0f, 0.0f)
        );
        CUDA_CHECK(cudaGetLastError());

        clear_accum<<<pixel_blocks, tpb>>>(d_accum, pixels);
        CUDA_CHECK(cudaGetLastError());

        raster_particles<<<particle_blocks, tpb>>>(
            d_pos,
            d_vel,
            count,
            d_accum,
            opt.width,
            opt.height,
            cam,
            progress,
            opt.regroup ? 1 : 0
        );
        CUDA_CHECK(cudaGetLastError());

        compose_frame<<<grid2d, block2d>>>(d_accum, d_rgb, opt.width, opt.height);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_rgb.data(), d_rgb, rgb_bytes, cudaMemcpyDeviceToHost));

        std::ostringstream oss;
        oss << "frame_" << std::setw(4) << std::setfill('0') << frame << ".png";
        std::filesystem::path out_path = std::filesystem::path(opt.out_dir) / oss.str();
        int ok = stbi_write_png(
            out_path.string().c_str(),
            opt.width,
            opt.height,
            3,
            h_rgb.data(),
            opt.width * 3
        );
        if (!ok) {
            std::cerr << "Failed to write frame: " << out_path << std::endl;
            break;
        }

        if (frame % 10 == 0 || frame == opt.frames - 1) {
            std::cout << "Rendered frame " << frame << " / " << (opt.frames - 1) << std::endl;
        }
    }

    CUDA_CHECK(cudaFree(d_p0));
    CUDA_CHECK(cudaFree(d_n0));
    CUDA_CHECK(cudaFree(d_pos));
    CUDA_CHECK(cudaFree(d_vel));
    CUDA_CHECK(cudaFree(d_accum));
    CUDA_CHECK(cudaFree(d_rgb));

    std::cout << "Done. Frames written to: " << opt.out_dir << std::endl;
    return 0;
}
