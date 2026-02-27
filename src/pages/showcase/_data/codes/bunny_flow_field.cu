#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define CUDA_CHECK(call)                                                                 \
    do {                                                                                 \
        cudaError_t err__ = (call);                                                      \
        if (err__ != cudaSuccess) {                                                      \
            std::fprintf(stderr, "CUDA error at %s:%d -> %s\n", __FILE__, __LINE__,   \
                         cudaGetErrorString(err__));                                      \
            std::exit(1);                                                                 \
        }                                                                                 \
    } while (0)

struct Config {
    std::string obj_path = "stanford-bunny.obj";
    std::string frame_dir = "bunny_flow_frames";
    int width = 960;
    int height = 960;
    int frames = 220;
    int particles = 260000;
    int substeps = 2;
    float dt = 0.0125f;
    uint32_t seed = 1337;
    std::array<float, 3> chaos_color{0.18f, 0.90f, 1.00f};
    std::array<float, 3> form_color{1.00f, 0.58f, 0.14f};
    std::array<float, 3> bg_color{0.018f, 0.016f, 0.045f};
};

struct CameraData {
    float3 pos;
    float3 right;
    float3 up;
    float3 forward;
    float tan_half_fov;
};

__host__ __device__ inline float clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

__host__ __device__ inline float3 make_f3(float x, float y, float z) {
    float3 r;
    r.x = x;
    r.y = y;
    r.z = z;
    return r;
}

__host__ __device__ inline float3 operator+(const float3 &a, const float3 &b) {
    return make_f3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ __device__ inline float3 operator-(const float3 &a, const float3 &b) {
    return make_f3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__host__ __device__ inline float3 operator*(const float3 &a, float s) {
    return make_f3(a.x * s, a.y * s, a.z * s);
}

__host__ __device__ inline float3 operator*(float s, const float3 &a) {
    return a * s;
}

__host__ __device__ inline float3 operator/(const float3 &a, float s) {
    return make_f3(a.x / s, a.y / s, a.z / s);
}

__host__ __device__ inline float dot3(const float3 &a, const float3 &b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ inline float3 cross3(const float3 &a, const float3 &b) {
    return make_f3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

__host__ __device__ inline float len3(const float3 &a) {
    return sqrtf(dot3(a, a));
}

__host__ __device__ inline float3 norm3(const float3 &a) {
    float l = len3(a);
    if (l < 1e-8f) {
        return make_f3(0.0f, 0.0f, 0.0f);
    }
    return a / l;
}

__host__ __device__ inline float smoothstep01(float x) {
    x = clampf(x, 0.0f, 1.0f);
    return x * x * (3.0f - 2.0f * x);
}

__host__ __device__ inline float mixf(float a, float b, float t) {
    return a + (b - a) * clampf(t, 0.0f, 1.0f);
}

static inline float morph_progress(float t) {
    return smoothstep01((t - 0.04f) / 0.58f);
}

static inline float settle_progress(float t) {
    return smoothstep01((t - 0.60f) / 0.35f);
}

__device__ inline uint32_t hash32_device(uint32_t x) {
    x ^= x >> 17;
    x *= 0xED5AD4BBu;
    x ^= x >> 11;
    x *= 0xAC4C1B51u;
    x ^= x >> 15;
    x *= 0x31848BABu;
    x ^= x >> 14;
    return x;
}

__device__ inline float rand01_device(uint32_t &state) {
    state = hash32_device(state);
    return static_cast<float>(state & 0x00FFFFFFu) * (1.0f / 16777216.0f);
}

__global__ void integrate_particles(float3 *pos,
                                    float3 *vel,
                                    const float3 *target,
                                    int count,
                                    float dt,
                                    float alpha,
                                    float settle,
                                    float time,
                                    uint32_t seed,
                                    int frame_idx) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) {
        return;
    }

    float3 p = pos[idx];
    float3 v = vel[idx];
    const float3 t = target[idx];

    float3 diff = t - p;
    float dist = sqrtf(dot3(diff, diff) + 1e-10f);
    float3 to_target = diff / (dist + 1e-6f);

    float3 chaos = make_f3(
        __sinf(2.2f * p.y + 1.15f * time) - __cosf(1.7f * p.z - 0.8f * time),
        __sinf(2.0f * p.z - 0.75f * time) + __cosf(1.9f * p.x + 0.55f * time),
        __sinf(2.35f * p.x + 1.05f * time) - __cosf(1.55f * p.y - 0.65f * time));

    float3 vortex = cross3(make_f3(0.0f, 1.0f, 0.0f), p);
    float3 axis = norm3(make_f3(__sinf(0.45f * time), 0.7f, __cosf(0.45f * time)));
    float3 spiral = cross3(to_target, axis);

    float pull = (0.85f + 3.4f * alpha + 2.6f * settle) * (1.0f + 0.55f / (0.12f + dist));
    float3 attract = diff * pull;

    uint32_t s = static_cast<uint32_t>(idx) * 9781u + static_cast<uint32_t>(frame_idx) * 6271u + seed * 26699u;
    float3 jitter = make_f3(rand01_device(s) * 2.0f - 1.0f,
                            rand01_device(s) * 2.0f - 1.0f,
                            rand01_device(s) * 2.0f - 1.0f);

    float chaos_gain = 1.35f * (1.0f - 0.85f * alpha) * (1.0f - 0.95f * settle);
    float vortex_gain = 0.34f * (1.0f - 0.75f * settle);
    float jitter_gain = 0.27f * (1.0f - 0.92f * alpha) * (1.0f - 0.98f * settle);
    float spiral_gain = 0.42f * (1.0f - 0.70f * settle);

    float3 f_chaos = chaos * chaos_gain + vortex * vortex_gain - p * 0.22f + jitter * jitter_gain;
    float3 f_form = attract + spiral * spiral_gain - v * (0.24f + 0.50f * alpha + 0.95f * settle);

    float blend = clampf(alpha + 0.45f * settle, 0.0f, 1.0f);
    float3 f = f_chaos * (1.0f - blend) + f_form * blend;

    v = v + f * dt;
    v = v * (0.992f - 0.05f * alpha - 0.22f * settle);
    p = p + v * dt;

    float snap = (0.012f * alpha + 0.110f * settle) * (dt / 0.0125f);
    snap = clampf(snap, 0.0f, 0.30f);
    p = p * (1.0f - snap) + t * snap;
    v = v * (1.0f - 0.22f * settle);

    float r = sqrtf(dot3(p, p));
    if (r > 3.2f) {
        float3 n = p / r;
        p = n * 3.2f;
        v = v - n * (1.45f * dot3(v, n));
    }

    pos[idx] = p;
    vel[idx] = v;
}

__global__ void decay_accum(float *accum, int total_values, float decay) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_values) {
        return;
    }
    accum[idx] *= decay;
}

__global__ void render_particles(const float3 *pos,
                                 const float3 *target,
                                 int count,
                                 CameraData cam,
                                 int width,
                                 int height,
                                 float alpha,
                                 float settle,
                                 float3 chaos_color,
                                 float3 form_color,
                                 float *accum) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) {
        return;
    }

    float3 p = pos[idx];
    float3 rel = p - cam.pos;

    float x_cam = dot3(rel, cam.right);
    float y_cam = dot3(rel, cam.up);
    float z_cam = dot3(rel, cam.forward);
    if (z_cam <= 0.02f) {
        return;
    }

    float ndc_x = x_cam / (z_cam * cam.tan_half_fov);
    float ndc_y = y_cam / (z_cam * cam.tan_half_fov);
    if (fabsf(ndc_x) > 1.2f || fabsf(ndc_y) > 1.2f) {
        return;
    }

    float sx = (ndc_x * 0.5f + 0.5f) * (width - 1);
    float sy = (0.5f - ndc_y * 0.5f) * (height - 1);

    int x0 = static_cast<int>(floorf(sx));
    int y0 = static_cast<int>(floorf(sy));
    float fx = sx - x0;
    float fy = sy - y0;

    float3 diff = target[idx] - p;
    float convergence = expf(-2.6f * sqrtf(dot3(diff, diff)));
    float mixv = clampf(0.60f * alpha + 0.25f * convergence + 0.25f * settle, 0.0f, 1.0f);

    float3 base = chaos_color * (1.0f - mixv) + form_color * mixv;
    uint32_t hs = hash32_device(static_cast<uint32_t>(idx) * 747796405u + 2891336453u);
    float tint = 0.85f + 0.30f * (static_cast<float>(hs & 1023u) / 1023.0f);

    float intensity = (0.20f + 0.62f * convergence + 0.22f * alpha + 0.25f * settle) * clampf(1.45f / z_cam, 0.20f, 1.10f);
    intensity *= clampf(1.0f - 0.45f * settle * (1.0f - convergence), 0.55f, 1.0f);
    float3 c = base * (tint * intensity);

    int x1 = x0 + 1;
    int y1 = y0 + 1;
    float w00 = (1.0f - fx) * (1.0f - fy);
    float w10 = fx * (1.0f - fy);
    float w01 = (1.0f - fx) * fy;
    float w11 = fx * fy;

    if (x0 >= 0 && x0 < width && y0 >= 0 && y0 < height) {
        int pi = (y0 * width + x0) * 3;
        atomicAdd(&accum[pi + 0], c.x * w00);
        atomicAdd(&accum[pi + 1], c.y * w00);
        atomicAdd(&accum[pi + 2], c.z * w00);
    }
    if (x1 >= 0 && x1 < width && y0 >= 0 && y0 < height) {
        int pi = (y0 * width + x1) * 3;
        atomicAdd(&accum[pi + 0], c.x * w10);
        atomicAdd(&accum[pi + 1], c.y * w10);
        atomicAdd(&accum[pi + 2], c.z * w10);
    }
    if (x0 >= 0 && x0 < width && y1 >= 0 && y1 < height) {
        int pi = (y1 * width + x0) * 3;
        atomicAdd(&accum[pi + 0], c.x * w01);
        atomicAdd(&accum[pi + 1], c.y * w01);
        atomicAdd(&accum[pi + 2], c.z * w01);
    }
    if (x1 >= 0 && x1 < width && y1 >= 0 && y1 < height) {
        int pi = (y1 * width + x1) * 3;
        atomicAdd(&accum[pi + 0], c.x * w11);
        atomicAdd(&accum[pi + 1], c.y * w11);
        atomicAdd(&accum[pi + 2], c.z * w11);
    }
}

__global__ void tone_map(const float *accum,
                         unsigned char *rgb,
                         int width,
                         int height,
                         float3 bg,
                         float exposure) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int pixels = width * height;
    if (idx >= pixels) {
        return;
    }

    int y = idx / width;
    float v = static_cast<float>(y) / static_cast<float>(height - 1);

    float3 bg_top = bg * 0.65f;
    float3 bg_bottom = bg * 1.2f + make_f3(0.004f, 0.005f, 0.010f);
    float3 bgg = bg_top * (1.0f - v) + bg_bottom * v;

    float r = accum[idx * 3 + 0];
    float g = accum[idx * 3 + 1];
    float b = accum[idx * 3 + 2];

    float3 c = make_f3(1.0f - expf(-r * exposure),
                       1.0f - expf(-g * exposure),
                       1.0f - expf(-b * exposure));

    float lum = clampf((c.x + c.y + c.z) * 0.5f, 0.0f, 1.0f);
    float3 out = bgg * (1.0f - 0.75f * lum) + c;

    out.x = powf(clampf(out.x, 0.0f, 1.0f), 1.0f / 2.2f);
    out.y = powf(clampf(out.y, 0.0f, 1.0f), 1.0f / 2.2f);
    out.z = powf(clampf(out.z, 0.0f, 1.0f), 1.0f / 2.2f);

    rgb[idx * 3 + 0] = static_cast<unsigned char>(clampf(out.x, 0.0f, 1.0f) * 255.0f + 0.5f);
    rgb[idx * 3 + 1] = static_cast<unsigned char>(clampf(out.y, 0.0f, 1.0f) * 255.0f + 0.5f);
    rgb[idx * 3 + 2] = static_cast<unsigned char>(clampf(out.z, 0.0f, 1.0f) * 255.0f + 0.5f);
}

static inline uint32_t hash32_host(uint32_t x) {
    x ^= x >> 17;
    x *= 0xED5AD4BBu;
    x ^= x >> 11;
    x *= 0xAC4C1B51u;
    x ^= x >> 15;
    x *= 0x31848BABu;
    x ^= x >> 14;
    return x;
}

static inline float rand01_host(uint32_t &state) {
    state = hash32_host(state);
    return static_cast<float>(state & 0x00FFFFFFu) * (1.0f / 16777216.0f);
}

static void integrate_particles_cpu(std::vector<float3> &pos,
                                    std::vector<float3> &vel,
                                    const std::vector<float3> &target,
                                    float dt,
                                    float alpha,
                                    float settle,
                                    float time,
                                    uint32_t seed,
                                    int frame_idx) {
    const int count = static_cast<int>(pos.size());
    for (int idx = 0; idx < count; ++idx) {
        float3 p = pos[idx];
        float3 v = vel[idx];
        const float3 t = target[idx];

        float3 diff = t - p;
        float dist = sqrtf(dot3(diff, diff) + 1e-10f);
        float3 to_target = diff / (dist + 1e-6f);

        float3 chaos = make_f3(
            sinf(2.2f * p.y + 1.15f * time) - cosf(1.7f * p.z - 0.8f * time),
            sinf(2.0f * p.z - 0.75f * time) + cosf(1.9f * p.x + 0.55f * time),
            sinf(2.35f * p.x + 1.05f * time) - cosf(1.55f * p.y - 0.65f * time));

        float3 vortex = cross3(make_f3(0.0f, 1.0f, 0.0f), p);
        float3 axis = norm3(make_f3(sinf(0.45f * time), 0.7f, cosf(0.45f * time)));
        float3 spiral = cross3(to_target, axis);

        float pull = (0.85f + 3.4f * alpha + 2.6f * settle) * (1.0f + 0.55f / (0.12f + dist));
        float3 attract = diff * pull;

        uint32_t s = static_cast<uint32_t>(idx) * 9781u + static_cast<uint32_t>(frame_idx) * 6271u + seed * 26699u;
        float3 jitter = make_f3(rand01_host(s) * 2.0f - 1.0f,
                                rand01_host(s) * 2.0f - 1.0f,
                                rand01_host(s) * 2.0f - 1.0f);

        float chaos_gain = 1.35f * (1.0f - 0.85f * alpha) * (1.0f - 0.95f * settle);
        float vortex_gain = 0.34f * (1.0f - 0.75f * settle);
        float jitter_gain = 0.27f * (1.0f - 0.92f * alpha) * (1.0f - 0.98f * settle);
        float spiral_gain = 0.42f * (1.0f - 0.70f * settle);

        float3 f_chaos = chaos * chaos_gain + vortex * vortex_gain - p * 0.22f + jitter * jitter_gain;
        float3 f_form = attract + spiral * spiral_gain - v * (0.24f + 0.50f * alpha + 0.95f * settle);

        float blend = clampf(alpha + 0.45f * settle, 0.0f, 1.0f);
        float3 f = f_chaos * (1.0f - blend) + f_form * blend;

        v = v + f * dt;
        v = v * (0.992f - 0.05f * alpha - 0.22f * settle);
        p = p + v * dt;

        float snap = (0.012f * alpha + 0.110f * settle) * (dt / 0.0125f);
        snap = clampf(snap, 0.0f, 0.30f);
        p = p * (1.0f - snap) + t * snap;
        v = v * (1.0f - 0.22f * settle);

        float r = sqrtf(dot3(p, p));
        if (r > 3.2f) {
            float3 n = p / r;
            p = n * 3.2f;
            v = v - n * (1.45f * dot3(v, n));
        }

        pos[idx] = p;
        vel[idx] = v;
    }
}

static void render_particles_cpu(const std::vector<float3> &pos,
                                 const std::vector<float3> &target,
                                 const CameraData &cam,
                                 int width,
                                 int height,
                                 float alpha,
                                 float settle,
                                 const float3 &chaos_color,
                                 const float3 &form_color,
                                 std::vector<float> &accum) {
    const int count = static_cast<int>(pos.size());

    for (int idx = 0; idx < count; ++idx) {
        float3 p = pos[idx];
        float3 rel = p - cam.pos;

        float x_cam = dot3(rel, cam.right);
        float y_cam = dot3(rel, cam.up);
        float z_cam = dot3(rel, cam.forward);
        if (z_cam <= 0.02f) {
            continue;
        }

        float ndc_x = x_cam / (z_cam * cam.tan_half_fov);
        float ndc_y = y_cam / (z_cam * cam.tan_half_fov);
        if (std::fabs(ndc_x) > 1.2f || std::fabs(ndc_y) > 1.2f) {
            continue;
        }

        float sx = (ndc_x * 0.5f + 0.5f) * (width - 1);
        float sy = (0.5f - ndc_y * 0.5f) * (height - 1);

        int x0 = static_cast<int>(floorf(sx));
        int y0 = static_cast<int>(floorf(sy));
        float fx = sx - x0;
        float fy = sy - y0;

        float3 diff = target[idx] - p;
        float convergence = expf(-2.6f * sqrtf(dot3(diff, diff)));
        float mixv = clampf(0.60f * alpha + 0.25f * convergence + 0.25f * settle, 0.0f, 1.0f);

        float3 base = chaos_color * (1.0f - mixv) + form_color * mixv;
        uint32_t hs = hash32_host(static_cast<uint32_t>(idx) * 747796405u + 2891336453u);
        float tint = 0.85f + 0.30f * (static_cast<float>(hs & 1023u) / 1023.0f);

        float intensity = (0.20f + 0.62f * convergence + 0.22f * alpha + 0.25f * settle) * clampf(1.45f / z_cam, 0.20f, 1.10f);
        intensity *= clampf(1.0f - 0.45f * settle * (1.0f - convergence), 0.55f, 1.0f);
        float3 c = base * (tint * intensity);

        int x1 = x0 + 1;
        int y1 = y0 + 1;
        float w00 = (1.0f - fx) * (1.0f - fy);
        float w10 = fx * (1.0f - fy);
        float w01 = (1.0f - fx) * fy;
        float w11 = fx * fy;

        if (x0 >= 0 && x0 < width && y0 >= 0 && y0 < height) {
            int pi = (y0 * width + x0) * 3;
            accum[pi + 0] += c.x * w00;
            accum[pi + 1] += c.y * w00;
            accum[pi + 2] += c.z * w00;
        }
        if (x1 >= 0 && x1 < width && y0 >= 0 && y0 < height) {
            int pi = (y0 * width + x1) * 3;
            accum[pi + 0] += c.x * w10;
            accum[pi + 1] += c.y * w10;
            accum[pi + 2] += c.z * w10;
        }
        if (x0 >= 0 && x0 < width && y1 >= 0 && y1 < height) {
            int pi = (y1 * width + x0) * 3;
            accum[pi + 0] += c.x * w01;
            accum[pi + 1] += c.y * w01;
            accum[pi + 2] += c.z * w01;
        }
        if (x1 >= 0 && x1 < width && y1 >= 0 && y1 < height) {
            int pi = (y1 * width + x1) * 3;
            accum[pi + 0] += c.x * w11;
            accum[pi + 1] += c.y * w11;
            accum[pi + 2] += c.z * w11;
        }
    }
}

static void tone_map_cpu(const std::vector<float> &accum,
                         std::vector<unsigned char> &rgb,
                         int width,
                         int height,
                         const float3 &bg,
                         float exposure) {
    const int pixels = width * height;
    for (int idx = 0; idx < pixels; ++idx) {
        int y = idx / width;
        float v = static_cast<float>(y) / static_cast<float>(height - 1);

        float3 bg_top = bg * 0.65f;
        float3 bg_bottom = bg * 1.2f + make_f3(0.004f, 0.005f, 0.010f);
        float3 bgg = bg_top * (1.0f - v) + bg_bottom * v;

        float r = accum[idx * 3 + 0];
        float g = accum[idx * 3 + 1];
        float b = accum[idx * 3 + 2];

        float3 c = make_f3(1.0f - expf(-r * exposure),
                           1.0f - expf(-g * exposure),
                           1.0f - expf(-b * exposure));

        float lum = clampf((c.x + c.y + c.z) * 0.5f, 0.0f, 1.0f);
        float3 out = bgg * (1.0f - 0.75f * lum) + c;

        out.x = powf(clampf(out.x, 0.0f, 1.0f), 1.0f / 2.2f);
        out.y = powf(clampf(out.y, 0.0f, 1.0f), 1.0f / 2.2f);
        out.z = powf(clampf(out.z, 0.0f, 1.0f), 1.0f / 2.2f);

        rgb[idx * 3 + 0] = static_cast<unsigned char>(clampf(out.x, 0.0f, 1.0f) * 255.0f + 0.5f);
        rgb[idx * 3 + 1] = static_cast<unsigned char>(clampf(out.y, 0.0f, 1.0f) * 255.0f + 0.5f);
        rgb[idx * 3 + 2] = static_cast<unsigned char>(clampf(out.z, 0.0f, 1.0f) * 255.0f + 0.5f);
    }
}

static bool parse_color(const std::string &s, std::array<float, 3> &dst) {
    std::string tmp = s;
    std::replace(tmp.begin(), tmp.end(), ',', ' ');
    std::stringstream ss(tmp);
    float r = 0.0f, g = 0.0f, b = 0.0f;
    if (!(ss >> r >> g >> b)) {
        return false;
    }
    dst[0] = clampf(r, 0.0f, 1.0f);
    dst[1] = clampf(g, 0.0f, 1.0f);
    dst[2] = clampf(b, 0.0f, 1.0f);
    return true;
}

static void print_usage() {
    std::cout
        << "Usage: ./bunny_flow_field [options]\n"
        << "  --obj <path>                OBJ file path\n"
        << "  --frame-dir <dir>           output PNG frame directory\n"
        << "  --width <int>               image width\n"
        << "  --height <int>              image height\n"
        << "  --frames <int>              number of frames\n"
        << "  --particles <int>           particle count\n"
        << "  --substeps <int>            integration substeps per frame\n"
        << "  --dt <float>                integration timestep\n"
        << "  --seed <int>                random seed\n"
        << "  --chaos-color r,g,b         color at chaos stage (0..1)\n"
        << "  --form-color r,g,b          color at structure stage (0..1)\n"
        << "  --bg-color r,g,b            background base color (0..1)\n";
}

static bool parse_args(int argc, char **argv, Config &cfg) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto need_value = [&](const char *name) -> const char * {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for " << name << "\n";
                std::exit(1);
            }
            return argv[++i];
        };

        if (arg == "--obj") {
            cfg.obj_path = need_value("--obj");
        } else if (arg == "--frame-dir") {
            cfg.frame_dir = need_value("--frame-dir");
        } else if (arg == "--width") {
            cfg.width = std::max(64, std::atoi(need_value("--width")));
        } else if (arg == "--height") {
            cfg.height = std::max(64, std::atoi(need_value("--height")));
        } else if (arg == "--frames") {
            cfg.frames = std::max(8, std::atoi(need_value("--frames")));
        } else if (arg == "--particles") {
            cfg.particles = std::max(1000, std::atoi(need_value("--particles")));
        } else if (arg == "--substeps") {
            cfg.substeps = std::max(1, std::atoi(need_value("--substeps")));
        } else if (arg == "--dt") {
            cfg.dt = std::max(0.0005f, std::strtof(need_value("--dt"), nullptr));
        } else if (arg == "--seed") {
            cfg.seed = static_cast<uint32_t>(std::strtoul(need_value("--seed"), nullptr, 10));
        } else if (arg == "--chaos-color") {
            if (!parse_color(need_value("--chaos-color"), cfg.chaos_color)) {
                std::cerr << "Invalid --chaos-color\n";
                return false;
            }
        } else if (arg == "--form-color") {
            if (!parse_color(need_value("--form-color"), cfg.form_color)) {
                std::cerr << "Invalid --form-color\n";
                return false;
            }
        } else if (arg == "--bg-color") {
            if (!parse_color(need_value("--bg-color"), cfg.bg_color)) {
                std::cerr << "Invalid --bg-color\n";
                return false;
            }
        } else if (arg == "--help" || arg == "-h") {
            print_usage();
            std::exit(0);
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            return false;
        }
    }
    return true;
}

static int parse_face_index(const std::string &token, int vcount) {
    if (token.empty()) {
        return 0;
    }
    std::string num = token;
    size_t slash = token.find('/');
    if (slash != std::string::npos) {
        num = token.substr(0, slash);
    }
    if (num.empty()) {
        return 0;
    }
    int idx = std::stoi(num);
    if (idx < 0) {
        idx = vcount + idx + 1;
    }
    return idx;
}

static bool load_obj(const std::string &path, std::vector<float3> &vertices, std::vector<int3> &triangles) {
    std::ifstream fin(path);
    if (!fin) {
        std::cerr << "Failed to open OBJ: " << path << "\n";
        return false;
    }

    vertices.clear();
    triangles.clear();

    std::string line;
    while (std::getline(fin, line)) {
        if (line.size() < 2) {
            continue;
        }

        if (line[0] == 'v' && std::isspace(static_cast<unsigned char>(line[1]))) {
            std::stringstream ss(line);
            char c;
            float x, y, z;
            ss >> c >> x >> y >> z;
            vertices.push_back(make_f3(x, y, z));
        } else if (line[0] == 'f' && std::isspace(static_cast<unsigned char>(line[1]))) {
            std::stringstream ss(line);
            char c;
            ss >> c;
            std::vector<int> face;
            std::string tok;
            while (ss >> tok) {
                int idx = parse_face_index(tok, static_cast<int>(vertices.size()));
                if (idx > 0) {
                    face.push_back(idx - 1);
                }
            }
            if (face.size() >= 3) {
                for (size_t k = 1; k + 1 < face.size(); ++k) {
                    int3 tri;
                    tri.x = face[0];
                    tri.y = face[k];
                    tri.z = face[k + 1];
                    triangles.push_back(tri);
                }
            }
        }
    }

    if (vertices.empty() || triangles.empty()) {
        std::cerr << "OBJ parse produced empty vertices/triangles\n";
        return false;
    }

    float3 mn = make_f3(std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
    float3 mx = make_f3(-std::numeric_limits<float>::max(), -std::numeric_limits<float>::max(), -std::numeric_limits<float>::max());

    for (const auto &v : vertices) {
        mn.x = std::min(mn.x, v.x);
        mn.y = std::min(mn.y, v.y);
        mn.z = std::min(mn.z, v.z);
        mx.x = std::max(mx.x, v.x);
        mx.y = std::max(mx.y, v.y);
        mx.z = std::max(mx.z, v.z);
    }

    float3 center = (mn + mx) * 0.5f;
    float ex = mx.x - mn.x;
    float ey = mx.y - mn.y;
    float ez = mx.z - mn.z;
    float extent = std::max(ex, std::max(ey, ez));
    float scale = (extent > 1e-8f) ? (2.2f / extent) : 1.0f;

    for (auto &v : vertices) {
        v = (v - center) * scale;
        v.y -= 0.1f;
    }

    return true;
}

static CameraData build_camera(float t) {
    float phi = 0.0f;
    float radius = 2.85f;
    float y = 0.74f;

    if (t < 0.70f) {
        float u = t / 0.70f;
        phi = -0.90f + 1.25f * u;
        radius = 2.95f - 0.20f * smoothstep01(u);
        y = 0.76f + 0.08f * std::sin(5.2f * u);
    } else {
        float u = (t - 0.70f) / 0.30f;
        phi = 1.0f + 6.28318530718f * u;
        radius = 3.05f;
        y = 0.62f + 0.06f * std::sin(6.28318f * u);
    }

    float3 pos = make_f3(radius * std::sin(phi), y, radius * std::cos(phi));
    float3 target = make_f3(0.0f, 0.0f, 0.0f);
    float3 forward = norm3(target - pos);
    float3 world_up = make_f3(0.0f, 1.0f, 0.0f);
    float3 right = norm3(cross3(forward, world_up));
    float3 up = norm3(cross3(right, forward));

    CameraData cam;
    cam.pos = pos;
    cam.right = right;
    cam.up = up;
    cam.forward = forward;
    cam.tan_half_fov = std::tan(44.0f * 0.5f * 3.1415926535f / 180.0f);
    return cam;
}

static bool render_gpu(const Config &cfg,
                       std::vector<float3> &h_pos,
                       std::vector<float3> &h_vel,
                       std::vector<float3> &h_target) {
    float3 *d_pos = nullptr;
    float3 *d_vel = nullptr;
    float3 *d_target = nullptr;
    float *d_accum = nullptr;
    unsigned char *d_rgb = nullptr;

    size_t p_bytes = static_cast<size_t>(cfg.particles) * sizeof(float3);
    size_t pixel_count = static_cast<size_t>(cfg.width) * static_cast<size_t>(cfg.height);
    size_t accum_values = pixel_count * 3;
    size_t accum_bytes = accum_values * sizeof(float);
    size_t rgb_bytes = pixel_count * 3;

    CUDA_CHECK(cudaMalloc(&d_pos, p_bytes));
    CUDA_CHECK(cudaMalloc(&d_vel, p_bytes));
    CUDA_CHECK(cudaMalloc(&d_target, p_bytes));
    CUDA_CHECK(cudaMalloc(&d_accum, accum_bytes));
    CUDA_CHECK(cudaMalloc(&d_rgb, rgb_bytes));

    CUDA_CHECK(cudaMemcpy(d_pos, h_pos.data(), p_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vel, h_vel.data(), p_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_target, h_target.data(), p_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_accum, 0, accum_bytes));

    std::vector<unsigned char> h_rgb(rgb_bytes);

    int p_blocks = (cfg.particles + 255) / 256;
    int accum_blocks = (static_cast<int>(accum_values) + 255) / 256;
    int pix_blocks = (static_cast<int>(pixel_count) + 255) / 256;

    float3 chaos_col = make_f3(cfg.chaos_color[0], cfg.chaos_color[1], cfg.chaos_color[2]);
    float3 form_col = make_f3(cfg.form_color[0], cfg.form_color[1], cfg.form_color[2]);
    float3 bg_col = make_f3(cfg.bg_color[0], cfg.bg_color[1], cfg.bg_color[2]);

    for (int f = 0; f < cfg.frames; ++f) {
        float t = (cfg.frames <= 1) ? 1.0f : static_cast<float>(f) / static_cast<float>(cfg.frames - 1);
        float alpha = morph_progress(t);
        float settle = settle_progress(t);
        float time = 7.5f * t;

        int substeps = cfg.substeps + (settle > 0.75f ? 1 : 0);
        for (int s = 0; s < substeps; ++s) {
            integrate_particles<<<p_blocks, 256>>>(
                d_pos,
                d_vel,
                d_target,
                cfg.particles,
                cfg.dt,
                alpha,
                settle,
                time + s * cfg.dt,
                cfg.seed,
                f * substeps + s);
        }

        float decay = (f == 0) ? 0.0f : mixf(0.935f, 0.55f, settle);
        decay_accum<<<accum_blocks, 256>>>(d_accum, static_cast<int>(accum_values), decay);

        CameraData cam = build_camera(t);
        render_particles<<<p_blocks, 256>>>(
            d_pos, d_target, cfg.particles, cam, cfg.width, cfg.height, alpha, settle, chaos_col, form_col, d_accum);

        float exposure = mixf(1.62f, 1.08f, settle);
        tone_map<<<pix_blocks, 256>>>(d_accum, d_rgb, cfg.width, cfg.height, bg_col, exposure);

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(h_rgb.data(), d_rgb, rgb_bytes, cudaMemcpyDeviceToHost));

        char name[256];
        std::snprintf(name, sizeof(name), "frame_%04d.png", f);
        std::filesystem::path frame_path = std::filesystem::path(cfg.frame_dir) / name;
        if (stbi_write_png(frame_path.string().c_str(), cfg.width, cfg.height, 3, h_rgb.data(), cfg.width * 3) == 0) {
            std::cerr << "Failed writing PNG: " << frame_path.string() << "\n";
            CUDA_CHECK(cudaFree(d_pos));
            CUDA_CHECK(cudaFree(d_vel));
            CUDA_CHECK(cudaFree(d_target));
            CUDA_CHECK(cudaFree(d_accum));
            CUDA_CHECK(cudaFree(d_rgb));
            return false;
        }

        if (f % 10 == 0 || f == cfg.frames - 1) {
            std::cout << "[FlowField][GPU] Frame " << f + 1 << "/" << cfg.frames << "\n";
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaFree(d_pos));
    CUDA_CHECK(cudaFree(d_vel));
    CUDA_CHECK(cudaFree(d_target));
    CUDA_CHECK(cudaFree(d_accum));
    CUDA_CHECK(cudaFree(d_rgb));
    return true;
}

static bool render_cpu(const Config &cfg,
                       std::vector<float3> &h_pos,
                       std::vector<float3> &h_vel,
                       std::vector<float3> &h_target) {
    size_t pixel_count = static_cast<size_t>(cfg.width) * static_cast<size_t>(cfg.height);
    size_t accum_values = pixel_count * 3;
    size_t rgb_bytes = pixel_count * 3;

    std::vector<float> accum(accum_values, 0.0f);
    std::vector<unsigned char> rgb(rgb_bytes, 0);

    float3 chaos_col = make_f3(cfg.chaos_color[0], cfg.chaos_color[1], cfg.chaos_color[2]);
    float3 form_col = make_f3(cfg.form_color[0], cfg.form_color[1], cfg.form_color[2]);
    float3 bg_col = make_f3(cfg.bg_color[0], cfg.bg_color[1], cfg.bg_color[2]);

    for (int f = 0; f < cfg.frames; ++f) {
        float t = (cfg.frames <= 1) ? 1.0f : static_cast<float>(f) / static_cast<float>(cfg.frames - 1);
        float alpha = morph_progress(t);
        float settle = settle_progress(t);
        float time = 7.5f * t;

        int substeps = cfg.substeps + (settle > 0.75f ? 1 : 0);
        for (int s = 0; s < substeps; ++s) {
            integrate_particles_cpu(
                h_pos, h_vel, h_target, cfg.dt, alpha, settle, time + s * cfg.dt, cfg.seed, f * substeps + s);
        }

        float decay = (f == 0) ? 0.0f : mixf(0.935f, 0.55f, settle);
        for (size_t i = 0; i < accum_values; ++i) {
            accum[i] *= decay;
        }

        CameraData cam = build_camera(t);
        render_particles_cpu(h_pos, h_target, cam, cfg.width, cfg.height, alpha, settle, chaos_col, form_col, accum);
        float exposure = mixf(1.62f, 1.08f, settle);
        tone_map_cpu(accum, rgb, cfg.width, cfg.height, bg_col, exposure);

        char name[256];
        std::snprintf(name, sizeof(name), "frame_%04d.png", f);
        std::filesystem::path frame_path = std::filesystem::path(cfg.frame_dir) / name;
        if (stbi_write_png(frame_path.string().c_str(), cfg.width, cfg.height, 3, rgb.data(), cfg.width * 3) == 0) {
            std::cerr << "Failed writing PNG: " << frame_path.string() << "\n";
            return false;
        }

        if (f % 10 == 0 || f == cfg.frames - 1) {
            std::cout << "[FlowField][CPU] Frame " << f + 1 << "/" << cfg.frames << "\n";
        }
    }

    return true;
}

int main(int argc, char **argv) {
    Config cfg;
    if (!parse_args(argc, argv, cfg)) {
        print_usage();
        return 1;
    }

    std::cout << "[FlowField] Loading OBJ: " << cfg.obj_path << "\n";

    std::vector<float3> vertices;
    std::vector<int3> triangles;
    if (!load_obj(cfg.obj_path, vertices, triangles)) {
        return 1;
    }

    std::cout << "[FlowField] Vertices: " << vertices.size() << ", Triangles: " << triangles.size() << "\n";
    std::cout << "[FlowField] Particles: " << cfg.particles << ", Frames: " << cfg.frames << "\n";

    std::vector<double> cdf(triangles.size());
    double total_area = 0.0;
    for (size_t i = 0; i < triangles.size(); ++i) {
        const int3 &tri = triangles[i];
        float3 a = vertices[tri.x];
        float3 b = vertices[tri.y];
        float3 c = vertices[tri.z];
        float3 ab = b - a;
        float3 ac = c - a;
        double area = 0.5 * static_cast<double>(len3(cross3(ab, ac)));
        if (area < 1e-16) {
            area = 1e-16;
        }
        total_area += area;
        cdf[i] = total_area;
    }

    std::mt19937 rng(cfg.seed);
    std::uniform_real_distribution<float> unif01(0.0f, 1.0f);
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    std::uniform_real_distribution<double> area_pick(0.0, total_area);

    std::vector<float3> h_target(cfg.particles);
    std::vector<float3> h_pos(cfg.particles);
    std::vector<float3> h_vel(cfg.particles);

    for (int i = 0; i < cfg.particles; ++i) {
        double pick = area_pick(rng);
        size_t tri_id = static_cast<size_t>(std::lower_bound(cdf.begin(), cdf.end(), pick) - cdf.begin());
        if (tri_id >= triangles.size()) {
            tri_id = triangles.size() - 1;
        }

        const int3 &tri = triangles[tri_id];
        float3 a = vertices[tri.x];
        float3 b = vertices[tri.y];
        float3 c = vertices[tri.z];

        float r1 = std::sqrt(unif01(rng));
        float r2 = unif01(rng);
        float wa = 1.0f - r1;
        float wb = r1 * (1.0f - r2);
        float wc = r1 * r2;
        h_target[i] = a * wa + b * wb + c * wc;

        float u = unif01(rng);
        float v = unif01(rng);
        float w = unif01(rng);
        float theta = 2.0f * 3.1415926535f * u;
        float phi = std::acos(2.0f * v - 1.0f);
        float radius = 1.9f * std::pow(w, 0.34f);
        float3 dir = make_f3(std::sin(phi) * std::cos(theta), std::cos(phi), std::sin(phi) * std::sin(theta));
        float3 noise = make_f3(gauss(rng), gauss(rng), gauss(rng)) * 0.22f;
        h_pos[i] = dir * radius + noise;
        h_vel[i] = make_f3(gauss(rng), gauss(rng), gauss(rng)) * 0.06f;
    }

    std::filesystem::create_directories(cfg.frame_dir);
    std::cout << "[FlowField] Rendering frames into: " << cfg.frame_dir << "\n";

    int device_count = 0;
    cudaError_t dev_err = cudaGetDeviceCount(&device_count);
    bool use_gpu = (dev_err == cudaSuccess && device_count > 0);

    if (use_gpu) {
        std::cout << "[FlowField] CUDA path enabled. device_count=" << device_count << "\n";
        if (!render_gpu(cfg, h_pos, h_vel, h_target)) {
            return 1;
        }
    } else {
        std::cout << "[FlowField] CUDA unavailable (" << cudaGetErrorString(dev_err)
                  << "), using CPU fallback path.\n";
        if (!render_cpu(cfg, h_pos, h_vel, h_target)) {
            return 1;
        }
    }

    std::cout << "[FlowField] Completed PNG sequence at: " << cfg.frame_dir << "\n";
    std::cout << "[FlowField] Build GIF with:\n";
    std::cout << "  ffmpeg -y -framerate 30 -i " << cfg.frame_dir << "/frame_%04d.png -vf palettegen "
              << cfg.frame_dir << "/palette.png\n";
    std::cout << "  ffmpeg -y -framerate 30 -i " << cfg.frame_dir
              << "/frame_%04d.png -i " << cfg.frame_dir
              << "/palette.png -lavfi \"paletteuse=dither=bayer:bayer_scale=3\" bunny_flow_field.gif\n";

    return 0;
}
