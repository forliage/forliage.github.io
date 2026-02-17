// Build:
//   nvcc -O3 -std=c++17 -arch=sm_89 wood.cu -o bunny_wood \
//       -lglfw -lGL -lGLEW -ldl -lpthread
// Run:
//   ./bunny_wood stanford-bunny.obj
//
// Controls:
//   Left drag   : look around
//   W A S D     : move around
//   Mouse wheel : move forward/back
//   Click swatch: switch wood material
//   R           : reset accumulation
//   P           : save screenshot (out.png)
//   Q / ESC     : quit

#include <cuda_runtime.h>

#include <GL/glew.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <cuda_gl_interop.h>

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>

#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define CUDA_CHECK(stmt) do { \
    cudaError_t _err = (stmt); \
    if (_err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d -> %s\n", #stmt, __FILE__, __LINE__, cudaGetErrorString(_err)); \
        std::exit(1); \
    } \
} while(0)

struct Vec3 {
    float x, y, z;
    __host__ __device__ Vec3() : x(0), y(0), z(0) {}
    __host__ __device__ Vec3(float X, float Y, float Z) : x(X), y(Y), z(Z) {}

    __host__ __device__ Vec3 operator + (const Vec3& b) const { return Vec3(x + b.x, y + b.y, z + b.z); }
    __host__ __device__ Vec3 operator - (const Vec3& b) const { return Vec3(x - b.x, y - b.y, z - b.z); }
    __host__ __device__ Vec3 operator - () const { return Vec3(-x, -y, -z); }
    __host__ __device__ Vec3 operator * (float s) const { return Vec3(x * s, y * s, z * s); }
    __host__ __device__ Vec3 operator / (float s) const {
        float inv = 1.0f / s;
        return Vec3(x * inv, y * inv, z * inv);
    }
    __host__ __device__ Vec3& operator += (const Vec3& b) {
        x += b.x; y += b.y; z += b.z;
        return *this;
    }
};

__host__ __device__ inline Vec3 operator*(float s, const Vec3& v) { return Vec3(v.x * s, v.y * s, v.z * s); }
__host__ __device__ inline Vec3 mul(const Vec3& a, const Vec3& b) { return Vec3(a.x * b.x, a.y * b.y, a.z * b.z); }
__host__ __device__ inline float dot(const Vec3& a, const Vec3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
__host__ __device__ inline Vec3 cross(const Vec3& a, const Vec3& b) {
    return Vec3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}
__host__ __device__ inline float length(const Vec3& v) { return sqrtf(dot(v, v)); }
__host__ __device__ inline Vec3 normalize(const Vec3& v) {
    float len = length(v);
    return (len > 0.0f) ? (v / len) : Vec3(0, 0, 0);
}
__host__ __device__ inline Vec3 clamp01(const Vec3& v) {
    return Vec3(
        fminf(fmaxf(v.x, 0.0f), 1.0f),
        fminf(fmaxf(v.y, 0.0f), 1.0f),
        fminf(fmaxf(v.z, 0.0f), 1.0f)
    );
}
__host__ __device__ inline float max_comp(const Vec3& v) {
    return fmaxf(v.x, fmaxf(v.y, v.z));
}
__host__ __device__ inline Vec3 reflect_dir(const Vec3& v, const Vec3& n) {
    return normalize(v - n * (2.0f * dot(v, n)));
}
__host__ __device__ inline Vec3 exp3(const Vec3& a) {
    return Vec3(expf(a.x), expf(a.y), expf(a.z));
}

struct Ray {
    Vec3 o;
    Vec3 d;
};

struct RNG {
    uint32_t state;
    __host__ __device__ explicit RNG(uint32_t s = 1u) : state(s) {}

    __host__ __device__ inline uint32_t next_u32() {
        uint32_t x = state;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        state = x;
        return x;
    }

    __host__ __device__ inline float next_f() {
        return (next_u32() & 0x00FFFFFFu) * (1.0f / 16777216.0f);
    }
};

struct Triangle {
    Vec3 v0, v1, v2;
    Vec3 n0, n1, n2;
};

struct AABB {
    Vec3 bmin, bmax;
};

struct BVHNode {
    AABB box;
    int left;
    int right;
    int start;
    int count;
};

__host__ __device__ inline AABB tri_bounds(const Triangle& t) {
    AABB b;
    b.bmin = Vec3(
        fminf(t.v0.x, fminf(t.v1.x, t.v2.x)),
        fminf(t.v0.y, fminf(t.v1.y, t.v2.y)),
        fminf(t.v0.z, fminf(t.v1.z, t.v2.z))
    );
    b.bmax = Vec3(
        fmaxf(t.v0.x, fmaxf(t.v1.x, t.v2.x)),
        fmaxf(t.v0.y, fmaxf(t.v1.y, t.v2.y)),
        fmaxf(t.v0.z, fmaxf(t.v1.z, t.v2.z))
    );
    const float eps = 1e-4f;
    b.bmin = b.bmin - Vec3(eps, eps, eps);
    b.bmax = b.bmax + Vec3(eps, eps, eps);
    return b;
}

__host__ __device__ inline AABB aabb_union(const AABB& a, const AABB& b) {
    AABB r;
    r.bmin = Vec3(
        fminf(a.bmin.x, b.bmin.x),
        fminf(a.bmin.y, b.bmin.y),
        fminf(a.bmin.z, b.bmin.z)
    );
    r.bmax = Vec3(
        fmaxf(a.bmax.x, b.bmax.x),
        fmaxf(a.bmax.y, b.bmax.y),
        fmaxf(a.bmax.z, b.bmax.z)
    );
    return r;
}

static inline Vec3 tri_centroid(const Triangle& t) {
    return (t.v0 + t.v1 + t.v2) * (1.0f / 3.0f);
}

static inline AABB bounds_of_range(const std::vector<Triangle>& tris,
                                   const std::vector<int>& idx,
                                   int start, int count)
{
    AABB b = tri_bounds(tris[idx[start]]);
    for (int i = 1; i < count; i++) {
        b = aabb_union(b, tri_bounds(tris[idx[start + i]]));
    }
    return b;
}

static int build_bvh_recursive(std::vector<BVHNode>& nodes,
                               std::vector<int>& tri_idx,
                               const std::vector<Triangle>& tris,
                               int start, int count)
{
    BVHNode node{};
    node.left = node.right = -1;
    node.start = start;
    node.count = count;
    node.box = bounds_of_range(tris, tri_idx, start, count);

    int my = (int)nodes.size();
    nodes.push_back(node);

    if (count <= 4) {
        return my;
    }

    Vec3 cmin(1e30f, 1e30f, 1e30f);
    Vec3 cmax(-1e30f, -1e30f, -1e30f);
    for (int i = 0; i < count; i++) {
        Vec3 c = tri_centroid(tris[tri_idx[start + i]]);
        cmin = Vec3(fminf(cmin.x, c.x), fminf(cmin.y, c.y), fminf(cmin.z, c.z));
        cmax = Vec3(fmaxf(cmax.x, c.x), fmaxf(cmax.y, c.y), fmaxf(cmax.z, c.z));
    }

    Vec3 ext = cmax - cmin;
    int axis = 0;
    if (ext.y > ext.x && ext.y > ext.z) axis = 1;
    else if (ext.z > ext.x) axis = 2;

    int mid = start + count / 2;
    std::nth_element(tri_idx.begin() + start,
                     tri_idx.begin() + mid,
                     tri_idx.begin() + start + count,
                     [&](int a, int b) {
                         Vec3 ca = tri_centroid(tris[a]);
                         Vec3 cb = tri_centroid(tris[b]);
                         if (axis == 0) return ca.x < cb.x;
                         if (axis == 1) return ca.y < cb.y;
                         return ca.z < cb.z;
                     });

    int left_count = mid - start;
    int right_count = count - left_count;

    int left = build_bvh_recursive(nodes, tri_idx, tris, start, left_count);
    int right = build_bvh_recursive(nodes, tri_idx, tris, mid, right_count);

    nodes[my].left = left;
    nodes[my].right = right;
    nodes[my].start = -1;
    nodes[my].count = 0;
    return my;
}

enum HitType : int {
    HIT_NONE = -1,
    HIT_BUNNY = 0,
    HIT_GROUND = 1,
    HIT_LIGHT = 2,
};

struct Hit {
    float t;
    Vec3 n;
    int type;
    float u, v;
};

struct DeviceScene {
    Triangle* tris;
    int* tri_idx;
    BVHNode* nodes;
    int tri_count;
    int node_count;
};

struct Camera {
    Vec3 pos;
    Vec3 forward;
    Vec3 right;
    Vec3 up;
    float fov_y;
};

struct RenderParams {
    int width;
    int height;
    int frame_index;
    int spp_per_frame;
    int max_bounces;

    Vec3 wood_light;
    Vec3 wood_dark;
    float bunny_roughness;
    float bunny_spec_strength;
    float wood_grain_scale;
    float wood_ring_scale;
    float wood_ring_strength;
    float wood_pore_strength;

    float ground_roughness;

    Vec3 light_center;
    Vec3 light_u;      // half-vector along U
    Vec3 light_v;      // half-vector along V
    Vec3 light_n;      // emission side normal
    Vec3 light_emission;
    float light_area;
};

struct UIParams {
    int sidebar_w;
    int pad;
    int swatch_h;
    int gap;
    int top;
};

struct WoodPreset {
    const char* name;
    Vec3 light_color;
    Vec3 dark_color;
    float roughness;
    float spec_strength;
    float grain_scale;
    float ring_scale;
    float ring_strength;
    float pore_strength;
};

static constexpr int kPresetCount = 10;
static const WoodPreset kPresets[kPresetCount] = {
    {"Oak",       Vec3(0.78f, 0.62f, 0.42f), Vec3(0.45f, 0.30f, 0.17f), 0.30f, 0.16f,  8.0f, 14.0f, 0.65f, 0.16f},
    {"Walnut",    Vec3(0.47f, 0.32f, 0.22f), Vec3(0.23f, 0.14f, 0.09f), 0.26f, 0.14f,  9.5f, 18.0f, 0.70f, 0.20f},
    {"Maple",     Vec3(0.86f, 0.78f, 0.62f), Vec3(0.67f, 0.56f, 0.40f), 0.36f, 0.12f,  7.2f, 10.0f, 0.45f, 0.12f},
    {"Cherry",    Vec3(0.74f, 0.46f, 0.33f), Vec3(0.47f, 0.24f, 0.16f), 0.28f, 0.16f,  8.8f, 15.5f, 0.60f, 0.15f},
    {"Mahogany",  Vec3(0.62f, 0.30f, 0.20f), Vec3(0.30f, 0.11f, 0.08f), 0.24f, 0.18f, 10.5f, 19.0f, 0.72f, 0.22f},
    {"Teak",      Vec3(0.66f, 0.52f, 0.34f), Vec3(0.40f, 0.28f, 0.16f), 0.27f, 0.15f,  8.4f, 13.0f, 0.56f, 0.16f},
    {"Pine",      Vec3(0.82f, 0.70f, 0.45f), Vec3(0.59f, 0.45f, 0.24f), 0.38f, 0.10f,  6.4f,  9.0f, 0.40f, 0.11f},
    {"Ash",       Vec3(0.79f, 0.73f, 0.60f), Vec3(0.57f, 0.50f, 0.38f), 0.34f, 0.11f,  7.5f, 11.0f, 0.44f, 0.13f},
    {"Birch",     Vec3(0.88f, 0.83f, 0.70f), Vec3(0.70f, 0.61f, 0.46f), 0.35f, 0.10f,  6.8f,  8.8f, 0.38f, 0.12f},
    {"Rosewood",  Vec3(0.45f, 0.22f, 0.18f), Vec3(0.20f, 0.07f, 0.06f), 0.22f, 0.20f, 11.5f, 21.0f, 0.78f, 0.24f},
};

__constant__ float3 c_preset_ui_color[kPresetCount];

__host__ __device__ inline Vec3 preset_display_color(const Vec3& ui_color) {
    return clamp01(Vec3(powf(ui_color.x, 1.0f / 2.2f), powf(ui_color.y, 1.0f / 2.2f), powf(ui_color.z, 1.0f / 2.2f)));
}

__host__ __device__ inline void swatch_rect(const UIParams& ui, int W, int idx,
                                            int& x0, int& y0, int& x1, int& y1)
{
    x0 = W - ui.sidebar_w + ui.pad;
    x1 = W - ui.pad;
    y0 = ui.top + idx * (ui.swatch_h + ui.gap);
    y1 = y0 + ui.swatch_h;
}

__host__ __device__ inline bool in_rect(int x, int y, int x0, int y0, int x1, int y1) {
    return (x >= x0 && x < x1 && y >= y0 && y < y1);
}

__device__ inline bool intersect_aabb(const AABB& box, const Ray& r, float tmin, float tmax) {
    for (int a = 0; a < 3; a++) {
        float ro = (a == 0) ? r.o.x : (a == 1 ? r.o.y : r.o.z);
        float rd = (a == 0) ? r.d.x : (a == 1 ? r.d.y : r.d.z);
        float bmin = (a == 0) ? box.bmin.x : (a == 1 ? box.bmin.y : box.bmin.z);
        float bmax = (a == 0) ? box.bmax.x : (a == 1 ? box.bmax.y : box.bmax.z);

        float invD = 1.0f / rd;
        float t0 = (bmin - ro) * invD;
        float t1 = (bmax - ro) * invD;
        if (invD < 0.0f) {
            float tmp = t0;
            t0 = t1;
            t1 = tmp;
        }
        tmin = fmaxf(tmin, t0);
        tmax = fminf(tmax, t1);
        if (tmax <= tmin) return false;
    }
    return true;
}

__device__ inline bool intersect_triangle(const Triangle& tri, const Ray& r, float& t, float& u, float& v) {
    const Vec3 e1 = tri.v1 - tri.v0;
    const Vec3 e2 = tri.v2 - tri.v0;
    const Vec3 p = cross(r.d, e2);
    const float det = dot(e1, p);
    if (fabsf(det) < 1e-8f) return false;

    float invDet = 1.0f / det;
    const Vec3 s = r.o - tri.v0;
    u = dot(s, p) * invDet;
    if (u < 0.0f || u > 1.0f) return false;

    const Vec3 q = cross(s, e1);
    v = dot(r.d, q) * invDet;
    if (v < 0.0f || (u + v) > 1.0f) return false;

    t = dot(e2, q) * invDet;
    return t > 1e-4f;
}

__device__ inline bool traverse_bvh(const DeviceScene& sc, const Ray& r, Hit& hit) {
    int stack[96];
    int sp = 0;
    stack[sp++] = 0;

    bool any = false;
    while (sp) {
        int ni = stack[--sp];
        const BVHNode& node = sc.nodes[ni];

        if (!intersect_aabb(node.box, r, 1e-4f, hit.t)) continue;

        if (node.left < 0 && node.right < 0) {
            for (int i = 0; i < node.count; i++) {
                int tid = sc.tri_idx[node.start + i];
                const Triangle& tri = sc.tris[tid];

                float t, u, v;
                if (intersect_triangle(tri, r, t, u, v) && t < hit.t) {
                    hit.t = t;
                    hit.u = u;
                    hit.v = v;

                    Vec3 n = tri.n0 * (1.0f - u - v) + tri.n1 * u + tri.n2 * v;
                    if (length(n) < 1e-6f) {
                        n = normalize(cross(tri.v1 - tri.v0, tri.v2 - tri.v0));
                    } else {
                        n = normalize(n);
                    }
                    hit.n = n;
                    hit.type = HIT_BUNNY;
                    any = true;
                }
            }
        } else {
            if (node.left >= 0) stack[sp++] = node.left;
            if (node.right >= 0) stack[sp++] = node.right;
        }
    }
    return any;
}

__device__ inline bool intersect_ground(const Ray& r, Hit& hit) {
    if (fabsf(r.d.y) < 1e-6f) return false;
    float t = -r.o.y / r.d.y;
    if (t > 1e-4f && t < hit.t) {
        hit.t = t;
        hit.n = Vec3(0, 1, 0);
        hit.type = HIT_GROUND;
        return true;
    }
    return false;
}

__device__ inline bool intersect_rect_light(const Ray& r, const RenderParams& rp, Hit& hit) {
    float denom = dot(r.d, rp.light_n);
    if (denom >= -1e-6f) return false; // one-sided: only emitting side

    float t = dot(rp.light_center - r.o, rp.light_n) / denom;
    if (t <= 1e-4f || t >= hit.t) return false;

    Vec3 p = r.o + r.d * t;
    Vec3 rel = p - rp.light_center;

    float uu = dot(rel, rp.light_u) / fmaxf(dot(rp.light_u, rp.light_u), 1e-8f);
    float vv = dot(rel, rp.light_v) / fmaxf(dot(rp.light_v, rp.light_v), 1e-8f);

    if (fabsf(uu) <= 1.0f && fabsf(vv) <= 1.0f) {
        hit.t = t;
        hit.n = rp.light_n;
        hit.type = HIT_LIGHT;
        return true;
    }
    return false;
}

__device__ inline bool intersect_scene(const DeviceScene& sc, const RenderParams& rp, const Ray& r, Hit& hit) {
    hit.t = 1e30f;
    hit.type = HIT_NONE;
    hit.u = hit.v = 0.0f;

    traverse_bvh(sc, r, hit);
    intersect_ground(r, hit);
    intersect_rect_light(r, rp, hit);

    return hit.type != HIT_NONE;
}

__device__ inline bool occluded(const DeviceScene& sc, const Ray& r, float max_t) {
    Hit h;
    h.t = max_t;
    h.type = HIT_NONE;
    traverse_bvh(sc, r, h);
    intersect_ground(r, h);
    return h.type != HIT_NONE;
}

__device__ inline float schlick(float cosTheta, float etaI, float etaT) {
    float r0 = (etaI - etaT) / (etaI + etaT);
    r0 = r0 * r0;
    float m = fmaxf(0.0f, 1.0f - cosTheta);
    return r0 + (1.0f - r0) * m * m * m * m * m;
}

__device__ inline bool refract_eta(const Vec3& v, const Vec3& n, float eta, Vec3& refr) {
    float cosi = dot(-v, n);
    float sint2 = eta * eta * fmaxf(0.0f, 1.0f - cosi * cosi);
    if (sint2 > 1.0f) return false;
    float cost = sqrtf(fmaxf(0.0f, 1.0f - sint2));
    refr = normalize(eta * v + (eta * cosi - cost) * n);
    return true;
}

__device__ inline float power_heuristic(float pdfA, float pdfB) {
    float a2 = pdfA * pdfA;
    float b2 = pdfB * pdfB;
    return a2 / fmaxf(a2 + b2, 1e-20f);
}

__device__ inline void make_basis(const Vec3& n, Vec3& t, Vec3& b) {
    if (fabsf(n.y) < 0.999f) {
        t = normalize(cross(Vec3(0, 1, 0), n));
    } else {
        t = normalize(cross(Vec3(1, 0, 0), n));
    }
    b = cross(n, t);
}

__device__ inline Vec3 cosine_sample_hemisphere(RNG& rng) {
    float u1 = rng.next_f();
    float u2 = rng.next_f();
    float r = sqrtf(u1);
    float phi = 2.0f * (float)M_PI * u2;
    float x = r * cosf(phi);
    float z = r * sinf(phi);
    float y = sqrtf(fmaxf(0.0f, 1.0f - u1));
    return Vec3(x, y, z);
}

__device__ inline Vec3 to_world(const Vec3& local, const Vec3& n) {
    Vec3 t, b;
    make_basis(n, t, b);
    return normalize(t * local.x + n * local.y + b * local.z);
}

__device__ inline float roughness_to_exp(float roughness) {
    float r = fmaxf(roughness, 0.02f);
    return fmaxf(2.0f / (r * r) - 2.0f, 1.0f);
}

__device__ inline Vec3 sample_power_cosine_axis(const Vec3& axis, float exponent, RNG& rng) {
    float u1 = rng.next_f();
    float u2 = rng.next_f();

    float cosTheta = powf(u1, 1.0f / (exponent + 1.0f));
    float sinTheta = sqrtf(fmaxf(0.0f, 1.0f - cosTheta * cosTheta));
    float phi = 2.0f * (float)M_PI * u2;

    Vec3 local(sinTheta * cosf(phi), cosTheta, sinTheta * sinf(phi));
    return to_world(local, axis);
}

__device__ inline float power_cosine_pdf(const Vec3& axis, const Vec3& dir, float exponent) {
    float c = fmaxf(dot(axis, dir), 0.0f);
    return (exponent + 1.0f) * powf(c, exponent) * (1.0f / (2.0f * (float)M_PI));
}

__device__ inline Vec3 ground_albedo(const Vec3& p) {
    float s = 1.25f;
    int xi = (int)floorf(p.x * s);
    int zi = (int)floorf(p.z * s);
    bool odd = ((xi + zi) & 1) != 0;
    Vec3 a = odd ? Vec3(0.27f, 0.26f, 0.24f) : Vec3(0.14f, 0.14f, 0.15f);

    float r = sqrtf(p.x * p.x + p.z * p.z);
    float fade = 1.0f - 0.10f * fminf(r * 0.18f, 1.0f);
    return a * fade;
}

struct GroundEval {
    Vec3 f;
    float pdf;
};

__device__ inline GroundEval eval_ground_bsdf(const Vec3& n, const Vec3& wo, const Vec3& wi,
                                              const Vec3& p, float roughness)
{
    GroundEval e{};

    float NoV = dot(n, wo);
    float NoL = dot(n, wi);
    if (NoV <= 0.0f || NoL <= 0.0f) {
        e.f = Vec3(0, 0, 0);
        e.pdf = 0.0f;
        return e;
    }

    const float kd = 0.72f;
    const float ks = 0.28f;

    Vec3 albedo = ground_albedo(p);
    Vec3 diff = albedo * (1.0f / (float)M_PI);

    float expn = roughness_to_exp(roughness);
    Vec3 wr = reflect_dir(-wo, n);
    float ca = fmaxf(dot(wr, wi), 0.0f);
    Vec3 spec = Vec3(1, 1, 1) * ((expn + 2.0f) * (1.0f / (2.0f * (float)M_PI))) * powf(ca, expn);

    float pdf_diff = NoL * (1.0f / (float)M_PI);
    float pdf_spec = power_cosine_pdf(wr, wi, expn);

    e.f = kd * diff + ks * spec;
    e.pdf = kd * pdf_diff + ks * pdf_spec;
    return e;
}

struct BSDFSample {
    Vec3 wi;
    Vec3 weight;
    Vec3 f;
    float pdf;
    bool delta;
    bool valid;
    bool toggled_inside;
};

__device__ inline BSDFSample sample_ground(const Vec3& n, const Vec3& wo, const Vec3& p,
                                           float roughness, RNG& rng)
{
    BSDFSample s{};
    s.delta = false;
    s.valid = true;
    s.toggled_inside = false;

    const float kd = 0.72f;
    const float ks = 0.28f;
    float p_spec = ks / (kd + ks);

    float choose = rng.next_f();
    Vec3 wi;

    if (choose < p_spec) {
        float expn = roughness_to_exp(roughness);
        Vec3 wr = reflect_dir(-wo, n);

        bool ok = false;
        for (int i = 0; i < 5; i++) {
            wi = sample_power_cosine_axis(wr, expn, rng);
            if (dot(n, wi) > 0.0f) {
                ok = true;
                break;
            }
        }
        if (!ok) wi = wr;
    } else {
        wi = to_world(cosine_sample_hemisphere(rng), n);
    }

    GroundEval e = eval_ground_bsdf(n, wo, wi, p, roughness);
    if (e.pdf <= 1e-8f) {
        s.valid = false;
        return s;
    }

    s.wi = wi;
    s.f = e.f;
    s.pdf = e.pdf;
    s.weight = Vec3(1, 1, 1);
    return s;
}

__device__ inline float fractf(float x) { return x - floorf(x); }

__device__ inline float hash13(const Vec3& p) {
    float h = dot(p, Vec3(127.1f, 311.7f, 74.7f));
    return fractf(sinf(h) * 43758.5453123f);
}

__device__ inline float value_noise3d(const Vec3& p) {
    Vec3 i(floorf(p.x), floorf(p.y), floorf(p.z));
    Vec3 f(fractf(p.x), fractf(p.y), fractf(p.z));
    Vec3 u = mul(f, mul(f, Vec3(3.0f, 3.0f, 3.0f) - 2.0f * f));

    float n000 = hash13(i + Vec3(0, 0, 0));
    float n100 = hash13(i + Vec3(1, 0, 0));
    float n010 = hash13(i + Vec3(0, 1, 0));
    float n110 = hash13(i + Vec3(1, 1, 0));
    float n001 = hash13(i + Vec3(0, 0, 1));
    float n101 = hash13(i + Vec3(1, 0, 1));
    float n011 = hash13(i + Vec3(0, 1, 1));
    float n111 = hash13(i + Vec3(1, 1, 1));

    float nx00 = n000 * (1.0f - u.x) + n100 * u.x;
    float nx10 = n010 * (1.0f - u.x) + n110 * u.x;
    float nx01 = n001 * (1.0f - u.x) + n101 * u.x;
    float nx11 = n011 * (1.0f - u.x) + n111 * u.x;
    float nxy0 = nx00 * (1.0f - u.y) + nx10 * u.y;
    float nxy1 = nx01 * (1.0f - u.y) + nx11 * u.y;
    return nxy0 * (1.0f - u.z) + nxy1 * u.z;
}

__device__ inline float fbm3(const Vec3& p) {
    float sum = 0.0f;
    float amp = 0.5f;
    float freq = 1.0f;
    for (int i = 0; i < 5; i++) {
        sum += amp * value_noise3d(p * freq);
        amp *= 0.5f;
        freq *= 2.03f;
    }
    return sum;
}

__device__ inline Vec3 wood_albedo(const Vec3& p, const RenderParams& rp) {
    const float twopi = 6.28318530718f;
    Vec3 q = p * rp.wood_grain_scale;
    float warp = fbm3(q * 0.45f) * 2.0f - 1.0f;

    float xw = p.x + 0.18f * warp;
    float zw = p.z - 0.12f * warp;
    float radial = sqrtf(xw * xw + zw * zw);

    float rings = radial * rp.wood_ring_scale + warp * rp.wood_ring_strength;
    float ring_band = 0.5f + 0.5f * sinf(rings * twopi);
    float long_grain = fbm3(Vec3(q.x * 0.8f, q.y * 3.6f, q.z * 0.8f));
    float streak = powf(fmaxf(0.0f, sinf((p.y * rp.wood_grain_scale + warp * 0.8f) * 10.0f)), 2.0f);

    float t = 0.58f * ring_band + 0.32f * long_grain + 0.10f * streak;
    t = fminf(fmaxf(t, 0.0f), 1.0f);

    Vec3 base = rp.wood_dark * (1.0f - t) + rp.wood_light * t;

    float pore = fbm3(Vec3(q.x * 3.3f, q.y * 6.0f, q.z * 3.3f));
    float pore_mask = powf(fmaxf(0.0f, pore - 0.45f), 2.2f);
    base = base * (1.0f - rp.wood_pore_strength * pore_mask);

    return clamp01(base);
}

__device__ inline GroundEval eval_wood_bsdf(const Vec3& n, const Vec3& wo, const Vec3& wi,
                                            const Vec3& p, const RenderParams& rp)
{
    GroundEval e{};
    float NoV = dot(n, wo);
    float NoL = dot(n, wi);
    if (NoV <= 0.0f || NoL <= 0.0f) {
        e.f = Vec3(0, 0, 0);
        e.pdf = 0.0f;
        return e;
    }

    float ks = fminf(fmaxf(rp.bunny_spec_strength, 0.02f), 0.35f);
    float kd = 1.0f - ks;

    Vec3 albedo = wood_albedo(p, rp);
    Vec3 diff = albedo * (1.0f / (float)M_PI);

    float expn = roughness_to_exp(rp.bunny_roughness);
    Vec3 wr = reflect_dir(-wo, n);
    float ca = fmaxf(dot(wr, wi), 0.0f);
    Vec3 spec = Vec3(1, 1, 1) * ((expn + 2.0f) * (1.0f / (2.0f * (float)M_PI))) * powf(ca, expn);

    float pdf_diff = NoL * (1.0f / (float)M_PI);
    float pdf_spec = power_cosine_pdf(wr, wi, expn);

    e.f = kd * diff + ks * spec;
    e.pdf = kd * pdf_diff + ks * pdf_spec;
    return e;
}

__device__ inline BSDFSample sample_wood(const Vec3& n, const Vec3& wo, const Vec3& p,
                                         const RenderParams& rp, RNG& rng)
{
    BSDFSample s{};
    s.delta = false;
    s.valid = true;
    s.toggled_inside = false;

    float ks = fminf(fmaxf(rp.bunny_spec_strength, 0.02f), 0.35f);
    float kd = 1.0f - ks;
    float p_spec = ks / (kd + ks);

    Vec3 wi;
    if (rng.next_f() < p_spec) {
        float expn = roughness_to_exp(rp.bunny_roughness);
        Vec3 wr = reflect_dir(-wo, n);
        bool ok = false;
        for (int i = 0; i < 6; i++) {
            wi = sample_power_cosine_axis(wr, expn, rng);
            if (dot(n, wi) > 0.0f) {
                ok = true;
                break;
            }
        }
        if (!ok) wi = wr;
    } else {
        wi = to_world(cosine_sample_hemisphere(rng), n);
    }

    GroundEval e = eval_wood_bsdf(n, wo, wi, p, rp);
    if (e.pdf <= 1e-8f) {
        s.valid = false;
        return s;
    }

    s.wi = wi;
    s.f = e.f;
    s.pdf = e.pdf;
    s.weight = Vec3(1, 1, 1);
    return s;
}

__device__ inline Vec3 environment_radiance(const Vec3& d) {
    float t = 0.5f * (d.y + 1.0f);
    Vec3 sky = (1.0f - t) * Vec3(0.03f, 0.035f, 0.042f) + t * Vec3(0.22f, 0.26f, 0.31f);
    float sunish = powf(fmaxf(dot(d, normalize(Vec3(0.18f, 0.95f, 0.18f))), 0.0f), 200.0f);
    sky += Vec3(0.4f, 0.35f, 0.25f) * (0.2f * sunish);
    return sky;
}

__device__ inline Vec3 sample_rect_light(const RenderParams& rp, RNG& rng, Vec3& nlight, float& pdfA) {
    float su = rng.next_f() * 2.0f - 1.0f;
    float sv = rng.next_f() * 2.0f - 1.0f;

    Vec3 p = rp.light_center + rp.light_u * su + rp.light_v * sv;
    nlight = rp.light_n;
    pdfA = 1.0f / rp.light_area;
    return p;
}

__device__ inline float light_pdf_solid_angle(const RenderParams& rp,
                                              const Vec3& shading_p,
                                              const Vec3& light_p,
                                              const Vec3& wi)
{
    Vec3 d = light_p - shading_p;
    float dist2 = dot(d, d);
    float cosL = fmaxf(dot(rp.light_n, -wi), 0.0f);
    if (cosL <= 1e-8f) return 0.0f;
    return dist2 / (cosL * rp.light_area);
}

__global__ void render_kernel(Vec3* accum,
                              int* accum_n,
                              uint32_t* rng_states,
                              DeviceScene sc,
                              Camera cam,
                              RenderParams rp)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= rp.width || y >= rp.height) return;

    int idx = y * rp.width + x;
    RNG rng(rng_states[idx] ^ (uint32_t)(rp.frame_index * 9781 + idx * 6271 + 17));

    Vec3 sum(0, 0, 0);

    for (int s = 0; s < rp.spp_per_frame; s++) {
        float sx = rng.next_f();
        float sy = rng.next_f();

        float u = ((x + sx) / (float)rp.width) * 2.0f - 1.0f;
        float v = 1.0f - ((y + sy) / (float)rp.height) * 2.0f;

        float aspect = rp.width / (float)rp.height;
        float tanHalf = tanf(0.5f * cam.fov_y);

        Vec3 dir = normalize(cam.forward + cam.right * (u * aspect * tanHalf) + cam.up * (v * tanHalf));
        Ray ray{cam.pos, dir};

        Vec3 throughput(1, 1, 1);
        Vec3 radiance(0, 0, 0);

        bool prev_delta = false;
        float prev_bsdf_pdf = 1.0f;
        Vec3 prev_surface_p(0, 0, 0);

        for (int bounce = 0; bounce < rp.max_bounces; bounce++) {
            Hit hit;
            if (!intersect_scene(sc, rp, ray, hit)) {
                radiance += mul(throughput, environment_radiance(ray.d));
                break;
            }

            Vec3 p = ray.o + ray.d * hit.t;
            Vec3 n = normalize(hit.n);

            if (hit.type == HIT_LIGHT) {
                float mis_w = 1.0f;
                if (!prev_delta && bounce > 0) {
                    float pdf_light = light_pdf_solid_angle(rp, prev_surface_p, p, ray.d);
                    if (pdf_light > 0.0f) {
                        mis_w = power_heuristic(prev_bsdf_pdf, pdf_light);
                    }
                }
                radiance += mul(throughput, rp.light_emission * mis_w);
                break;
            }

            Vec3 wo = -ray.d;

            if (hit.type == HIT_GROUND || hit.type == HIT_BUNNY) {
                bool is_bunny = (hit.type == HIT_BUNNY);

                // Next-event estimation with MIS
                {
                    Vec3 ln;
                    float pdfA = 0.0f;
                    Vec3 lp = sample_rect_light(rp, rng, ln, pdfA);

                    Vec3 toL = lp - p;
                    float dist2 = dot(toL, toL);
                    float dist = sqrtf(dist2);
                    Vec3 wi = toL / fmaxf(dist, 1e-8f);

                    float NoL = fmaxf(dot(n, wi), 0.0f);
                    float LoN = fmaxf(dot(ln, -wi), 0.0f);
                    if (NoL > 0.0f && LoN > 0.0f) {
                        float pdf_light = dist2 / (fmaxf(LoN * rp.light_area, 1e-8f));

                        Ray shadow;
                        shadow.o = p + n * 1e-4f;
                        shadow.d = wi;
                        bool blocked = occluded(sc, shadow, dist - 2e-4f);

                        if (!blocked) {
                            GroundEval e = is_bunny
                                           ? eval_wood_bsdf(n, wo, wi, p, rp)
                                           : eval_ground_bsdf(n, wo, wi, p, rp.ground_roughness);
                            if (e.pdf > 1e-8f && pdf_light > 1e-8f) {
                                float w = power_heuristic(pdf_light, e.pdf);
                                Vec3 contrib = mul(throughput, e.f);
                                contrib = contrib * (NoL * w / pdf_light);
                                contrib = mul(contrib, rp.light_emission);
                                radiance += contrib;
                            }
                        }
                    }
                    (void)pdfA;
                }

                BSDFSample bs = is_bunny
                                ? sample_wood(n, wo, p, rp, rng)
                                : sample_ground(n, wo, p, rp.ground_roughness, rng);
                if (!bs.valid || bs.pdf <= 1e-8f) break;

                float NoL = fmaxf(dot(n, bs.wi), 0.0f);
                throughput = mul(throughput, bs.f * (NoL / bs.pdf));

                prev_delta = false;
                prev_bsdf_pdf = bs.pdf;
                prev_surface_p = p;

                ray.o = p + bs.wi * 1e-4f;
                ray.d = bs.wi;
            } else {
                break;
            }

            if (bounce >= 3) {
                float pcont = fminf(fmaxf(max_comp(throughput), 0.05f), 0.98f);
                if (rng.next_f() > pcont) break;
                throughput = throughput / pcont;
            }
        }

        float rmax = max_comp(radiance);
        if (rmax > 12.0f) {
            radiance = radiance * (12.0f / rmax);
        }
        sum += radiance;
    }

    accum[idx] += sum;
    accum_n[idx] += rp.spp_per_frame;
    rng_states[idx] = rng.state;
}

__device__ inline uint8_t to_byte(float v) {
    return (uint8_t)lrintf(fminf(fmaxf(v, 0.0f), 1.0f) * 255.0f);
}

__device__ inline Vec3 avg_hdr(const Vec3* accum, const int* accum_n, int idx) {
    int n = max(accum_n[idx], 1);
    return accum[idx] / (float)n;
}

__device__ inline float luminance(const Vec3& c) {
    return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
}

__device__ inline Vec3 tonemap_gamma(const Vec3& c) {
    Vec3 t = Vec3(c.x / (1.0f + c.x), c.y / (1.0f + c.y), c.z / (1.0f + c.z));
    t = Vec3(powf(t.x, 1.0f / 2.2f), powf(t.y, 1.0f / 2.2f), powf(t.z, 1.0f / 2.2f));
    return clamp01(t);
}

__global__ void tonemap_ui_kernel(const Vec3* accum,
                                  const int* accum_n,
                                  uchar4* out_rgba,
                                  RenderParams rp,
                                  UIParams ui,
                                  int selected_preset,
                                  int draw_ui)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= rp.width || y >= rp.height) return;

    int src_idx = y * rp.width + x;

    Vec3 center = avg_hdr(accum, accum_n, src_idx);
    float center_lum = fmaxf(luminance(center), 1e-4f);
    float spp = (float)max(accum_n[src_idx], 1);
    float sigma_s = 1.5f;
    float sigma_c = 0.30f + 0.90f / sqrtf(spp + 1.0f);
    float inv2_sigma_s2 = 1.0f / (2.0f * sigma_s * sigma_s);
    float inv2_sigma_c2 = 1.0f / (2.0f * sigma_c * sigma_c);

    Vec3 filt(0, 0, 0);
    float wsum = 0.0f;

    for (int dy = -2; dy <= 2; dy++) {
        int yy = y + dy;
        if (yy < 0 || yy >= rp.height) continue;
        for (int dx = -2; dx <= 2; dx++) {
            int xx = x + dx;
            if (xx < 0 || xx >= rp.width) continue;

            int ni = yy * rp.width + xx;
            Vec3 cn = avg_hdr(accum, accum_n, ni);

            float ds2 = (float)(dx * dx + dy * dy);
            float ws = expf(-ds2 * inv2_sigma_s2);

            Vec3 diff = cn - center;
            float rel_scale = 0.15f + center_lum;
            float dc = sqrtf(dot(diff, diff)) / rel_scale;
            float wc = expf(-(dc * dc) * inv2_sigma_c2);

            float w = ws * wc;
            filt += cn * w;
            wsum += w;
        }
    }

    Vec3 c = (wsum > 0.0f) ? (filt / wsum) : center;
    c = tonemap_gamma(c);

    if (draw_ui && x >= rp.width - ui.sidebar_w) {
        c = Vec3(0.09f, 0.10f, 0.12f);

        for (int i = 0; i < kPresetCount; i++) {
            int x0, y0, x1, y1;
            swatch_rect(ui, rp.width, i, x0, y0, x1, y1);

            bool inside = in_rect(x, y, x0, y0, x1, y1);
            bool border = in_rect(x, y, x0 - 1, y0 - 1, x1 + 1, y0) ||
                          in_rect(x, y, x0 - 1, y1, x1 + 1, y1 + 1) ||
                          in_rect(x, y, x0 - 1, y0 - 1, x0, y1 + 1) ||
                          in_rect(x, y, x1, y0 - 1, x1 + 1, y1 + 1);

            if (inside) {
                float3 ps = c_preset_ui_color[i];
                Vec3 dc = preset_display_color(Vec3(ps.x, ps.y, ps.z));
                c = dc;
            }
            if (border) {
                c = Vec3(0.88f, 0.88f, 0.90f);
            }

            if (i == selected_preset) {
                bool sel = in_rect(x, y, x0 - 3, y0 - 3, x1 + 3, y0 - 2) ||
                           in_rect(x, y, x0 - 3, y1 + 2, x1 + 3, y1 + 3) ||
                           in_rect(x, y, x0 - 3, y0 - 3, x0 - 2, y1 + 3) ||
                           in_rect(x, y, x1 + 2, y0 - 3, x1 + 3, y1 + 3);
                if (sel) c = Vec3(1.0f, 0.78f, 0.12f);
            }
        }
    }

    int dst_idx = (rp.height - 1 - y) * rp.width + x;
    out_rgba[dst_idx] = make_uchar4(to_byte(c.x), to_byte(c.y), to_byte(c.z), 255);
}

struct OBJMesh {
    std::vector<Triangle> tris;
    Vec3 bmin;
    Vec3 bmax;
};

struct FaceIndex {
    int v;
    int vt;
    int vn;
};

struct TempTri {
    int v[3];
    int n[3];
};

static inline int fix_obj_index(int idx, int n) {
    if (idx > 0) return idx - 1;
    if (idx < 0) return n + idx;
    return -1;
}

static inline bool parse_face_token(const std::string& tok, FaceIndex& out) {
    out.v = out.vt = out.vn = 0;

    size_t p1 = tok.find('/');
    if (p1 == std::string::npos) {
        out.v = std::stoi(tok);
        return true;
    }

    size_t p2 = tok.find('/', p1 + 1);

    std::string s0 = tok.substr(0, p1);
    std::string s1 = (p2 == std::string::npos) ? tok.substr(p1 + 1) : tok.substr(p1 + 1, p2 - p1 - 1);
    std::string s2 = (p2 == std::string::npos) ? "" : tok.substr(p2 + 1);

    if (!s0.empty()) out.v = std::stoi(s0);
    if (!s1.empty()) out.vt = std::stoi(s1);
    if (!s2.empty()) out.vn = std::stoi(s2);

    return true;
}

static OBJMesh load_obj(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        fprintf(stderr, "Cannot open OBJ: %s\n", path.c_str());
        std::exit(1);
    }

    std::vector<Vec3> pos;
    std::vector<Vec3> nrm;
    std::vector<Vec3> tex;
    std::vector<TempTri> temp_tris;

    std::string line;
    int line_no = 0;
    while (std::getline(in, line)) {
        line_no++;
        if (line.empty() || line[0] == '#') continue;

        std::istringstream ss(line);
        std::string tag;
        ss >> tag;

        if (tag == "v") {
            float x, y, z;
            ss >> x >> y >> z;
            pos.emplace_back(x, y, z);
        } else if (tag == "vn") {
            float x, y, z;
            ss >> x >> y >> z;
            nrm.emplace_back(x, y, z);
        } else if (tag == "vt") {
            float u = 0.0f, v = 0.0f, w = 0.0f;
            ss >> u >> v >> w;
            tex.emplace_back(u, v, w);
        } else if (tag == "f") {
            std::vector<FaceIndex> face;
            std::string tok;
            while (ss >> tok) {
                FaceIndex fi{};
                if (!parse_face_token(tok, fi)) continue;

                fi.v = fix_obj_index(fi.v, (int)pos.size());
                fi.vt = fix_obj_index(fi.vt, (int)tex.size());
                fi.vn = fix_obj_index(fi.vn, (int)nrm.size());

                if (fi.v < 0 || fi.v >= (int)pos.size()) {
                    fprintf(stderr, "[OBJ] invalid vertex index at line %d\n", line_no);
                    continue;
                }
                face.push_back(fi);
            }

            if (face.size() < 3) continue;

            for (size_t i = 1; i + 1 < face.size(); i++) {
                TempTri t{};
                t.v[0] = face[0].v;
                t.v[1] = face[i].v;
                t.v[2] = face[i + 1].v;

                t.n[0] = face[0].vn;
                t.n[1] = face[i].vn;
                t.n[2] = face[i + 1].vn;
                temp_tris.push_back(t);
            }
        }
    }

    if (temp_tris.empty()) {
        fprintf(stderr, "OBJ parse produced 0 triangles.\n");
        std::exit(1);
    }

    std::vector<Vec3> smooth(pos.size(), Vec3(0, 0, 0));
    for (const TempTri& t : temp_tris) {
        Vec3 p0 = pos[t.v[0]];
        Vec3 p1 = pos[t.v[1]];
        Vec3 p2 = pos[t.v[2]];
        Vec3 fn = normalize(cross(p1 - p0, p2 - p0));
        if (length(fn) > 1e-8f) {
            smooth[t.v[0]] += fn;
            smooth[t.v[1]] += fn;
            smooth[t.v[2]] += fn;
        }
    }
    for (Vec3& n : smooth) {
        n = normalize(n);
        if (length(n) < 1e-8f) n = Vec3(0, 1, 0);
    }

    OBJMesh mesh;
    mesh.tris.reserve(temp_tris.size());

    for (const TempTri& t : temp_tris) {
        Triangle tri{};
        tri.v0 = pos[t.v[0]];
        tri.v1 = pos[t.v[1]];
        tri.v2 = pos[t.v[2]];

        Vec3 fn = normalize(cross(tri.v1 - tri.v0, tri.v2 - tri.v0));
        if (length(fn) < 1e-8f) fn = Vec3(0, 1, 0);

        tri.n0 = (t.n[0] >= 0 && t.n[0] < (int)nrm.size()) ? normalize(nrm[t.n[0]]) : smooth[t.v[0]];
        tri.n1 = (t.n[1] >= 0 && t.n[1] < (int)nrm.size()) ? normalize(nrm[t.n[1]]) : smooth[t.v[1]];
        tri.n2 = (t.n[2] >= 0 && t.n[2] < (int)nrm.size()) ? normalize(nrm[t.n[2]]) : smooth[t.v[2]];

        if (length(tri.n0) < 1e-8f) tri.n0 = fn;
        if (length(tri.n1) < 1e-8f) tri.n1 = fn;
        if (length(tri.n2) < 1e-8f) tri.n2 = fn;

        mesh.tris.push_back(tri);
    }

    Vec3 bmin(1e30f, 1e30f, 1e30f);
    Vec3 bmax(-1e30f, -1e30f, -1e30f);
    for (const Triangle& t : mesh.tris) {
        const Vec3 vv[3] = {t.v0, t.v1, t.v2};
        for (int i = 0; i < 3; i++) {
            const Vec3& v = vv[i];
            bmin = Vec3(fminf(bmin.x, v.x), fminf(bmin.y, v.y), fminf(bmin.z, v.z));
            bmax = Vec3(fmaxf(bmax.x, v.x), fmaxf(bmax.y, v.y), fmaxf(bmax.z, v.z));
        }
    }

    Vec3 centerXZ((bmin.x + bmax.x) * 0.5f, bmin.y, (bmin.z + bmax.z) * 0.5f);
    Vec3 size = bmax - bmin;
    float max_dim = fmaxf(size.x, fmaxf(size.y, size.z));
    float scale = 1.7f / fmaxf(max_dim, 1e-8f);

    for (Triangle& t : mesh.tris) {
        t.v0 = (t.v0 - centerXZ) * scale;
        t.v1 = (t.v1 - centerXZ) * scale;
        t.v2 = (t.v2 - centerXZ) * scale;
    }

    bmin = Vec3(1e30f, 1e30f, 1e30f);
    bmax = Vec3(-1e30f, -1e30f, -1e30f);
    for (const Triangle& t : mesh.tris) {
        const Vec3 vv[3] = {t.v0, t.v1, t.v2};
        for (int i = 0; i < 3; i++) {
            const Vec3& v = vv[i];
            bmin = Vec3(fminf(bmin.x, v.x), fminf(bmin.y, v.y), fminf(bmin.z, v.z));
            bmax = Vec3(fmaxf(bmax.x, v.x), fmaxf(bmax.y, v.y), fmaxf(bmax.z, v.z));
        }
    }

    mesh.bmin = bmin;
    mesh.bmax = bmax;

    fprintf(stderr, "[OBJ] vertices=%zu normals=%zu texcoords=%zu tris=%zu\n",
            pos.size(), nrm.size(), tex.size(), mesh.tris.size());
    return mesh;
}

static Camera make_fps_camera(const Vec3& pos, float yaw, float pitch) {
    float cy = cosf(yaw), sy = sinf(yaw);
    float cp = cosf(pitch), sp = sinf(pitch);
    Vec3 fwd = normalize(Vec3(cp * sy, sp, cp * cy));
    Vec3 worldUp(0, 1, 0);
    Vec3 right = normalize(cross(fwd, worldUp));
    Vec3 up = normalize(cross(right, fwd));

    Camera c{};
    c.pos = pos;
    c.forward = fwd;
    c.right = right;
    c.up = up;
    c.fov_y = 42.0f * (float)M_PI / 180.0f;
    return c;
}

static void draw_textured_fullscreen(GLuint tex) {
    glDisable(GL_DEPTH_TEST);
    glClear(GL_COLOR_BUFFER_BIT);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, tex);

    glBegin(GL_QUADS);
    glTexCoord2f(0.0f, 0.0f); glVertex2f(-1.0f, -1.0f);
    glTexCoord2f(1.0f, 0.0f); glVertex2f( 1.0f, -1.0f);
    glTexCoord2f(1.0f, 1.0f); glVertex2f( 1.0f,  1.0f);
    glTexCoord2f(0.0f, 1.0f); glVertex2f(-1.0f,  1.0f);
    glEnd();

    glDisable(GL_TEXTURE_2D);
}

static bool pick_preset(int mx, int my, int W, const UIParams& ui, int& out_index) {
    if (mx < W - ui.sidebar_w) return false;
    for (int i = 0; i < kPresetCount; i++) {
        int x0, y0, x1, y1;
        swatch_rect(ui, W, i, x0, y0, x1, y1);
        if (in_rect(mx, my, x0, y0, x1, y1)) {
            out_index = i;
            return true;
        }
    }
    return false;
}

struct InputState {
    float scroll_delta = 0.0f;
};

static void on_scroll(GLFWwindow* win, double xoff, double yoff) {
    (void)xoff;
    InputState* input = (InputState*)glfwGetWindowUserPointer(win);
    if (input) {
        input->scroll_delta += (float)yoff;
    }
}

int main(int argc, char** argv) {
    std::string obj_path = (argc >= 2) ? argv[1] : "stanford-bunny.obj";

    OBJMesh mesh = load_obj(obj_path);
    if (mesh.tris.empty()) {
        fprintf(stderr, "OBJ has no triangles.\n");
        return 1;
    }

    std::vector<int> tri_idx(mesh.tris.size());
    for (size_t i = 0; i < tri_idx.size(); i++) tri_idx[i] = (int)i;

    std::vector<BVHNode> nodes;
    nodes.reserve(mesh.tris.size() * 2);
    build_bvh_recursive(nodes, tri_idx, mesh.tris, 0, (int)mesh.tris.size());
    fprintf(stderr, "[BVH] nodes=%zu\n", nodes.size());

    const int W = 1280;
    const int H = 720;

    UIParams ui{};
    ui.sidebar_w = 180;
    ui.pad = 16;
    ui.swatch_h = 34;
    ui.gap = 12;
    ui.top = 24;

    if (!glfwInit()) {
        fprintf(stderr, "glfwInit failed\n");
        return 1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

    GLFWwindow* win = glfwCreateWindow(W, H, "CUDA Wood Bunny (GLFW)", nullptr, nullptr);
    if (!win) {
        fprintf(stderr, "glfwCreateWindow failed\n");
        glfwTerminate();
        return 1;
    }
    glfwMakeContextCurrent(win);
    glfwSwapInterval(1);

    glewExperimental = GL_TRUE;
    GLenum glew_err = glewInit();
    if (glew_err != GLEW_OK) {
        fprintf(stderr, "glewInit failed: %s\n", glewGetErrorString(glew_err));
        glfwDestroyWindow(win);
        glfwTerminate();
        return 1;
    }

    InputState input_state{};
    glfwSetWindowUserPointer(win, &input_state);
    glfwSetScrollCallback(win, on_scroll);

    GLuint tex = 0;
    GLuint pbo = 0;

    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, W, H, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, (size_t)W * H * sizeof(uchar4), nullptr, GL_STREAM_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    CUDA_CHECK(cudaSetDevice(0));

    cudaGraphicsResource* cuda_pbo = nullptr;
    bool use_gl_interop = false;
    cudaError_t reg_err = cudaGraphicsGLRegisterBuffer(&cuda_pbo, pbo, cudaGraphicsRegisterFlagsWriteDiscard);
    if (reg_err == cudaSuccess) {
        use_gl_interop = true;
        fprintf(stderr, "[Display] CUDA-OpenGL interop enabled.\n");
    } else {
        fprintf(stderr,
                "[Display] CUDA-OpenGL interop unavailable (%s). Falling back to CUDA->CPU->OpenGL upload.\n",
                cudaGetErrorString(reg_err));
        (void)cudaGetLastError();
    }

    Vec3* d_accum = nullptr;
    int* d_accum_n = nullptr;
    uint32_t* d_rng = nullptr;
    uchar4* d_display = nullptr;
    uchar4* d_save = nullptr;

    CUDA_CHECK(cudaMalloc(&d_accum, (size_t)W * H * sizeof(Vec3)));
    CUDA_CHECK(cudaMalloc(&d_accum_n, (size_t)W * H * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_rng, (size_t)W * H * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_display, (size_t)W * H * sizeof(uchar4)));
    CUDA_CHECK(cudaMalloc(&d_save, (size_t)W * H * sizeof(uchar4)));

    const size_t display_bytes = (size_t)W * H * sizeof(uchar4);
    std::vector<uchar4> h_display((size_t)W * H);

    CUDA_CHECK(cudaMemset(d_accum, 0, (size_t)W * H * sizeof(Vec3)));
    CUDA_CHECK(cudaMemset(d_accum_n, 0, (size_t)W * H * sizeof(int)));

    {
        std::vector<uint32_t> seeds((size_t)W * H);
        for (int i = 0; i < W * H; i++) {
            seeds[i] = (uint32_t)(1337u + i * 9781u + 123u);
        }
        CUDA_CHECK(cudaMemcpy(d_rng, seeds.data(), (size_t)W * H * sizeof(uint32_t), cudaMemcpyHostToDevice));
    }

    Triangle* d_tris = nullptr;
    int* d_tri_idx = nullptr;
    BVHNode* d_nodes = nullptr;

    CUDA_CHECK(cudaMalloc(&d_tris, mesh.tris.size() * sizeof(Triangle)));
    CUDA_CHECK(cudaMalloc(&d_tri_idx, tri_idx.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_nodes, nodes.size() * sizeof(BVHNode)));

    CUDA_CHECK(cudaMemcpy(d_tris, mesh.tris.data(), mesh.tris.size() * sizeof(Triangle), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tri_idx, tri_idx.data(), tri_idx.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_nodes, nodes.data(), nodes.size() * sizeof(BVHNode), cudaMemcpyHostToDevice));

    DeviceScene dsc{};
    dsc.tris = d_tris;
    dsc.tri_idx = d_tri_idx;
    dsc.nodes = d_nodes;
    dsc.tri_count = (int)mesh.tris.size();
    dsc.node_count = (int)nodes.size();

    {
        float3 ui_colors[kPresetCount];
        for (int i = 0; i < kPresetCount; i++) {
            Vec3 c = kPresets[i].light_color * 0.62f + kPresets[i].dark_color * 0.38f;
            ui_colors[i] = make_float3(c.x, c.y, c.z);
        }
        CUDA_CHECK(cudaMemcpyToSymbol(c_preset_ui_color, ui_colors, sizeof(ui_colors)));
    }

    RenderParams rp{};
    rp.width = W;
    rp.height = H;
    rp.frame_index = 0;
    rp.spp_per_frame = 2;
    rp.max_bounces = 12;

    int selected_preset = 0;
    rp.ground_roughness = 0.11f;

    rp.light_center = Vec3(0.10f, 2.25f, 0.10f);
    rp.light_u = Vec3(0.78f, 0.0f, 0.0f);
    rp.light_v = Vec3(0.0f, 0.0f, 0.56f);
    rp.light_n = normalize(cross(rp.light_u, rp.light_v));
    rp.light_area = 4.0f * length(cross(rp.light_u, rp.light_v));
    rp.light_emission = Vec3(23.0f, 22.0f, 20.0f);

    auto apply_wood_preset = [&](int idx) {
        const WoodPreset& wp = kPresets[idx];
        rp.wood_light = wp.light_color;
        rp.wood_dark = wp.dark_color;
        rp.bunny_roughness = wp.roughness;
        rp.bunny_spec_strength = wp.spec_strength;
        rp.wood_grain_scale = wp.grain_scale;
        rp.wood_ring_scale = wp.ring_scale;
        rp.wood_ring_strength = wp.ring_strength;
        rp.wood_pore_strength = wp.pore_strength;
    };
    apply_wood_preset(selected_preset);

    Vec3 bbox_size = mesh.bmax - mesh.bmin;
    float radius = 0.5f * length(bbox_size);

    float eye_height = fmaxf(0.50f, mesh.bmax.y * 0.58f);
    float walk_limit = fmaxf(radius * 3.4f, 2.6f);
    float move_speed = 2.1f;
    float scroll_step = 0.35f;

    Vec3 cam_pos(0.0f, eye_height, fmaxf(radius * 2.45f, 2.0f));
    float yaw = (float)M_PI;
    float pitch = -0.08f;

    bool dragging = false;
    double last_mx = 0.0, last_my = 0.0;
    bool prev_left = false;

    bool prev_r = false;
    bool prev_p = false;

    bool save_png = false;

    auto reset_accum = [&]() {
        CUDA_CHECK(cudaMemset(d_accum, 0, (size_t)W * H * sizeof(Vec3)));
        CUDA_CHECK(cudaMemset(d_accum_n, 0, (size_t)W * H * sizeof(int)));
        rp.frame_index = 0;
    };

    double prev_time = glfwGetTime();
    fprintf(stderr, "[Controls] LMB drag look | WASD move | wheel forward/back | click swatch | R reset | P save | Q/ESC quit\n");

    while (!glfwWindowShouldClose(win)) {
        glfwPollEvents();

        if (glfwGetKey(win, GLFW_KEY_ESCAPE) == GLFW_PRESS ||
            glfwGetKey(win, GLFW_KEY_Q) == GLFW_PRESS)
        {
            break;
        }

        double now_time = glfwGetTime();
        float dt = (float)(now_time - prev_time);
        prev_time = now_time;
        dt = fmaxf(0.0f, fminf(dt, 0.05f));

        bool camera_changed = false;

        Camera cam_move = make_fps_camera(cam_pos, yaw, pitch);
        Vec3 forward_xz(cam_move.forward.x, 0.0f, cam_move.forward.z);
        if (length(forward_xz) > 1e-6f) forward_xz = normalize(forward_xz);
        else forward_xz = Vec3(0, 0, -1);
        Vec3 right_xz(cam_move.right.x, 0.0f, cam_move.right.z);
        if (length(right_xz) > 1e-6f) right_xz = normalize(right_xz);
        else right_xz = Vec3(1, 0, 0);

        float speed_mul = 1.0f;
        if (glfwGetKey(win, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ||
            glfwGetKey(win, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS) {
            speed_mul = 2.0f;
        }

        Vec3 move_dir(0, 0, 0);
        if (glfwGetKey(win, GLFW_KEY_W) == GLFW_PRESS) move_dir += forward_xz;
        if (glfwGetKey(win, GLFW_KEY_S) == GLFW_PRESS) move_dir = move_dir - forward_xz;
        if (glfwGetKey(win, GLFW_KEY_D) == GLFW_PRESS) move_dir += right_xz;
        if (glfwGetKey(win, GLFW_KEY_A) == GLFW_PRESS) move_dir = move_dir - right_xz;

        if (length(move_dir) > 1e-6f) {
            move_dir = normalize(move_dir);
            cam_pos += move_dir * (move_speed * speed_mul * dt);
            camera_changed = true;
        }

        if (fabsf(input_state.scroll_delta) > 0.0f) {
            cam_pos += forward_xz * (input_state.scroll_delta * scroll_step);
            input_state.scroll_delta = 0.0f;
            camera_changed = true;
        }

        cam_pos.x = fmaxf(-walk_limit, fminf(walk_limit, cam_pos.x));
        cam_pos.z = fmaxf(-walk_limit, fminf(walk_limit, cam_pos.z));
        cam_pos.y = eye_height;

        bool now_r = glfwGetKey(win, GLFW_KEY_R) == GLFW_PRESS;
        bool now_p = glfwGetKey(win, GLFW_KEY_P) == GLFW_PRESS;

        if (now_r && !prev_r) {
            reset_accum();
        }
        if (now_p && !prev_p) {
            save_png = true;
        }
        prev_r = now_r;
        prev_p = now_p;

        double mx = 0.0, my = 0.0;
        glfwGetCursorPos(win, &mx, &my);

        int win_w = 0, win_h = 0, fb_w = 0, fb_h = 0;
        glfwGetWindowSize(win, &win_w, &win_h);
        glfwGetFramebufferSize(win, &fb_w, &fb_h);

        float sx = (win_w > 0) ? (fb_w / (float)win_w) : 1.0f;
        float sy = (win_h > 0) ? (fb_h / (float)win_h) : 1.0f;
        int mxi = (int)floor(mx * sx);
        int myi = (int)floor(my * sy);

        bool now_left = glfwGetMouseButton(win, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
        if (now_left && !prev_left) {
            int picked = -1;
            if (pick_preset(mxi, myi, W, ui, picked)) {
                selected_preset = picked;
                apply_wood_preset(selected_preset);
                fprintf(stderr, "[Wood] %s\n", kPresets[selected_preset].name);
                reset_accum();
            } else {
                dragging = true;
                last_mx = mx;
                last_my = my;
            }
        }
        if (!now_left && prev_left) {
            dragging = false;
        }

        if (dragging && now_left) {
            double dx = mx - last_mx;
            double dy = my - last_my;
            last_mx = mx;
            last_my = my;

            if (dx != 0.0 || dy != 0.0) {
                yaw += (float)(dx * 0.0055);
                pitch += (float)(-dy * 0.0055);
                pitch = std::max(-1.35f, std::min(1.35f, pitch));
                camera_changed = true;
            }
        }
        prev_left = now_left;

        if (camera_changed) {
            reset_accum();
        }

        Camera cam = make_fps_camera(cam_pos, yaw, pitch);

        dim3 block(16, 16);
        dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y);
        rp.spp_per_frame = camera_changed ? 1 : 2;

        render_kernel<<<grid, block>>>(d_accum, d_accum_n, d_rng, dsc, cam, rp);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        uchar4* d_pixels = d_display;
        if (use_gl_interop) {
            CUDA_CHECK(cudaGraphicsMapResources(1, &cuda_pbo, 0));
            size_t pbo_bytes = 0;
            CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void**)&d_pixels, &pbo_bytes, cuda_pbo));
            if (pbo_bytes < display_bytes) {
                fprintf(stderr, "Mapped PBO too small (%zu < %zu)\n", pbo_bytes, display_bytes);
                break;
            }
        }

        tonemap_ui_kernel<<<grid, block>>>(d_accum, d_accum_n, d_pixels, rp, ui, selected_preset, 1);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        if (save_png) {
            tonemap_ui_kernel<<<grid, block>>>(d_accum, d_accum_n, d_save, rp, ui, selected_preset, 0);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            std::vector<uint8_t> raw((size_t)W * H * 4);
            CUDA_CHECK(cudaMemcpy(raw.data(), d_save, raw.size(), cudaMemcpyDeviceToHost));

            int save_w = W - ui.sidebar_w;
            std::vector<uint8_t> png((size_t)save_w * H * 4);
            for (int yy = 0; yy < H; yy++) {
                const uint8_t* src = &raw[(size_t)(H - 1 - yy) * W * 4];
                std::memcpy(&png[(size_t)yy * save_w * 4], src, (size_t)save_w * 4);
            }
            stbi_write_png("out.png", save_w, H, 4, png.data(), save_w * 4);
            fprintf(stderr, "[Saved] out.png\n");
            save_png = false;
        }

        glViewport(0, 0, W, H);
        glBindTexture(GL_TEXTURE_2D, tex);
        if (use_gl_interop) {
            CUDA_CHECK(cudaGraphicsUnmapResources(1, &cuda_pbo, 0));
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, W, H, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
        } else {
            CUDA_CHECK(cudaMemcpy(h_display.data(), d_pixels, display_bytes, cudaMemcpyDeviceToHost));
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, W, H, GL_RGBA, GL_UNSIGNED_BYTE, h_display.data());
        }

        draw_textured_fullscreen(tex);
        glfwSwapBuffers(win);

        rp.frame_index++;
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    if (use_gl_interop) {
        CUDA_CHECK(cudaGraphicsUnregisterResource(cuda_pbo));
    }

    glDeleteBuffers(1, &pbo);
    glDeleteTextures(1, &tex);

    glfwDestroyWindow(win);
    glfwTerminate();

    CUDA_CHECK(cudaFree(d_accum));
    CUDA_CHECK(cudaFree(d_accum_n));
    CUDA_CHECK(cudaFree(d_rng));
    CUDA_CHECK(cudaFree(d_display));
    CUDA_CHECK(cudaFree(d_save));

    CUDA_CHECK(cudaFree(d_tris));
    CUDA_CHECK(cudaFree(d_tri_idx));
    CUDA_CHECK(cudaFree(d_nodes));

    return 0;
}
