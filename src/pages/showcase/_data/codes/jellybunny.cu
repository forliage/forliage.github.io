// Build:
//   nvcc -O3 -std=c++17 -arch=sm_89 jelly.cu -o bunny_jelly
// Run:
//   ./bunny_jelly stanford-bunny.obj
//
// Terminal prompts:
//   - jelly color
//   - drop height / impact boost
//   - view yaw/pitch
// Then export drop-impact wobble GIF offline.

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <cctype>

#include <algorithm>
#include <array>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_set>
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
    // Keep BVH valid while jelly vertices wobble around rest pose.
    const float eps = 0.70f;
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
    int fast_tracing;
    int preview_mode;

    float ior;
    float glass_roughness;
    Vec3 sigma_a;
    Vec3 sigma_s;
    float medium_aniso_g;

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

struct JellyPreset {
    const char* name;
    Vec3 sigma_a;
    Vec3 sigma_s;
    float roughness;
    float ior;
    Vec3 ui_color;
};

static constexpr int kPresetCount = 8;
static const JellyPreset kPresets[kPresetCount] = {
    {"Clear",      Vec3(0.03f, 0.03f, 0.03f), Vec3(0.010f, 0.010f, 0.010f), 0.040f, 1.35f, Vec3(0.85f, 0.88f, 0.92f)},
    {"Strawberry", Vec3(0.10f, 1.20f, 1.20f), Vec3(0.035f, 0.008f, 0.008f), 0.045f, 1.36f, Vec3(0.90f, 0.25f, 0.30f)},
    {"Orange",     Vec3(0.08f, 0.35f, 1.10f), Vec3(0.030f, 0.018f, 0.007f), 0.042f, 1.36f, Vec3(0.95f, 0.52f, 0.22f)},
    {"Lemon",      Vec3(0.20f, 0.18f, 1.05f), Vec3(0.034f, 0.032f, 0.008f), 0.050f, 1.35f, Vec3(0.95f, 0.88f, 0.24f)},
    {"Lime",       Vec3(1.00f, 0.14f, 0.95f), Vec3(0.010f, 0.035f, 0.010f), 0.050f, 1.34f, Vec3(0.45f, 0.90f, 0.35f)},
    {"Blueberry",  Vec3(1.20f, 0.80f, 0.10f), Vec3(0.008f, 0.012f, 0.034f), 0.046f, 1.36f, Vec3(0.30f, 0.45f, 0.92f)},
    {"Grape",      Vec3(0.55f, 1.10f, 0.22f), Vec3(0.022f, 0.010f, 0.026f), 0.044f, 1.36f, Vec3(0.62f, 0.35f, 0.84f)},
    {"Peach",      Vec3(0.08f, 0.35f, 0.45f), Vec3(0.026f, 0.018f, 0.015f), 0.048f, 1.35f, Vec3(0.95f, 0.70f, 0.60f)},
};

__constant__ float3 c_preset_ui[kPresetCount];

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

__host__ __device__ inline void force_button_rect(const UIParams& ui, int W, int H,
                                                  int& x0, int& y0, int& x1, int& y1)
{
    x0 = W - ui.sidebar_w + ui.pad;
    x1 = W - ui.pad;
    y1 = H - ui.pad - 4;
    y0 = y1 - 42;
}

__host__ __device__ inline void direction_button_rect(const UIParams& ui, int W, int H,
                                                      int& x0, int& y0, int& x1, int& y1)
{
    x0 = W - ui.sidebar_w + ui.pad;
    x1 = W - ui.pad;
    y1 = H - ui.pad - 58;
    y0 = y1 - 30;
}

__host__ __device__ inline void height_slider_rect(const UIParams& ui, int W, int H,
                                                   int& x0, int& y0, int& x1, int& y1)
{
    x0 = W - ui.sidebar_w + ui.pad;
    x1 = W - ui.pad;
    y1 = H - ui.pad - 102;
    y0 = y1 - 24;
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

__device__ inline BSDFSample sample_rough_glass(const Vec3& n,
                                                const Vec3& ray_dir,
                                                float ior,
                                                float roughness,
                                                RNG& rng,
                                                bool inside_glass)
{
    BSDFSample s{};
    s.valid = true;
    s.toggled_inside = false;

    Vec3 ns = (dot(ray_dir, n) < 0.0f) ? n : -n; // face-forward normal against incoming ray

    bool entering = dot(ray_dir, n) < 0.0f;
    float etaI = entering ? 1.0f : ior;
    float etaT = entering ? ior : 1.0f;
    float eta = etaI / etaT;

    float cosi = fmaxf(dot(-ray_dir, ns), 0.0f);
    float Fr = schlick(cosi, etaI, etaT);

    Vec3 wr = reflect_dir(ray_dir, ns);

    Vec3 wt;
    bool tir = !refract_eta(ray_dir, ns, eta, wt);

    bool refl = tir || (rng.next_f() < Fr);

    float expn = roughness_to_exp(roughness);
    bool non_delta = roughness > 0.015f;

    Vec3 axis = refl ? wr : wt;
    Vec3 wi = axis;
    float lobe_pdf = 1.0f;

    if (non_delta) {
        bool want_same_side = refl;
        bool ok = false;
        for (int i = 0; i < 6; i++) {
            Vec3 cand = sample_power_cosine_axis(axis, expn, rng);
            bool same_side = dot(cand, ns) > 0.0f;
            if (same_side == want_same_side) {
                wi = cand;
                ok = true;
                break;
            }
        }
        if (!ok) wi = axis;
        lobe_pdf = fmaxf(power_cosine_pdf(axis, wi, expn), 1e-8f);
    }

    float p_refl = tir ? 1.0f : Fr;
    float p_tran = tir ? 0.0f : (1.0f - Fr);

    if (refl) {
        s.pdf = non_delta ? fmaxf(p_refl * lobe_pdf, 1e-8f) : 1.0f;
        s.weight = Vec3(1, 1, 1);
        s.toggled_inside = false;
    } else {
        s.pdf = non_delta ? fmaxf(p_tran * lobe_pdf, 1e-8f) : 1.0f;
        s.weight = Vec3(1, 1, 1);
        s.toggled_inside = true;
    }

    s.wi = wi;
    s.f = Vec3(0, 0, 0);
    s.delta = !non_delta;

    (void)inside_glass;
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

        bool inside_glass = false;

        bool prev_delta = true;
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

            // Participating medium inside jelly (absorption + weak scattering).
            if (inside_glass) {
                Vec3 sigma_t = rp.sigma_a + rp.sigma_s;
                Vec3 tr = exp3(Vec3(-sigma_t.x * hit.t, -sigma_t.y * hit.t, -sigma_t.z * hit.t));
                if (!rp.fast_tracing) {
                    Vec3 albedo_s(
                        rp.sigma_s.x / fmaxf(sigma_t.x, 1e-6f),
                        rp.sigma_s.y / fmaxf(sigma_t.y, 1e-6f),
                        rp.sigma_s.z / fmaxf(sigma_t.z, 1e-6f)
                    );
                    float scatter_amount = 1.0f - (tr.x + tr.y + tr.z) * (1.0f / 3.0f);
                    Vec3 approx_env = environment_radiance(-ray.d);
                    float g = fminf(fmaxf(rp.medium_aniso_g, -0.95f), 0.95f);
                    float phase_boost = 0.35f + 0.65f * (1.0f - g * g);
                    Vec3 in_scatter = (0.30f * approx_env + 0.04f * rp.light_emission) * (scatter_amount * phase_boost);
                    radiance += mul(throughput, mul(albedo_s, in_scatter));
                }
                throughput = mul(throughput, tr);
            }

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

            if (hit.type == HIT_GROUND) {
                // Next-event estimation on ground with MIS (skip in fast mode).
                if (!rp.fast_tracing) {
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
                            GroundEval g = eval_ground_bsdf(n, wo, wi, p, rp.ground_roughness);
                            if (g.pdf > 1e-8f && pdf_light > 1e-8f) {
                                float w = power_heuristic(pdf_light, g.pdf);
                                Vec3 contrib = mul(throughput, g.f);
                                contrib = contrib * (NoL * w / pdf_light);
                                contrib = mul(contrib, rp.light_emission);
                                radiance += contrib;
                            }
                        }
                    }
                    (void)pdfA;
                }

                BSDFSample bs = sample_ground(n, wo, p, rp.ground_roughness, rng);
                if (!bs.valid || bs.pdf <= 1e-8f) break;

                float NoL = fmaxf(dot(n, bs.wi), 0.0f);
                throughput = mul(throughput, bs.f * (NoL / bs.pdf));

                prev_delta = false;
                prev_bsdf_pdf = bs.pdf;
                prev_surface_p = p;

                ray.o = p + bs.wi * 1e-4f;
                ray.d = bs.wi;
            } else {
                BSDFSample gs = sample_rough_glass(n, ray.d, rp.ior, rp.glass_roughness, rng, inside_glass);
                if (!gs.valid) break;

                throughput = mul(throughput, gs.weight);

                if (gs.toggled_inside) {
                    inside_glass = !inside_glass;
                }

                prev_delta = gs.delta;
                prev_bsdf_pdf = gs.pdf;
                prev_surface_p = p;

                ray.o = p + gs.wi * 1e-4f;
                ray.d = gs.wi;
            }

            int rr_start = rp.fast_tracing ? 2 : 3;
            if (bounce >= rr_start) {
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

    if (rp.preview_mode) {
        accum[idx] = sum;
        accum_n[idx] = rp.spp_per_frame;
    } else {
        accum[idx] += sum;
        accum_n[idx] += rp.spp_per_frame;
    }
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
                                  int draw_ui,
                                  float force_height01,
                                  int force_sign,
                                  int export_busy,
                                  int fast_mode)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= rp.width || y >= rp.height) return;

    int src_idx = y * rp.width + x;

    Vec3 center = avg_hdr(accum, accum_n, src_idx);
    float center_lum = fmaxf(luminance(center), 1e-4f);
    float spp = (float)max(accum_n[src_idx], 1);
    float sigma_s = fast_mode ? 0.0f : 1.6f;
    float sigma_c = 0.30f + 0.90f / sqrtf(spp + 1.0f);
    float inv2_sigma_s2 = (sigma_s > 0.0f) ? (1.0f / (2.0f * sigma_s * sigma_s)) : 0.0f;
    float inv2_sigma_c2 = 1.0f / (2.0f * sigma_c * sigma_c);
    int radius = fast_mode ? 0 : 2;

    Vec3 filt(0, 0, 0);
    float wsum = 0.0f;

    for (int dy = -radius; dy <= radius; dy++) {
        int yy = y + dy;
        if (yy < 0 || yy >= rp.height) continue;
        for (int dx = -radius; dx <= radius; dx++) {
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
                float3 ps = c_preset_ui[i];
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

        int sx0, sy0, sx1, sy1;
        height_slider_rect(ui, rp.width, rp.height, sx0, sy0, sx1, sy1);
        if (in_rect(x, y, sx0, sy0, sx1, sy1)) c = Vec3(0.22f, 0.24f, 0.28f);
        int kx = sx0 + (int)lrintf(force_height01 * (float)(sx1 - sx0 - 1));
        bool knob = in_rect(x, y, kx - 2, sy0 - 3, kx + 3, sy1 + 3);
        if (knob) c = Vec3(0.95f, 0.82f, 0.22f);

        int dx0, dy0, dx1, dy1;
        direction_button_rect(ui, rp.width, rp.height, dx0, dy0, dx1, dy1);
        if (in_rect(x, y, dx0, dy0, dx1, dy1)) {
            c = (force_sign > 0) ? Vec3(0.28f, 0.45f, 0.88f) : Vec3(0.88f, 0.38f, 0.28f);
        }

        int bx0, by0, bx1, by1;
        force_button_rect(ui, rp.width, rp.height, bx0, by0, bx1, by1);
        if (in_rect(x, y, bx0, by0, bx1, by1)) {
            c = export_busy ? Vec3(0.50f, 0.22f, 0.22f) : Vec3(0.20f, 0.62f, 0.48f);
        }
    }

    int dst_idx = (rp.height - 1 - y) * rp.width + x;
    out_rgba[dst_idx] = make_uchar4(to_byte(c.x), to_byte(c.y), to_byte(c.z), 255);
}

struct Int3 { int x, y, z; };

struct OBJMesh {
    std::vector<Triangle> tris;
    std::vector<Vec3> vertices;
    std::vector<Int3> faces;
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
    for (Vec3& v : pos) {
        v = (v - centerXZ) * scale;
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
    mesh.vertices = pos;
    mesh.faces.reserve(temp_tris.size());
    for (const TempTri& t : temp_tris) {
        mesh.faces.push_back(Int3{t.v[0], t.v[1], t.v[2]});
    }

    fprintf(stderr, "[OBJ] vertices=%zu normals=%zu texcoords=%zu tris=%zu\n",
            pos.size(), nrm.size(), tex.size(), mesh.tris.size());
    return mesh;
}

static Camera make_orbit_camera(const Vec3& target, float distance, float yaw_deg, float pitch_deg) {
    float yaw = yaw_deg * ((float)M_PI / 180.0f);
    float pitch = fmaxf(fminf(pitch_deg, 80.0f), -80.0f) * ((float)M_PI / 180.0f);

    Vec3 from_target(
        cosf(pitch) * sinf(yaw),
        sinf(pitch),
        cosf(pitch) * cosf(yaw)
    );
    Vec3 pos = target + from_target * fmaxf(distance, 0.1f);
    Vec3 fwd = normalize(target - pos);

    Vec3 world_up(0, 1, 0);
    Vec3 right = normalize(cross(fwd, world_up));
    if (length(right) < 1e-6f) right = Vec3(1, 0, 0);
    Vec3 up = normalize(cross(right, fwd));

    Camera c{};
    c.pos = pos;
    c.forward = fwd;
    c.right = right;
    c.up = up;
    c.fov_y = 42.0f * (float)M_PI / 180.0f;
    return c;
}

struct EdgeConstraint {
    int i, j;
    float rest_len;
    float lambda;
};

struct SoftBodyState {
    std::vector<Vec3> rest_pos;
    std::vector<Vec3> pos;
    std::vector<Vec3> prev_pos;
    std::vector<Vec3> vel;
    std::vector<float> inv_mass;
    std::vector<EdgeConstraint> edges;
    Vec3 rest_bmin;
    Vec3 rest_bmax;
    Vec3 center_rest;

    float edge_compliance = 6.0e-5f;
    float shape_compliance = 1.8e-2f;
    float damping = 1.4f;
    float gravity = -12.8f;
    float ground_restitution = 0.11f;
    float ground_friction = 0.90f;
    bool active = false;
    int sleep_counter = 0;
};

static inline uint64_t edge_key(int a, int b) {
    if (a > b) std::swap(a, b);
    return ((uint64_t)(uint32_t)a << 32) | (uint32_t)b;
}

static SoftBodyState build_softbody(const OBJMesh& mesh) {
    SoftBodyState sb;
    sb.rest_pos = mesh.vertices;
    sb.pos = sb.rest_pos;
    sb.prev_pos = sb.rest_pos;
    sb.vel.assign(sb.rest_pos.size(), Vec3(0, 0, 0));
    sb.inv_mass.assign(sb.rest_pos.size(), 1.0f);

    sb.rest_bmin = Vec3(1e30f, 1e30f, 1e30f);
    sb.rest_bmax = Vec3(-1e30f, -1e30f, -1e30f);
    Vec3 csum(0, 0, 0);
    for (const Vec3& p : sb.rest_pos) {
        sb.rest_bmin = Vec3(fminf(sb.rest_bmin.x, p.x), fminf(sb.rest_bmin.y, p.y), fminf(sb.rest_bmin.z, p.z));
        sb.rest_bmax = Vec3(fmaxf(sb.rest_bmax.x, p.x), fmaxf(sb.rest_bmax.y, p.y), fmaxf(sb.rest_bmax.z, p.z));
        csum += p;
    }
    sb.center_rest = csum / fmaxf((float)sb.rest_pos.size(), 1.0f);

    std::unordered_set<uint64_t> dedup;
    dedup.reserve(mesh.faces.size() * 3);
    sb.edges.reserve(mesh.faces.size() * 3);
    for (const Int3& f : mesh.faces) {
        const int ids[3] = {f.x, f.y, f.z};
        for (int e = 0; e < 3; e++) {
            int a = ids[e];
            int b = ids[(e + 1) % 3];
            if (a < 0 || b < 0 || a >= (int)sb.rest_pos.size() || b >= (int)sb.rest_pos.size()) continue;
            uint64_t k = edge_key(a, b);
            if (dedup.find(k) != dedup.end()) continue;
            dedup.insert(k);
            float rl = length(sb.rest_pos[b] - sb.rest_pos[a]);
            if (rl < 1e-6f) continue;
            sb.edges.push_back(EdgeConstraint{a, b, rl, 0.0f});
        }
    }
    fprintf(stderr, "[XPBD] verts=%zu edges=%zu\n", sb.rest_pos.size(), sb.edges.size());
    return sb;
}

static float step_softbody_xpbd(SoftBodyState& sb, float dt, bool fast_mode) {
    if (dt <= 0.0f || sb.pos.empty()) return 0.0f;

    const int substeps = fast_mode ? 1 : 2;
    const int iters = fast_mode ? 2 : 3;
    float dt_sub = dt / (float)substeps;
    float dt2 = dt_sub * dt_sub;
    float damping_mul = expf(-sb.damping * dt_sub);
    float kinetic = 0.0f;

    for (int s = 0; s < substeps; s++) {
        for (EdgeConstraint& e : sb.edges) e.lambda = 0.0f;

        for (size_t i = 0; i < sb.pos.size(); i++) {
            if (sb.inv_mass[i] <= 0.0f) continue;
            sb.vel[i] = sb.vel[i] * damping_mul;
            sb.vel[i].y += sb.gravity * dt_sub;
            sb.prev_pos[i] = sb.pos[i];
            sb.pos[i] += sb.vel[i] * dt_sub;
        }

        float alpha_edge = sb.edge_compliance / fmaxf(dt2, 1e-8f);
        float alpha_shape = sb.shape_compliance / fmaxf(dt2, 1e-8f);

        for (int it = 0; it < iters; it++) {
            for (EdgeConstraint& e : sb.edges) {
                int i = e.i, j = e.j;
                float wi = sb.inv_mass[i];
                float wj = sb.inv_mass[j];
                Vec3 d = sb.pos[j] - sb.pos[i];
                float len = length(d);
                if (len < 1e-7f) continue;

                float C = len - e.rest_len;
                float denom = wi + wj + alpha_edge;
                float dl = (-C - alpha_edge * e.lambda) / fmaxf(denom, 1e-8f);
                e.lambda += dl;
                Vec3 grad = d / len;
                sb.pos[i] += grad * (-wi * dl);
                sb.pos[j] += grad * (wj * dl);
            }

            Vec3 center(0, 0, 0);
            for (const Vec3& p : sb.pos) center += p;
            center = center / fmaxf((float)sb.pos.size(), 1.0f);
            Vec3 center_off = center - sb.center_rest;

            for (size_t i = 0; i < sb.pos.size(); i++) {
                float wi = sb.inv_mass[i];
                if (wi <= 0.0f) continue;
                Vec3 target = sb.rest_pos[i] + center_off;
                Vec3 C = sb.pos[i] - target;
                float denom = wi + alpha_shape;
                Vec3 corr = C * (-wi / fmaxf(denom, 1e-8f));
                sb.pos[i] += corr;
            }
        }

        for (size_t i = 0; i < sb.pos.size(); i++) {
            if (sb.pos[i].y < 0.0f) {
                sb.pos[i].y = 0.0f;
                Vec3 v_hit = (sb.pos[i] - sb.prev_pos[i]) / fmaxf(dt_sub, 1e-8f);
                if (v_hit.y < 0.0f) v_hit.y = -v_hit.y * sb.ground_restitution;
                v_hit.x *= sb.ground_friction;
                v_hit.z *= sb.ground_friction;
                sb.prev_pos[i] = sb.pos[i] - v_hit * dt_sub;
            }
            sb.vel[i] = (sb.pos[i] - sb.prev_pos[i]) / fmaxf(dt_sub, 1e-8f);
            kinetic += dot(sb.vel[i], sb.vel[i]);
        }
    }

    kinetic /= fmaxf((float)sb.pos.size(), 1.0f);
    if (kinetic > 1.8e-5f) sb.active = true;
    if (kinetic < 8e-6f) sb.sleep_counter++;
    else sb.sleep_counter = 0;
    if (sb.sleep_counter > 36) {
        sb.active = false;
        for (Vec3& v : sb.vel) v = Vec3(0, 0, 0);
    }
    return kinetic;
}

static void setup_drop(SoftBodyState& sb, float drop_height, float impact_boost) {
    float h = fmaxf(drop_height, 0.0f);
    float v0 = -fmaxf(impact_boost, 0.0f);
    Vec3 lift(0.0f, h, 0.0f);
    for (size_t i = 0; i < sb.pos.size(); i++) {
        sb.pos[i] = sb.rest_pos[i] + lift;
        sb.prev_pos[i] = sb.pos[i];
        sb.vel[i] = Vec3(0.0f, v0, 0.0f);
    }
    sb.active = true;
    sb.sleep_counter = 0;
}

static void build_deformed_tris(const OBJMesh& mesh, const SoftBodyState& sb, std::vector<Triangle>& out) {
    static std::vector<Vec3> n_acc;
    n_acc.assign(sb.pos.size(), Vec3(0, 0, 0));
    out.resize(mesh.faces.size());

    for (size_t i = 0; i < mesh.faces.size(); i++) {
        const Int3& f = mesh.faces[i];
        if (f.x < 0 || f.y < 0 || f.z < 0 ||
            f.x >= (int)sb.pos.size() || f.y >= (int)sb.pos.size() || f.z >= (int)sb.pos.size()) {
            continue;
        }
        Vec3 p0 = sb.pos[f.x];
        Vec3 p1 = sb.pos[f.y];
        Vec3 p2 = sb.pos[f.z];
        Vec3 fn = cross(p1 - p0, p2 - p0);
        if (length(fn) > 1e-9f) {
            n_acc[f.x] += fn;
            n_acc[f.y] += fn;
            n_acc[f.z] += fn;
        }
    }
    for (Vec3& n : n_acc) {
        n = normalize(n);
        if (length(n) < 1e-7f) n = Vec3(0, 1, 0);
    }

    for (size_t i = 0; i < mesh.faces.size(); i++) {
        const Int3& f = mesh.faces[i];
        Triangle tri{};
        tri.v0 = sb.pos[f.x];
        tri.v1 = sb.pos[f.y];
        tri.v2 = sb.pos[f.z];
        tri.n0 = n_acc[f.x];
        tri.n1 = n_acc[f.y];
        tri.n2 = n_acc[f.z];
        out[i] = tri;
    }
}

int main(int argc, char** argv) {
    std::string obj_path = (argc >= 2) ? argv[1] : "stanford-bunny.obj";

    OBJMesh mesh = load_obj(obj_path);
    if (mesh.tris.empty()) {
        fprintf(stderr, "OBJ has no triangles.\n");
        return 1;
    }

    SoftBodyState soft_body = build_softbody(mesh);
    std::vector<Triangle> deformed_tris;
    build_deformed_tris(mesh, soft_body, deformed_tris);
    mesh.tris = deformed_tris;

    std::vector<int> tri_idx(mesh.tris.size());
    for (size_t i = 0; i < tri_idx.size(); i++) tri_idx[i] = (int)i;

    std::vector<BVHNode> nodes;
    nodes.reserve(mesh.tris.size() * 2);
    build_bvh_recursive(nodes, tri_idx, mesh.tris, 0, (int)mesh.tris.size());
    fprintf(stderr, "[BVH] nodes=%zu\n", nodes.size());

    const int base_w = 640;
    const int base_h = 360;

    auto trim = [](const std::string& s) {
        size_t i = 0;
        while (i < s.size() && std::isspace((unsigned char)s[i])) i++;
        size_t j = s.size();
        while (j > i && std::isspace((unsigned char)s[j - 1])) j--;
        return s.substr(i, j - i);
    };
    auto lower = [](std::string s) {
        for (char& c : s) c = (char)std::tolower((unsigned char)c);
        return s;
    };
    auto prompt_line = [](const char* prompt) {
        std::cout << prompt << std::flush;
        std::string line;
        std::getline(std::cin, line);
        return line;
    };
    auto prompt_float = [&](const char* prompt, float defv, float minv, float maxv) {
        while (true) {
            std::string line = trim(prompt_line(prompt));
            if (line.empty()) return defv;
            std::stringstream ss(line);
            float v = defv;
            if ((ss >> v) && ss.eof()) {
                return fminf(fmaxf(v, minv), maxv);
            }
            std::cout << "Invalid number, try again.\n";
        }
    };

    std::cout << "\nJelly color presets:\n";
    for (int i = 0; i < kPresetCount; i++) {
        std::cout << "  " << i << ": " << kPresets[i].name << "\n";
    }

    int selected_preset = 0;
    while (true) {
        std::string line = lower(trim(prompt_line("Color (index or name, default 0): ")));
        if (line.empty()) break;

        std::stringstream ss(line);
        int idx = -1;
        if ((ss >> idx) && ss.eof() && idx >= 0 && idx < kPresetCount) {
            selected_preset = idx;
            break;
        }

        bool found = false;
        for (int i = 0; i < kPresetCount; i++) {
            std::string nm = lower(kPresets[i].name);
            if (line == nm || nm.find(line) == 0) {
                selected_preset = i;
                found = true;
                break;
            }
        }
        if (found) break;
        std::cout << "Invalid preset, try again.\n";
    }

    float drop_height = prompt_float("Drop height [0.2..2.5] (default 1.15): ", 1.15f, 0.2f, 2.5f);
    float impact_boost = prompt_float("Impact boost [0..8] (default 1.3): ", 1.3f, 0.0f, 8.0f);
    float quality_scale = prompt_float("Quality scale [0.5..2] (default 1): ", 1.0f, 0.5f, 2.0f);
    float render_scale = prompt_float("Resolution scale [0.5..1] (default 0.75): ", 0.75f, 0.5f, 1.0f);
    float yaw_deg = prompt_float("View yaw deg (default 180): ", 180.0f, -720.0f, 720.0f);
    float pitch_deg = prompt_float("View pitch deg (default -8): ", -8.0f, -80.0f, 80.0f);

    int W = (int)lrintf(base_w * render_scale);
    int H = (int)lrintf(base_h * render_scale);
    W = std::max(W, 320);
    H = std::max(H, 180);
    if (W & 1) W++;
    if (H & 1) H++;

    CUDA_CHECK(cudaSetDevice(0));

    Vec3* d_accum = nullptr;
    int* d_accum_n = nullptr;
    uint32_t* d_rng = nullptr;
    uchar4* d_save = nullptr;
    Triangle* d_tris = nullptr;
    int* d_tri_idx = nullptr;
    BVHNode* d_nodes = nullptr;

    CUDA_CHECK(cudaMalloc(&d_accum, (size_t)W * H * sizeof(Vec3)));
    CUDA_CHECK(cudaMalloc(&d_accum_n, (size_t)W * H * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_rng, (size_t)W * H * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_save, (size_t)W * H * sizeof(uchar4)));
    CUDA_CHECK(cudaMalloc(&d_tris, mesh.tris.size() * sizeof(Triangle)));
    CUDA_CHECK(cudaMalloc(&d_tri_idx, tri_idx.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_nodes, nodes.size() * sizeof(BVHNode)));

    CUDA_CHECK(cudaMemcpy(d_tris, mesh.tris.data(), mesh.tris.size() * sizeof(Triangle), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tri_idx, tri_idx.data(), tri_idx.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_nodes, nodes.data(), nodes.size() * sizeof(BVHNode), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemset(d_accum, 0, (size_t)W * H * sizeof(Vec3)));
    CUDA_CHECK(cudaMemset(d_accum_n, 0, (size_t)W * H * sizeof(int)));
    {
        std::vector<uint32_t> seeds((size_t)W * H);
        for (int i = 0; i < W * H; i++) seeds[i] = (uint32_t)(1337u + i * 9781u + 123u);
        CUDA_CHECK(cudaMemcpy(d_rng, seeds.data(), (size_t)W * H * sizeof(uint32_t), cudaMemcpyHostToDevice));
    }

    DeviceScene dsc{};
    dsc.tris = d_tris;
    dsc.tri_idx = d_tri_idx;
    dsc.nodes = d_nodes;
    dsc.tri_count = (int)mesh.tris.size();
    dsc.node_count = (int)nodes.size();

    RenderParams rp{};
    rp.width = W;
    rp.height = H;
    rp.frame_index = 0;
    rp.spp_per_frame = 2;
    rp.max_bounces = 5;
    rp.fast_tracing = 0;
    rp.preview_mode = 0;
    rp.ior = kPresets[selected_preset].ior;
    rp.glass_roughness = kPresets[selected_preset].roughness;
    rp.sigma_a = kPresets[selected_preset].sigma_a;
    rp.sigma_s = kPresets[selected_preset].sigma_s;
    rp.medium_aniso_g = 0.35f;
    rp.ground_roughness = 0.11f;
    rp.light_center = Vec3(0.10f, 2.25f, 0.10f);
    rp.light_u = Vec3(0.78f, 0.0f, 0.0f);
    rp.light_v = Vec3(0.0f, 0.0f, 0.56f);
    rp.light_n = normalize(cross(rp.light_u, rp.light_v));
    rp.light_area = 4.0f * length(cross(rp.light_u, rp.light_v));
    rp.light_emission = Vec3(20.0f, 19.0f, 18.0f);

    Vec3 bsize = mesh.bmax - mesh.bmin;
    float radius = 0.5f * length(bsize);
    Vec3 target((mesh.bmin.x + mesh.bmax.x) * 0.5f,
                mesh.bmin.y + bsize.y * 0.45f,
                (mesh.bmin.z + mesh.bmax.z) * 0.5f);
    float cam_dist = fmaxf(radius * 2.45f, 1.7f);
    Camera cam = make_orbit_camera(target, cam_dist, yaw_deg, pitch_deg);

    const float ui_dummy_height01 = 0.5f;
    const int ui_dummy_force_sign = 1;
    const int fps = 20;
    const float frame_dt = 1.0f / (float)fps;
    const int min_frames = 72;
    const int max_frames = 120;
    const int max_total_spp = std::max(18, (int)lrintf(34.0f * quality_scale));
    const int min_total_spp = std::max(12, (int)lrintf(20.0f * quality_scale));
    const int warmup_base_spp = std::max(8, (int)lrintf(14.0f * quality_scale));
    const int ultra_quality_frames = 12;
    const int ultra_min_spp = std::max(min_total_spp, (int)lrintf(42.0f * quality_scale));
    const int early_quality_frames = 26;
    const int early_min_spp = std::max(min_total_spp, (int)lrintf(28.0f * quality_scale));

    std::cout << std::fixed << std::setprecision(3)
              << "\n[Render] preset=" << kPresets[selected_preset].name
              << ", drop_height=" << drop_height
              << ", impact_boost=" << impact_boost
              << ", yaw=" << yaw_deg
              << ", pitch=" << pitch_deg
              << ", size=" << W << "x" << H
              << ", total_spp=[" << min_total_spp << "," << max_total_spp << "]"
              << ", warmup_spp>=" << warmup_base_spp
              << ", ultra_spp>=" << ultra_min_spp
              << " for first " << ultra_quality_frames << " frames"
              << ", early_spp>=" << early_min_spp
              << " for first " << early_quality_frames << " frames\n";

    namespace fs = std::filesystem;
    fs::path frames_dir = fs::path("tmp_jelly_gif_frames");
    std::error_code ec;
    fs::remove_all(frames_dir, ec);
    fs::create_directories(frames_dir, ec);

    SoftBodyState sim = soft_body;
    setup_drop(sim, drop_height, impact_boost);

    std::vector<uint8_t> raw((size_t)W * H * 4);
    std::vector<uint8_t> png((size_t)W * H * 4);
    std::vector<std::vector<uint8_t>> frames_cpu;
    frames_cpu.reserve(max_frames);
    UIParams ui{};

    dim3 block(16, 16);
    dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y);
    int rendered_frames = 0;
    int render_pass_counter = 0;

    for (int fi = 0; fi < max_frames; fi++) {
        float kinetic = step_softbody_xpbd(sim, frame_dt, false);

        float sum_disp = 0.0f;
        float max_disp = 0.0f;
        Vec3 csum(0, 0, 0);
        for (size_t vi = 0; vi < sim.pos.size(); vi++) {
            float d = length(sim.pos[vi] - sim.rest_pos[vi]);
            sum_disp += d;
            if (d > max_disp) max_disp = d;
            csum += sim.pos[vi];
        }
        float avg_disp = sum_disp / fmaxf((float)sim.pos.size(), 1.0f);
        Vec3 center = csum / fmaxf((float)sim.pos.size(), 1.0f);
        float drop_offset = center.y - sim.center_rest.y;
        float deform01 = fminf(fmaxf(max_disp / 0.26f, 0.0f), 1.0f);
        float kinetic01 = fminf(fmaxf(sqrtf(fmaxf(kinetic, 0.0f)) * 0.42f, 0.0f), 1.0f);
        float motion01 = fminf(fmaxf(fmaxf(deform01, kinetic01), 0.0f), 1.0f);

        int frame_total_spp = (int)lrintf((1.0f - motion01) * max_total_spp + motion01 * min_total_spp);
        frame_total_spp = std::max(min_total_spp, std::min(max_total_spp, frame_total_spp));
        bool ultra_phase = (fi < ultra_quality_frames);
        bool early_phase = (fi < early_quality_frames);
        if (ultra_phase) {
            frame_total_spp = std::max(frame_total_spp, ultra_min_spp);
        }
        if (early_phase) {
            frame_total_spp = std::max(frame_total_spp, early_min_spp);
        }
        int warmup_spp = std::max(warmup_base_spp, frame_total_spp / 2);
        if (early_phase) warmup_spp = std::max(warmup_spp, warmup_base_spp + 6);
        rp.spp_per_frame = 2;
        int warmup_passes = (warmup_spp + rp.spp_per_frame - 1) / rp.spp_per_frame;
        int offline_passes = (frame_total_spp + rp.spp_per_frame - 1) / rp.spp_per_frame;
        rp.max_bounces = 6;
        rp.fast_tracing = 0;
        rp.preview_mode = 0;

        build_deformed_tris(mesh, sim, deformed_tris);
        CUDA_CHECK(cudaMemcpy(d_tris, deformed_tris.data(), deformed_tris.size() * sizeof(Triangle), cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMemset(d_accum, 0, (size_t)W * H * sizeof(Vec3)));
        CUDA_CHECK(cudaMemset(d_accum_n, 0, (size_t)W * H * sizeof(int)));

        for (int p = 0; p < warmup_passes; p++) {
            rp.frame_index = render_pass_counter++;
            render_kernel<<<grid, block>>>(d_accum, d_accum_n, d_rng, dsc, cam, rp);
            CUDA_CHECK(cudaGetLastError());
        }
        for (int p = 0; p < offline_passes; p++) {
            rp.frame_index = render_pass_counter++;
            render_kernel<<<grid, block>>>(d_accum, d_accum_n, d_rng, dsc, cam, rp);
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        tonemap_ui_kernel<<<grid, block>>>(
            d_accum, d_accum_n, d_save, rp, ui, selected_preset,
            0, ui_dummy_height01, ui_dummy_force_sign, 1, 0);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(raw.data(), d_save, raw.size(), cudaMemcpyDeviceToHost));

        for (int yy = 0; yy < H; yy++) {
            const uint8_t* src = &raw[(size_t)(H - 1 - yy) * W * 4];
            std::memcpy(&png[(size_t)yy * W * 4], src, (size_t)W * 4);
        }
        frames_cpu.push_back(png);

        rendered_frames = fi + 1;

        if ((rendered_frames % 5) == 0 || rendered_frames == 1) {
            std::cout << "[Sim] frame " << rendered_frames
                      << " avg_disp=" << avg_disp
                      << " max_disp=" << max_disp
                      << " drop_y=" << drop_offset
                      << " kinetic=" << kinetic
                      << " warmup=" << warmup_spp
                      << " spp=" << frame_total_spp
                      << " bounces=" << rp.max_bounces << "\n";
        }

        if (rendered_frames >= min_frames && !sim.active && max_disp < 0.0040f && kinetic < 6e-6f) break;
    }

    // Temporal + spatial denoise before writing PNG sequence for GIF encode.
    std::vector<uint8_t> temporal((size_t)W * H * 4);
    std::vector<uint8_t> filtered((size_t)W * H * 4);
    auto spatial_filter = [&](const std::vector<uint8_t>& src,
                              std::vector<uint8_t>& dst,
                              int radius,
                              int color_thr)
    {
        for (int y = 0; y < H; y++) {
            for (int x = 0; x < W; x++) {
                int base = (y * W + x) * 4;
                int cr = src[base + 0];
                int cg = src[base + 1];
                int cb = src[base + 2];
                float sumr = 0.0f, sumg = 0.0f, sumb = 0.0f, wsum = 0.0f;

                for (int dy = -radius; dy <= radius; dy++) {
                    int yy = y + dy;
                    if (yy < 0 || yy >= H) continue;
                    for (int dx = -radius; dx <= radius; dx++) {
                        int xx = x + dx;
                        if (xx < 0 || xx >= W) continue;
                        int nb = (yy * W + xx) * 4;
                        int nr = src[nb + 0];
                        int ng = src[nb + 1];
                        int nbv = src[nb + 2];
                        int cd = std::abs(nr - cr) + std::abs(ng - cg) + std::abs(nbv - cb);
                        if (cd > color_thr) continue;

                        float ds2 = (float)(dx * dx + dy * dy);
                        float w = 1.0f / (1.0f + ds2);
                        sumr += w * (float)nr;
                        sumg += w * (float)ng;
                        sumb += w * (float)nbv;
                        wsum += w;
                    }
                }

                if (wsum > 0.0f) {
                    dst[base + 0] = (uint8_t)fminf(fmaxf(sumr / wsum, 0.0f), 255.0f);
                    dst[base + 1] = (uint8_t)fminf(fmaxf(sumg / wsum, 0.0f), 255.0f);
                    dst[base + 2] = (uint8_t)fminf(fmaxf(sumb / wsum, 0.0f), 255.0f);
                } else {
                    dst[base + 0] = (uint8_t)cr;
                    dst[base + 1] = (uint8_t)cg;
                    dst[base + 2] = (uint8_t)cb;
                }
                dst[base + 3] = src[base + 3];
            }
        }
    };

    for (int fi = 0; fi < rendered_frames; fi++) {
        const std::vector<uint8_t>& cur = frames_cpu[fi];
        bool ultra_phase = (fi < ultra_quality_frames);
        bool early_phase = (fi < early_quality_frames);
        for (int px = 0; px < W * H; px++) {
            int base = px * 4;
            for (int c = 0; c < 3; c++) {
                int c0 = cur[base + c];
                if (ultra_phase) {
                    temporal[base + c] = (uint8_t)c0;
                } else {
                    int center_w = early_phase ? 3 : 2;
                    int thr1 = early_phase ? 36 : 24;
                    int thr2 = early_phase ? 50 : 32;
                    int sum = center_w * c0;
                    int wsum = center_w;
                    if (fi > 0) {
                        int cp = frames_cpu[fi - 1][base + c];
                        int d = std::abs(cp - c0);
                        if (d <= thr1) {
                            int w = (d <= (thr1 / 2)) ? 2 : 1;
                            sum += cp * w;
                            wsum += w;
                        }
                    }
                    if (fi + 1 < rendered_frames) {
                        int cn = frames_cpu[fi + 1][base + c];
                        int d = std::abs(cn - c0);
                        if (d <= thr1) {
                            int w = (d <= (thr1 / 2)) ? 2 : 1;
                            sum += cn * w;
                            wsum += w;
                        }
                    }
                    if (early_phase && fi > 1) {
                        int cp2 = frames_cpu[fi - 2][base + c];
                        int d2 = std::abs(cp2 - c0);
                        if (d2 <= thr2) {
                            sum += cp2;
                            wsum += 1;
                        }
                    }
                    if (early_phase && fi + 2 < rendered_frames) {
                        int cn2 = frames_cpu[fi + 2][base + c];
                        int d2 = std::abs(cn2 - c0);
                        if (d2 <= thr2) {
                            sum += cn2;
                            wsum += 1;
                        }
                    }
                    temporal[base + c] = (uint8_t)(sum / wsum);
                }
            }
            temporal[base + 3] = cur[base + 3];
        }

        int radius = ultra_phase ? 2 : 1;
        int color_thr = ultra_phase ? 72 : (early_phase ? 56 : 42);
        spatial_filter(temporal, filtered, radius, color_thr);

        char fn[64];
        std::snprintf(fn, sizeof(fn), "frame_%04d.png", fi);
        fs::path out_png = frames_dir / fn;
        stbi_write_png(out_png.string().c_str(), W, H, 4, filtered.data(), W * 4);
    }

    std::time_t tt = std::time(nullptr);
    std::tm tmv{};
    localtime_r(&tt, &tmv);
    char out_name[128];
    std::strftime(out_name, sizeof(out_name), "jelly_drop_wobble_%Y%m%d_%H%M%S.gif", &tmv);
    fs::path palette = frames_dir / "palette.png";

    std::string seq = (frames_dir / "frame_%04d.png").string();
    std::string cmd_palette = "ffmpeg -y -loglevel error -framerate " + std::to_string(fps) +
                              " -i " + seq +
                              " -vf palettegen=stats_mode=full " + palette.string();
    std::string cmd_gif = "ffmpeg -y -loglevel error -framerate " + std::to_string(fps) +
                          " -i " + seq +
                          " -i " + palette.string() +
                          " -lavfi paletteuse=dither=bayer:bayer_scale=3 " + std::string(out_name);

    int rc1 = std::system(cmd_palette.c_str());
    int rc2 = std::system(cmd_gif.c_str());
    if (rc1 == 0 && rc2 == 0) {
        std::cout << "[GIF] saved " << out_name << " (" << rendered_frames << " frames)\n";
    } else {
        std::cout << "[GIF] ffmpeg failed. Please install ffmpeg.\n";
    }

    CUDA_CHECK(cudaFree(d_accum));
    CUDA_CHECK(cudaFree(d_accum_n));
    CUDA_CHECK(cudaFree(d_rng));
    CUDA_CHECK(cudaFree(d_save));
    CUDA_CHECK(cudaFree(d_tris));
    CUDA_CHECK(cudaFree(d_tri_idx));
    CUDA_CHECK(cudaFree(d_nodes));

    return 0;
}
