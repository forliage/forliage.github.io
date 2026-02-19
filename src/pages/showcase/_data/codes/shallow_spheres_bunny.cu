// Build (Debian + CUDA 12.9 + RTX 4060 example):
//   nvcc -O3 -std=c++17 -arch=sm_89 shallow_spheres_bunny_cuda.cu -o shallow_spheres_bunny_cuda \
//       -lglfw -ldl -lpthread -lm
//
// Run examples:
//   ./shallow_spheres_bunny_cuda spheres stanford-bunny.obj 360 shallow_frames
//   ./shallow_spheres_bunny_cuda bunny   stanford-bunny.obj 360 shallow_frames
//   ./shallow_spheres_bunny_cuda both    stanford-bunny.obj 360 shallow_frames
// Output:
//   shallow_frames/top/frame_XXXX.png
//
// Notes:
//   - Single-file CUDA shallow-water engine inspired by:
//     "Solving General Shallow Wave Equations on Surfaces" (Wang, Miller, Turk 2007).
//   - SPHERES setting follows Table 1 spirit: regular grid, g=9.8, gamma=0, tau=0.
//   - No CUDA-OpenGL interop is used. GLFW is used only for event polling/title updates.

#include <cuda_runtime.h>
#include <GLFW/glfw3.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
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
        std::fprintf(stderr, "CUDA error %s at %s:%d -> %s\n", #stmt, __FILE__, __LINE__, cudaGetErrorString(_err)); \
        std::exit(1); \
    } \
} while (0)

struct Vec2 {
    float x, y;
    Vec2() : x(0.0f), y(0.0f) {}
    Vec2(float X, float Y) : x(X), y(Y) {}
};

struct Vec3 {
    float x, y, z;
    __host__ __device__ Vec3() : x(0.0f), y(0.0f), z(0.0f) {}
    __host__ __device__ Vec3(float X, float Y, float Z) : x(X), y(Y), z(Z) {}

    __host__ __device__ Vec3 operator+(const Vec3& b) const { return Vec3(x + b.x, y + b.y, z + b.z); }
    __host__ __device__ Vec3 operator-(const Vec3& b) const { return Vec3(x - b.x, y - b.y, z - b.z); }
    __host__ __device__ Vec3 operator*(float s) const { return Vec3(x * s, y * s, z * s); }
    __host__ __device__ Vec3 operator/(float s) const {
        float inv = 1.0f / s;
        return Vec3(x * inv, y * inv, z * inv);
    }
    __host__ __device__ Vec3& operator+=(const Vec3& b) {
        x += b.x; y += b.y; z += b.z;
        return *this;
    }
};

__host__ __device__ inline Vec3 operator*(float s, const Vec3& v) {
    return Vec3(v.x * s, v.y * s, v.z * s);
}
__host__ __device__ inline float dot3(const Vec3& a, const Vec3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
__host__ __device__ inline float length3(const Vec3& v) {
    return sqrtf(dot3(v, v));
}
__host__ __device__ inline Vec3 normalize3(const Vec3& v) {
    float l = length3(v);
    return (l > 1e-8f) ? (v / l) : Vec3(0.0f, 0.0f, 0.0f);
}
__host__ __device__ inline float clampf(float x, float a, float b) {
    return fminf(fmaxf(x, a), b);
}
__host__ __device__ inline float mixf(float a, float b, float t) {
    return a + (b - a) * t;
}

static inline uint32_t xorshift32(uint32_t& s) {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    return s;
}

static inline float rand01(uint32_t& s) {
    return (xorshift32(s) & 0x00FFFFFFu) * (1.0f / 16777216.0f);
}

static inline std::string trim(const std::string& s) {
    size_t b = 0;
    while (b < s.size() && std::isspace((unsigned char)s[b])) b++;
    size_t e = s.size();
    while (e > b && std::isspace((unsigned char)s[e - 1])) e--;
    return s.substr(b, e - b);
}

static inline int parse_obj_index(const std::string& token, int vcount) {
    size_t slash = token.find('/');
    std::string id = (slash == std::string::npos) ? token : token.substr(0, slash);
    int idx = std::stoi(id);
    if (idx < 0) idx = vcount + idx + 1;
    return idx - 1;
}

struct Tri {
    Vec3 a, b, c;
};

static bool load_obj_triangles(const std::string& path, std::vector<Tri>& tris) {
    std::ifstream in(path);
    if (!in) {
        return false;
    }

    std::vector<Vec3> verts;
    verts.reserve(100000);
    tris.clear();

    std::string line;
    while (std::getline(in, line)) {
        line = trim(line);
        if (line.empty() || line[0] == '#') continue;

        std::istringstream ss(line);
        std::string tag;
        ss >> tag;
        if (tag == "v") {
            float x, y, z;
            ss >> x >> y >> z;
            // OBJ (x,y,z), we convert to world: X=x, Y=z, Z=y so Z is vertical.
            verts.emplace_back(x, z, y);
        } else if (tag == "f") {
            std::vector<std::string> tokens;
            std::string t;
            while (ss >> t) tokens.push_back(t);
            if (tokens.size() < 3) continue;

            int i0 = parse_obj_index(tokens[0], (int)verts.size());
            for (size_t k = 1; k + 1 < tokens.size(); ++k) {
                int i1 = parse_obj_index(tokens[k], (int)verts.size());
                int i2 = parse_obj_index(tokens[k + 1], (int)verts.size());
                if (i0 < 0 || i1 < 0 || i2 < 0 ||
                    i0 >= (int)verts.size() || i1 >= (int)verts.size() || i2 >= (int)verts.size()) {
                    continue;
                }
                tris.push_back({verts[i0], verts[i1], verts[i2]});
            }
        }
    }
    return !tris.empty();
}

static void normalize_bunny_mesh(std::vector<Tri>& tris) {
    Vec3 bmin(1e30f, 1e30f, 1e30f);
    Vec3 bmax(-1e30f, -1e30f, -1e30f);
    for (const Tri& t : tris) {
        bmin.x = std::min(bmin.x, std::min(t.a.x, std::min(t.b.x, t.c.x)));
        bmin.y = std::min(bmin.y, std::min(t.a.y, std::min(t.b.y, t.c.y)));
        bmin.z = std::min(bmin.z, std::min(t.a.z, std::min(t.b.z, t.c.z)));
        bmax.x = std::max(bmax.x, std::max(t.a.x, std::max(t.b.x, t.c.x)));
        bmax.y = std::max(bmax.y, std::max(t.a.y, std::max(t.b.y, t.c.y)));
        bmax.z = std::max(bmax.z, std::max(t.a.z, std::max(t.b.z, t.c.z)));
    }
    Vec3 ext = bmax - bmin;
    float max_ext = std::max(ext.x, std::max(ext.y, ext.z));
    float scale = 1.0f / std::max(max_ext, 1e-6f);
    Vec3 center = 0.5f * (bmin + bmax);

    for (Tri& t : tris) {
        t.a = (t.a - center) * scale;
        t.b = (t.b - center) * scale;
        t.c = (t.c - center) * scale;
    }
}

struct TriXY {
    Vec3 a, b, c;
    float minx, maxx;
    float miny, maxy;
    float det;
    bool valid;
};

struct BunnyProfile {
    int res_xy = 0;
    float min_x = 0.0f, max_x = 0.0f;
    float min_y = 0.0f, max_y = 0.0f;
    float min_z = 0.0f, max_z = 0.0f;
    float volume = 0.0f; // local unit volume after normalization
    // top view (x, y) -> z range
    std::vector<float> z_bot_xy;
    std::vector<float> z_top_xy;
    std::vector<float> mask_xy;
};

static BunnyProfile build_bunny_profile(const std::vector<Tri>& tris, int res_xy) {
    BunnyProfile p;
    p.res_xy = res_xy;
    p.z_bot_xy.assign((size_t)res_xy * res_xy, 0.0f);
    p.z_top_xy.assign((size_t)res_xy * res_xy, 0.0f);
    p.mask_xy.assign((size_t)res_xy * res_xy, 0.0f);

    Vec3 bmin(1e30f, 1e30f, 1e30f);
    Vec3 bmax(-1e30f, -1e30f, -1e30f);
    std::vector<TriXY> tri_xy;
    tri_xy.reserve(tris.size());

    for (const Tri& t : tris) {
        bmin.x = std::min(bmin.x, std::min(t.a.x, std::min(t.b.x, t.c.x)));
        bmin.y = std::min(bmin.y, std::min(t.a.y, std::min(t.b.y, t.c.y)));
        bmin.z = std::min(bmin.z, std::min(t.a.z, std::min(t.b.z, t.c.z)));
        bmax.x = std::max(bmax.x, std::max(t.a.x, std::max(t.b.x, t.c.x)));
        bmax.y = std::max(bmax.y, std::max(t.a.y, std::max(t.b.y, t.c.y)));
        bmax.z = std::max(bmax.z, std::max(t.a.z, std::max(t.b.z, t.c.z)));

        TriXY q{};
        q.a = t.a; q.b = t.b; q.c = t.c;
        q.minx = std::min(t.a.x, std::min(t.b.x, t.c.x));
        q.maxx = std::max(t.a.x, std::max(t.b.x, t.c.x));
        q.miny = std::min(t.a.y, std::min(t.b.y, t.c.y));
        q.maxy = std::max(t.a.y, std::max(t.b.y, t.c.y));
        float m00 = t.b.x - t.a.x;
        float m01 = t.c.x - t.a.x;
        float m10 = t.b.y - t.a.y;
        float m11 = t.c.y - t.a.y;
        q.det = m00 * m11 - m01 * m10;
        q.valid = std::fabs(q.det) > 1e-10f;
        tri_xy.push_back(q);

    }

    p.min_x = bmin.x; p.max_x = bmax.x;
    p.min_y = bmin.y; p.max_y = bmax.y;
    p.min_z = bmin.z; p.max_z = bmax.z;

    // Build (x, y)->z top/bottom profile and estimate volume.
    const float sx_xy = (p.max_x - p.min_x) / (float)res_xy;
    const float sy_xy = (p.max_y - p.min_y) / (float)res_xy;
    const float cell_area_xy = sx_xy * sy_xy;

    std::vector<float> vals;
    vals.reserve(256);
    double vol_accum = 0.0;

    for (int j = 0; j < res_xy; ++j) {
        for (int i = 0; i < res_xy; ++i) {
            float px = p.min_x + (i + 0.5f) * sx_xy;
            float py = p.min_y + (j + 0.5f) * sy_xy;
            vals.clear();

            for (const TriXY& t : tri_xy) {
                if (!t.valid) continue;
                if (px < t.minx - 1e-6f || px > t.maxx + 1e-6f) continue;
                if (py < t.miny - 1e-6f || py > t.maxy + 1e-6f) continue;

                float ax = t.a.x, ay = t.a.y;
                float bx = t.b.x, by = t.b.y;
                float cx = t.c.x, cy = t.c.y;

                float u = ((px - ax) * (cy - ay) - (py - ay) * (cx - ax)) / t.det;
                float v = ((bx - ax) * (py - ay) - (by - ay) * (px - ax)) / t.det;
                float w = 1.0f - u - v;
                if (u >= -1e-6f && v >= -1e-6f && w >= -1e-6f) {
                    float z = w * t.a.z + u * t.b.z + v * t.c.z;
                    vals.push_back(z);
                }
            }

            const size_t idx = (size_t)j * res_xy + i;
            if (vals.size() < 2) {
                continue;
            }

            std::sort(vals.begin(), vals.end());
            std::vector<float> uniq;
            uniq.reserve(vals.size());
            for (float z : vals) {
                if (uniq.empty() || std::fabs(z - uniq.back()) > 1e-5f) {
                    uniq.push_back(z);
                }
            }
            if (uniq.size() < 2) continue;

            float z0 = uniq.front();
            float z1 = uniq.back();
            if (z1 <= z0) continue;

            p.mask_xy[idx] = 1.0f;
            p.z_bot_xy[idx] = z0;
            p.z_top_xy[idx] = z1;
            vol_accum += (double)(z1 - z0) * (double)cell_area_xy;
        }
    }

    p.volume = (float)vol_accum;
    return p;
}

static inline float sample_host_bilinear_world(
    const std::vector<float>& f, int nx, int ny,
    float x, float y, float dx, float dy)
{
    x = std::max(0.0f, std::min(x, nx * dx - 1e-6f));
    y = std::max(0.0f, std::min(y, ny * dy - 1e-6f));

    float gx = x / dx - 0.5f;
    float gy = y / dy - 0.5f;
    int x0 = (int)std::floor(gx);
    int y0 = (int)std::floor(gy);
    int x1 = x0 + 1;
    int y1 = y0 + 1;
    float tx = gx - x0;
    float ty = gy - y0;

    auto read = [&](int ix, int iy) -> float {
        ix = std::max(0, std::min(ix, nx - 1));
        iy = std::max(0, std::min(iy, ny - 1));
        return f[(size_t)iy * nx + ix];
    };

    float c00 = read(x0, y0);
    float c10 = read(x1, y0);
    float c01 = read(x0, y1);
    float c11 = read(x1, y1);
    float c0 = c00 + (c10 - c00) * tx;
    float c1 = c01 + (c11 - c01) * tx;
    return c0 + (c1 - c0) * ty;
}

static inline Vec2 sample_host_gradient_world(
    const std::vector<float>& f, int nx, int ny,
    float x, float y, float dx, float dy)
{
    float hL = sample_host_bilinear_world(f, nx, ny, x - dx, y, dx, dy);
    float hR = sample_host_bilinear_world(f, nx, ny, x + dx, y, dx, dy);
    float hD = sample_host_bilinear_world(f, nx, ny, x, y - dy, dx, dy);
    float hU = sample_host_bilinear_world(f, nx, ny, x, y + dy, dx, dy);
    return Vec2((hR - hL) / (2.0f * dx), (hU - hD) / (2.0f * dy));
}

static inline float sphere_cap_volume(float R, float h) {
    h = std::max(0.0f, std::min(h, 2.0f * R));
    return (float)(M_PI) * h * h * (R - h / 3.0f);
}

enum ShapeType {
    SHAPE_SPHERE = 0,
    SHAPE_BUNNY = 1
};

struct Body {
    int shape = SHAPE_SPHERE;
    float mass = 1.0f;
    float radius_eq = 0.1f; // used for buoyancy proxy and splash radius
    float scale = 1.0f;     // bunny local->world scale

    float x = 0.0f, y = 0.0f, z = 0.0f;
    float vx = 0.0f, vy = 0.0f, vz = 0.0f;
    float yaw = 0.0f, yaw_rate = 0.0f;
};

static inline float bilerp_profile(
    const std::vector<float>& a, int res, float u, float v)
{
    if (u < 0.0f || v < 0.0f || u > (float)(res - 1) || v > (float)(res - 1)) return 0.0f;
    int x0 = (int)std::floor(u), y0 = (int)std::floor(v);
    int x1 = std::min(x0 + 1, res - 1);
    int y1 = std::min(y0 + 1, res - 1);
    float tx = u - x0;
    float ty = v - y0;

    float c00 = a[(size_t)y0 * res + x0];
    float c10 = a[(size_t)y0 * res + x1];
    float c01 = a[(size_t)y1 * res + x0];
    float c11 = a[(size_t)y1 * res + x1];
    float c0 = c00 + (c10 - c00) * tx;
    float c1 = c01 + (c11 - c01) * tx;
    return c0 + (c1 - c0) * ty;
}

static void stamp_gaussian(
    std::vector<float>& src, int nx, int ny, float dx, float dy,
    float cx, float cy, float sigma, float amp)
{
    if (sigma <= 1e-6f || std::fabs(amp) < 1e-10f) return;
    float r = 3.0f * sigma;
    int ix0 = std::max(0, (int)std::floor((cx - r) / dx));
    int ix1 = std::min(nx - 1, (int)std::ceil((cx + r) / dx));
    int iy0 = std::max(0, (int)std::floor((cy - r) / dy));
    int iy1 = std::min(ny - 1, (int)std::ceil((cy + r) / dy));
    float inv2 = 1.0f / (2.0f * sigma * sigma);

    for (int j = iy0; j <= iy1; ++j) {
        float y = (j + 0.5f) * dy;
        for (int i = ix0; i <= ix1; ++i) {
            float x = (i + 0.5f) * dx;
            float d2 = (x - cx) * (x - cx) + (y - cy) * (y - cy);
            src[(size_t)j * nx + i] += amp * std::exp(-d2 * inv2);
        }
    }
}

static void update_bodies_and_build_source(
    std::vector<Body>& bodies,
    const BunnyProfile& bunny,
    bool has_bunny,
    const std::vector<float>& h,
    int nx, int ny, float dx, float dy,
    float domain_x, float domain_y,
    float dt, float rho, float g,
    std::vector<float>& src)
{
    std::fill(src.begin(), src.end(), 0.0f);

    const float drag_z = 1.8f;
    const float drag_xy = 1.2f;
    const float slope_push = 0.42f;

    for (Body& b : bodies) {
        float eta = sample_host_bilinear_world(h, nx, ny, b.x, b.y, dx, dy);
        Vec2 grad = sample_host_gradient_world(h, nx, ny, b.x, b.y, dx, dy);

        float volume = 4.0f / 3.0f * (float)M_PI * b.radius_eq * b.radius_eq * b.radius_eq;
        if (b.shape == SHAPE_BUNNY && has_bunny) {
            volume = bunny.volume * b.scale * b.scale * b.scale;
        }
        float proxy_R = b.radius_eq;
        float submerged = eta - (b.z - proxy_R);
        float capV = sphere_cap_volume(proxy_R, submerged);
        float maxV = 4.0f / 3.0f * (float)M_PI * proxy_R * proxy_R * proxy_R;
        float buoyV = (maxV > 1e-8f) ? capV * (volume / maxV) : 0.0f;
        buoyV = std::max(0.0f, std::min(buoyV, volume));

        float Fb = rho * g * buoyV;
        float Fg = b.mass * g;
        float Fd = -drag_z * b.vz;
        float az = (Fb - Fg + Fd) / std::max(b.mass, 1e-5f);

        float ax = -slope_push * g * grad.x - drag_xy * b.vx / std::max(b.mass, 1e-5f);
        float ay = -slope_push * g * grad.y - drag_xy * b.vy / std::max(b.mass, 1e-5f);

        b.vz += dt * az;
        b.vx += dt * ax;
        b.vy += dt * ay;
        b.x += dt * b.vx;
        b.y += dt * b.vy;
        b.z += dt * b.vz;
        b.yaw += dt * b.yaw_rate;

        float local_min_z = -b.radius_eq;
        if (b.shape == SHAPE_BUNNY && has_bunny) {
            local_min_z = bunny.min_z * b.scale;
        }
        float floor_z = 0.005f;
        if (b.z + local_min_z < floor_z) {
            b.z = floor_z - local_min_z;
            if (b.vz < 0.0f) b.vz *= -0.38f;
        }

        float margin = std::max(0.10f, 1.25f * b.radius_eq);
        if (b.x < margin) {
            b.x = margin;
            if (b.vx < 0.0f) b.vx *= -0.55f;
        }
        if (b.x > domain_x - margin) {
            b.x = domain_x - margin;
            if (b.vx > 0.0f) b.vx *= -0.55f;
        }
        if (b.y < margin) {
            b.y = margin;
            if (b.vy < 0.0f) b.vy *= -0.55f;
        }
        if (b.y > domain_y - margin) {
            b.y = domain_y - margin;
            if (b.vy > 0.0f) b.vy *= -0.55f;
        }

        float splash_amp = 0.0f;
        if (submerged > 0.0f && b.vz < -0.01f) splash_amp += -b.vz * 0.018f;
        float speed_xy = std::sqrt(b.vx * b.vx + b.vy * b.vy);
        splash_amp += speed_xy * 0.0008f;
        splash_amp *= (1.0f + 0.6f * b.radius_eq);

        if (splash_amp > 1e-7f) {
            float sigma = std::max(0.035f, 0.9f * b.radius_eq);
            stamp_gaussian(src, nx, ny, dx, dy, b.x, b.y, sigma, splash_amp);
        }
    }
}

static void build_coupling_maps(
    const std::vector<Body>& bodies,
    const BunnyProfile& bunny,
    bool has_bunny,
    int nx, int ny, float dx, float dy, float base_h,
    std::vector<float>& c_weight,
    std::vector<float>& c_target,
    std::vector<float>& obj_mask)
{
    std::fill(c_weight.begin(), c_weight.end(), 0.0f);
    std::fill(c_target.begin(), c_target.end(), 0.0f);
    std::fill(obj_mask.begin(), obj_mask.end(), 0.0f);

    const float k_couple = 30.0f;

    for (const Body& b : bodies) {
        if (b.shape == SHAPE_SPHERE) {
            float R = b.radius_eq;
            int ix0 = std::max(0, (int)std::floor((b.x - R) / dx));
            int ix1 = std::min(nx - 1, (int)std::ceil((b.x + R) / dx));
            int iy0 = std::max(0, (int)std::floor((b.y - R) / dy));
            int iy1 = std::min(ny - 1, (int)std::ceil((b.y + R) / dy));

            for (int j = iy0; j <= iy1; ++j) {
                float y = (j + 0.5f) * dy;
                for (int i = ix0; i <= ix1; ++i) {
                    float x = (i + 0.5f) * dx;
                    float rx = x - b.x;
                    float ry = y - b.y;
                    float r2 = rx * rx + ry * ry;
                    if (r2 > R * R) continue;

                    float zbot = b.z - std::sqrt(std::max(0.0f, R * R - r2));
                    if (zbot > base_h + 0.30f) continue;
                    zbot = std::max(0.0f, zbot);

                    size_t idx = (size_t)j * nx + i;
                    c_weight[idx] = std::max(c_weight[idx], k_couple);
                    if (obj_mask[idx] < 0.5f) {
                        c_target[idx] = zbot;
                    } else {
                        c_target[idx] = std::min(c_target[idx], zbot);
                    }
                    obj_mask[idx] = 1.0f;
                }
            }
        } else if (b.shape == SHAPE_BUNNY && has_bunny) {
            float cs = std::cos(b.yaw);
            float sn = std::sin(b.yaw);

            float half_x = 0.5f * (bunny.max_x - bunny.min_x) * b.scale;
            float half_y = 0.5f * (bunny.max_y - bunny.min_y) * b.scale;
            float rad = std::sqrt(half_x * half_x + half_y * half_y);

            int ix0 = std::max(0, (int)std::floor((b.x - rad) / dx));
            int ix1 = std::min(nx - 1, (int)std::ceil((b.x + rad) / dx));
            int iy0 = std::max(0, (int)std::floor((b.y - rad) / dy));
            int iy1 = std::min(ny - 1, (int)std::ceil((b.y + rad) / dy));

            float span_x = std::max(1e-6f, bunny.max_x - bunny.min_x);
            float span_y = std::max(1e-6f, bunny.max_y - bunny.min_y);

            for (int j = iy0; j <= iy1; ++j) {
                float y = (j + 0.5f) * dy;
                for (int i = ix0; i <= ix1; ++i) {
                    float x = (i + 0.5f) * dx;

                    float wx = x - b.x;
                    float wy = y - b.y;
                    float lx = ( cs * wx + sn * wy) / b.scale;
                    float ly = (-sn * wx + cs * wy) / b.scale;

                    float u = (lx - bunny.min_x) / span_x * (bunny.res_xy - 1);
                    float v = (ly - bunny.min_y) / span_y * (bunny.res_xy - 1);
                    if (u < 0.0f || v < 0.0f || u > (float)(bunny.res_xy - 1) || v > (float)(bunny.res_xy - 1)) continue;

                    float mk = bilerp_profile(bunny.mask_xy, bunny.res_xy, u, v);
                    if (mk < 0.35f) continue;

                    float zbot_local = bilerp_profile(bunny.z_bot_xy, bunny.res_xy, u, v);
                    float zbot = b.z + b.scale * zbot_local;
                    if (zbot > base_h + 0.35f) continue;
                    zbot = std::max(0.0f, zbot);

                    size_t idx = (size_t)j * nx + i;
                    c_weight[idx] = std::max(c_weight[idx], k_couple);
                    if (obj_mask[idx] < 0.5f) {
                        c_target[idx] = zbot;
                    } else {
                        c_target[idx] = std::min(c_target[idx], zbot);
                    }
                    obj_mask[idx] = 1.0f;
                }
            }
        }
    }
}

static inline float bilerp_profile_clamped(
    const std::vector<float>& a, int res, float u, float v)
{
    u = std::max(0.0f, std::min(u, (float)(res - 1) - 1e-5f));
    v = std::max(0.0f, std::min(v, (float)(res - 1) - 1e-5f));
    return bilerp_profile(a, res, u, v);
}

static inline bool sample_bunny_top_surface(
    const BunnyProfile& bunny,
    const Body& b,
    float wx, float wy,
    float& z_top, float& z_bot, Vec3& normal)
{
    if (bunny.res_xy <= 1) return false;
    float cs = std::cos(b.yaw);
    float sn = std::sin(b.yaw);
    float qx = wx - b.x;
    float qy = wy - b.y;
    float lx = ( cs * qx + sn * qy) / b.scale;
    float ly = (-sn * qx + cs * qy) / b.scale;

    float span_x = std::max(1e-6f, bunny.max_x - bunny.min_x);
    float span_y = std::max(1e-6f, bunny.max_y - bunny.min_y);
    float u = (lx - bunny.min_x) / span_x * (bunny.res_xy - 1);
    float v = (ly - bunny.min_y) / span_y * (bunny.res_xy - 1);
    if (u < 0.0f || v < 0.0f || u > (float)(bunny.res_xy - 1) || v > (float)(bunny.res_xy - 1)) return false;

    float mk = bilerp_profile_clamped(bunny.mask_xy, bunny.res_xy, u, v);
    if (mk < 0.35f) return false;

    float zt_local = bilerp_profile_clamped(bunny.z_top_xy, bunny.res_xy, u, v);
    float zb_local = bilerp_profile_clamped(bunny.z_bot_xy, bunny.res_xy, u, v);
    z_top = b.z + b.scale * zt_local;
    z_bot = b.z + b.scale * zb_local;

    float z_u0 = bilerp_profile_clamped(bunny.z_top_xy, bunny.res_xy, u - 1.0f, v);
    float z_u1 = bilerp_profile_clamped(bunny.z_top_xy, bunny.res_xy, u + 1.0f, v);
    float z_v0 = bilerp_profile_clamped(bunny.z_top_xy, bunny.res_xy, u, v - 1.0f);
    float z_v1 = bilerp_profile_clamped(bunny.z_top_xy, bunny.res_xy, u, v + 1.0f);
    float dz_du = 0.5f * (z_u1 - z_u0);
    float dz_dv = 0.5f * (z_v1 - z_v0);
    float dz_dxl = dz_du * (bunny.res_xy - 1) / span_x;
    float dz_dyl = dz_dv * (bunny.res_xy - 1) / span_y;
    float dz_dxw = dz_dxl * cs - dz_dyl * sn;
    float dz_dyw = dz_dxl * sn + dz_dyl * cs;
    normal = normalize3(Vec3(-dz_dxw, -dz_dyw, 1.0f));
    return true;
}

static void render_topdown_cpu(
    const std::vector<float>& h,
    const std::vector<Body>& bodies,
    const BunnyProfile& bunny,
    bool has_bunny,
    int nx, int ny, float dx, float dy,
    int W, int H, float h_base,
    std::vector<uchar4>& out)
{
    out.resize((size_t)W * H);
    Vec3 light = normalize3(Vec3(-0.45f, 0.32f, 0.84f));
    Vec3 view(0.0f, 0.0f, 1.0f);

    for (int py = 0; py < H; ++py) {
        for (int px = 0; px < W; ++px) {
            float u = (px + 0.5f) / (float)W;
            float v = (py + 0.5f) / (float)H;
            float wx = u * nx * dx;
            float wy = (1.0f - v) * ny * dy;

            float hc = sample_host_bilinear_world(h, nx, ny, wx, wy, dx, dy);
            float hL = sample_host_bilinear_world(h, nx, ny, wx - dx, wy, dx, dy);
            float hR = sample_host_bilinear_world(h, nx, ny, wx + dx, wy, dx, dy);
            float hD = sample_host_bilinear_world(h, nx, ny, wx, wy - dy, dx, dy);
            float hU = sample_host_bilinear_world(h, nx, ny, wx, wy + dy, dx, dy);
            float dhdx = (hR - hL) / (2.0f * dx);
            float dhdy = (hU - hD) / (2.0f * dy);
            float lap = (hL + hR + hD + hU - 4.0f * hc) / (dx * dy);

            Vec3 n_w = normalize3(Vec3(-2.8f * dhdx, -2.8f * dhdy, 1.0f));
            float diff_w = std::max(0.0f, dot3(n_w, light));
            float fres = powf(1.0f - std::max(0.0f, n_w.z), 4.0f);
            float foam = clampf(std::fabs(lap) * 0.003f, 0.0f, 1.0f);
            float depth_t = clampf((hc - h_base) * 7.0f + 0.5f, 0.0f, 1.0f);

            Vec3 deep(0.05f, 0.17f, 0.35f);
            Vec3 shallow(0.32f, 0.60f, 0.84f);
            Vec3 col = deep * (1.0f - depth_t) + shallow * depth_t;
            col = col * (0.30f + 0.75f * diff_w);
            col += Vec3(0.20f, 0.24f, 0.28f) * fres;
            col = col + Vec3(0.85f, 0.88f, 0.90f) * (0.35f * foam);

            bool hit = false;
            int hit_shape = -1;
            float best_z = -1e30f;
            float best_z_bot = 0.0f;
            Vec3 best_n(0.0f, 0.0f, 1.0f);

            for (const Body& b : bodies) {
                if (b.shape == SHAPE_SPHERE) {
                    float rx = wx - b.x;
                    float ry = wy - b.y;
                    float r2 = rx * rx + ry * ry;
                    float R = b.radius_eq;
                    if (r2 > R * R) continue;
                    float dz = std::sqrt(std::max(0.0f, R * R - r2));
                    float z_top = b.z + dz;
                    float z_bot = b.z - dz;
                    if (z_top > best_z) {
                        best_z = z_top;
                        best_z_bot = z_bot;
                        best_n = normalize3(Vec3(rx, ry, dz));
                        hit = true;
                        hit_shape = SHAPE_SPHERE;
                    }
                } else if (b.shape == SHAPE_BUNNY && has_bunny) {
                    float z_top = 0.0f, z_bot = 0.0f;
                    Vec3 n_obj;
                    if (!sample_bunny_top_surface(bunny, b, wx, wy, z_top, z_bot, n_obj)) continue;
                    if (z_top > best_z) {
                        best_z = z_top;
                        best_z_bot = z_bot;
                        best_n = n_obj;
                        hit = true;
                        hit_shape = SHAPE_BUNNY;
                    }
                }
            }

            if (hit) {
                Vec3 base = (hit_shape == SHAPE_SPHERE) ? Vec3(0.96f, 0.82f, 0.15f) : Vec3(0.98f, 0.78f, 0.10f);
                float diff_o = std::max(0.0f, dot3(best_n, light));
                Vec3 hvec = normalize3(light + view);
                float spec_o = powf(std::max(0.0f, dot3(best_n, hvec)), 48.0f);
                Vec3 obj = base * (0.22f + 0.86f * diff_o) + Vec3(1.0f, 1.0f, 1.0f) * (0.20f * spec_o);

                float sub = clampf((hc - best_z_bot) / std::max(best_z - best_z_bot, 1e-5f), 0.0f, 1.0f);
                float under = clampf((hc - best_z) / 0.10f, 0.0f, 1.0f);
                obj = obj * (1.0f - 0.45f * under) + col * (0.45f * under);

                if (best_z > hc - 0.015f) {
                    col = obj;
                } else {
                    col = col * 0.82f + obj * (0.18f * (1.0f - under) + 0.08f * sub);
                }
            }

            col.x = powf(clampf(col.x, 0.0f, 1.0f), 1.0f / 2.2f);
            col.y = powf(clampf(col.y, 0.0f, 1.0f), 1.0f / 2.2f);
            col.z = powf(clampf(col.z, 0.0f, 1.0f), 1.0f / 2.2f);

            uchar4 o;
            o.x = (unsigned char)(clampf(col.x, 0.0f, 1.0f) * 255.0f);
            o.y = (unsigned char)(clampf(col.y, 0.0f, 1.0f) * 255.0f);
            o.z = (unsigned char)(clampf(col.z, 0.0f, 1.0f) * 255.0f);
            o.w = 255;
            out[(size_t)py * W + px] = o;
        }
    }
}

__device__ inline float read2D(const float* f, int x, int y, int nx, int ny) {
    x = max(0, min(nx - 1, x));
    y = max(0, min(ny - 1, y));
    return f[(size_t)y * nx + x];
}

__device__ inline float sample2D_world(
    const float* f, float x, float y, int nx, int ny, float dx, float dy)
{
    x = fminf(fmaxf(x, 0.0f), nx * dx - 1e-6f);
    y = fminf(fmaxf(y, 0.0f), ny * dy - 1e-6f);

    float gx = x / dx - 0.5f;
    float gy = y / dy - 0.5f;
    int x0 = (int)floorf(gx);
    int y0 = (int)floorf(gy);
    int x1 = x0 + 1;
    int y1 = y0 + 1;
    float tx = gx - x0;
    float ty = gy - y0;

    float c00 = read2D(f, x0, y0, nx, ny);
    float c10 = read2D(f, x1, y0, nx, ny);
    float c01 = read2D(f, x0, y1, nx, ny);
    float c11 = read2D(f, x1, y1, nx, ny);
    float c0 = c00 + (c10 - c00) * tx;
    float c1 = c01 + (c11 - c01) * tx;
    return c0 + (c1 - c0) * ty;
}

__global__ void init_fields_kernel(
    float* h_prev, float* h_cur, float* u, float* v,
    int nx, int ny, float base_h)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int n = nx * ny;
    if (id >= n) return;

    int x = id % nx;
    int y = id / nx;
    float fx = (x + 0.5f) / (float)nx;
    float fy = (y + 0.5f) / (float)ny;

    float rip = 0.002f * sinf(17.0f * fx) * cosf(13.0f * fy);
    float h0 = base_h + rip;
    h_prev[id] = h0;
    h_cur[id] = h0;
    u[id] = 0.0f;
    v[id] = 0.0f;
}

__global__ void advect_height_kernel(
    const float* h_cur, const float* u, const float* v, float* h_adv,
    int nx, int ny, float dx, float dy, float dt)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= nx || y >= ny) return;
    int id = y * nx + x;

    float wx = (x + 0.5f) * dx;
    float wy = (y + 0.5f) * dy;
    float ux = u[id];
    float vy = v[id];

    float px = wx - dt * ux;
    float py = wy - dt * vy;
    h_adv[id] = sample2D_world(h_cur, px, py, nx, ny, dx, dy);
}

__global__ void build_rhs_kernel(
    const float* h_adv,
    const float* h_prev,
    const float* src,
    float* rhs,
    int n, float tau, float dt)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n) return;
    float h = h_adv[id];
    rhs[id] = h + (1.0f - tau) * (h - h_prev[id]) + dt * src[id];
}

__global__ void jacobi_height_kernel(
    const float* rhs,
    const float* h_in,
    const float* c_weight,
    const float* c_target,
    float* h_out,
    int nx, int ny,
    float dt2_g_over_dx2,
    float min_h)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= nx || y >= ny) return;
    int id = y * nx + x;

    float hc = h_in[id];
    float hL = read2D(h_in, x - 1, y, nx, ny);
    float hR = read2D(h_in, x + 1, y, nx, ny);
    float hD = read2D(h_in, x, y - 1, nx, ny);
    float hU = read2D(h_in, x, y + 1, nx, ny);

    float sigma = dt2_g_over_dx2 * fmaxf(hc, min_h);
    float cw = c_weight[id];
    float b = rhs[id] + cw * c_target[id];
    float diag = 1.0f + 4.0f * sigma + cw;
    float numer = b + sigma * (hL + hR + hD + hU);
    float hnew = numer / fmaxf(diag, 1e-6f);
    h_out[id] = fmaxf(hnew, min_h);
}

__global__ void update_velocity_kernel(
    float* u, float* v,
    const float* h,
    int nx, int ny,
    float dt, float g, float dx, float dy,
    float friction_d)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= nx || y >= ny) return;
    int id = y * nx + x;

    float hL = read2D(h, x - 1, y, nx, ny);
    float hR = read2D(h, x + 1, y, nx, ny);
    float hD = read2D(h, x, y - 1, nx, ny);
    float hU = read2D(h, x, y + 1, nx, ny);
    float hc = h[id];

    float dhdx = (hR - hL) / (2.0f * dx);
    float dhdy = (hU - hD) / (2.0f * dy);

    float uu = u[id] - dt * g * dhdx;
    float vv = v[id] - dt * g * dhdy;

    float damp = hc / (hc + friction_d);
    damp = clampf(damp, 0.0f, 1.0f);
    u[id] = uu * damp;
    v[id] = vv * damp;
}

__global__ void damp_boundaries_kernel(
    float* h, float* u, float* v,
    int nx, int ny,
    int border,
    float vel_damp,
    float h_base,
    float h_relax)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= nx || y >= ny) return;
    int id = y * nx + x;

    int d = min(min(x, nx - 1 - x), min(y, ny - 1 - y));
    if (d >= border) return;

    float t = (float)d / (float)max(border - 1, 1);
    float k = mixf(0.55f, 1.0f, t);
    u[id] *= vel_damp * k;
    v[id] *= vel_damp * k;
    h[id] = mixf(h_base, h[id], h_relax * k);
}

int main(int argc, char** argv) {
    std::string scene = "bunny"; // spheres|bunny|both
    std::string obj_path = "stanford-bunny.obj";
    int frames = 360;
    std::string out_dir = "shallow_frames";

    int argi = 1;
    if (argi < argc) {
        std::string a = argv[argi];
        if (a == "spheres" || a == "bunny" || a == "both") {
            scene = a;
            argi++;
        }
    }
    if (argi < argc) obj_path = argv[argi++];
    if (argi < argc) frames = std::max(1, std::atoi(argv[argi++]));
    if (argi < argc) out_dir = argv[argi++];

    const int NX = 400;
    const int NY = 400;
    const int N = NX * NY;

    const float DOMAIN_X = 4.0f;
    const float DOMAIN_Y = 4.0f;
    const float dx = DOMAIN_X / NX;
    const float dy = DOMAIN_Y / NY;

    const float g = 9.8f;
    const float rho = 1000.0f;
    const float tau = 0.0f; // SPHERES scene: 0
    const float base_h = 0.18f;
    const float min_h = 0.001f;

    const float dt = 1.0f / 120.0f;
    const float dt2_g_over_dx2 = dt * dt * g / (dx * dx);
    const int jacobi_iters = 48;
    const float surface_friction_d = 0.055f;

    const int W = 960;
    const int H = 960;

    std::string out_top = out_dir + "/top";
    std::filesystem::create_directories(out_top);

    BunnyProfile bunny_profile;
    bool has_bunny_profile = false;
    if (scene == "bunny" || scene == "both") {
        std::vector<Tri> tris;
        if (!load_obj_triangles(obj_path, tris)) {
            std::fprintf(stderr, "[Error] Failed to load OBJ: %s\n", obj_path.c_str());
            return 1;
        }
        normalize_bunny_mesh(tris);
        bunny_profile = build_bunny_profile(tris, 128);
        if (bunny_profile.volume <= 0.0f) {
            std::fprintf(stderr, "[Error] Bunny profile build failed: empty volume.\n");
            return 1;
        }
        has_bunny_profile = true;
        std::fprintf(stderr,
            "[Bunny] profile xy=%d local volume=%.6f local bbox z:[%.4f, %.4f]\n",
            bunny_profile.res_xy, bunny_profile.volume,
            bunny_profile.min_z, bunny_profile.max_z);
    }

    std::vector<Body> bodies;
    {
        uint32_t rng = 0xC0FFEE12u;
        const float identical_mass = 0.72f; // all spheres use same mass by design

        if (scene == "spheres" || scene == "both") {
            const std::array<float, 8> radii = {0.060f, 0.075f, 0.090f, 0.105f, 0.070f, 0.085f, 0.115f, 0.130f};
            for (float R : radii) {
                Body b;
                b.shape = SHAPE_SPHERE;
                b.mass = identical_mass;
                b.radius_eq = R;
                b.x = 0.8f + 2.4f * rand01(rng);
                b.y = 0.8f + 2.4f * rand01(rng);
                b.z = base_h + 0.20f + 0.35f * rand01(rng);
                b.vx = (rand01(rng) - 0.5f) * 0.25f;
                b.vy = (rand01(rng) - 0.5f) * 0.25f;
                b.vz = -0.2f * rand01(rng);
                b.yaw = 0.0f;
                b.yaw_rate = 0.0f;
                b.scale = 1.0f;
                bodies.push_back(b);
            }
        }

        if (scene == "bunny" || scene == "both") {
            Body b;
            b.shape = SHAPE_BUNNY;
            b.mass = identical_mass;
            b.scale = 0.55f;
            float v_scaled = bunny_profile.volume * b.scale * b.scale * b.scale;
            b.radius_eq = cbrtf((3.0f * v_scaled) / (4.0f * (float)M_PI));
            b.x = DOMAIN_X * 0.5f;
            b.y = DOMAIN_Y * 0.5f;
            b.z = base_h + 0.32f;
            b.vx = 0.12f;
            b.vy = -0.05f;
            b.vz = -0.1f;
            b.yaw = 0.0f;
            b.yaw_rate = 0.0f;
            bodies.push_back(b);
        }
    }

    if (bodies.empty()) {
        std::fprintf(stderr, "[Error] No bodies spawned.\n");
        return 1;
    }
    std::fprintf(stderr, "[Scene] %s, bodies=%zu, frames=%d\n", scene.c_str(), bodies.size(), frames);

    GLFWwindow* win = nullptr;
    if (glfwInit()) {
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
        win = glfwCreateWindow(560, 72, "Shallow Water (CUDA)", nullptr, nullptr);
    } else {
        std::fprintf(stderr, "[Warn] GLFW init failed. Running headless.\n");
    }

    CUDA_CHECK(cudaSetDevice(0));

    float *d_h_prev = nullptr, *d_h_cur = nullptr, *d_h_adv = nullptr;
    float *d_h_rhs = nullptr, *d_h_a = nullptr, *d_h_b = nullptr;
    float *d_u = nullptr, *d_v = nullptr;
    float *d_src = nullptr, *d_cw = nullptr, *d_ct = nullptr;

    CUDA_CHECK(cudaMalloc(&d_h_prev, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_h_cur,  N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_h_adv,  N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_h_rhs,  N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_h_a,    N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_h_b,    N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_u,      N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v,      N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_src,    N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cw,     N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ct,     N * sizeof(float)));

    {
        int threads = 256;
        int blocks = (N + threads - 1) / threads;
        init_fields_kernel<<<blocks, threads>>>(d_h_prev, d_h_cur, d_u, d_v, NX, NY, base_h);
        CUDA_CHECK(cudaGetLastError());
    }

    std::vector<float> h_host((size_t)N, base_h);
    std::vector<float> src_host((size_t)N, 0.0f);
    std::vector<float> cw_host((size_t)N, 0.0f);
    std::vector<float> ct_host((size_t)N, 0.0f);
    std::vector<float> objmask_host((size_t)N, 0.0f);
    std::vector<uchar4> img_top((size_t)W * H);

    dim3 b2(16, 16);
    dim3 g2((NX + b2.x - 1) / b2.x, (NY + b2.y - 1) / b2.y);

    for (int f = 0; f < frames; ++f) {
        if (win) {
            glfwPollEvents();
            if (glfwWindowShouldClose(win)) {
                std::fprintf(stderr, "[Info] Window closed at frame %d.\n", f);
                break;
            }
        }

        CUDA_CHECK(cudaMemcpy(h_host.data(), d_h_cur, N * sizeof(float), cudaMemcpyDeviceToHost));

        update_bodies_and_build_source(
            bodies, bunny_profile, has_bunny_profile,
            h_host, NX, NY, dx, dy, DOMAIN_X, DOMAIN_Y,
            dt, rho, g, src_host);

        build_coupling_maps(
            bodies, bunny_profile, has_bunny_profile,
            NX, NY, dx, dy, base_h,
            cw_host, ct_host, objmask_host);

        CUDA_CHECK(cudaMemcpy(d_src, src_host.data(), N * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_cw, cw_host.data(), N * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_ct, ct_host.data(), N * sizeof(float), cudaMemcpyHostToDevice));

        advect_height_kernel<<<g2, b2>>>(d_h_cur, d_u, d_v, d_h_adv, NX, NY, dx, dy, dt);
        CUDA_CHECK(cudaGetLastError());

        {
            int threads = 256;
            int blocks = (N + threads - 1) / threads;
            build_rhs_kernel<<<blocks, threads>>>(d_h_adv, d_h_prev, d_src, d_h_rhs, N, tau, dt);
            CUDA_CHECK(cudaGetLastError());
        }

        CUDA_CHECK(cudaMemcpy(d_h_a, d_h_cur, N * sizeof(float), cudaMemcpyDeviceToDevice));
        for (int it = 0; it < jacobi_iters; ++it) {
            jacobi_height_kernel<<<g2, b2>>>(
                d_h_rhs, d_h_a, d_cw, d_ct, d_h_b,
                NX, NY, dt2_g_over_dx2, min_h);
            CUDA_CHECK(cudaGetLastError());
            std::swap(d_h_a, d_h_b);
        }

        update_velocity_kernel<<<g2, b2>>>(d_u, d_v, d_h_a, NX, NY, dt, g, dx, dy, surface_friction_d);
        CUDA_CHECK(cudaGetLastError());
        damp_boundaries_kernel<<<g2, b2>>>(d_h_a, d_u, d_v, NX, NY, 10, 0.90f, base_h, 0.90f);
        CUDA_CHECK(cudaGetLastError());

        std::swap(d_h_prev, d_h_cur);
        std::swap(d_h_cur, d_h_a);
        CUDA_CHECK(cudaMemcpy(h_host.data(), d_h_cur, N * sizeof(float), cudaMemcpyDeviceToHost));

        render_topdown_cpu(
            h_host, bodies, bunny_profile, has_bunny_profile,
            NX, NY, dx, dy, W, H, base_h, img_top);

        char name_top[256];
        std::snprintf(name_top, sizeof(name_top), "%s/frame_%04d.png", out_top.c_str(), f);
        if (!stbi_write_png(name_top, W, H, 4, img_top.data(), W * 4)) {
            std::fprintf(stderr, "[Warn] failed to write %s\n", name_top);
        }

        if (win) {
            char title[256];
            std::snprintf(
                title, sizeof(title),
                "Shallow Water CUDA | frame %d/%d | bodies=%zu",
                f + 1, frames, bodies.size());
            glfwSetWindowTitle(win, title);
        }

        if ((f % 20) == 0) {
            const Body& b0 = bodies[0];
            std::fprintf(stderr, "[%4d/%4d] b0=(%.2f,%.2f,%.2f) v=(%.2f,%.2f,%.2f)\n",
                f + 1, frames, b0.x, b0.y, b0.z, b0.vx, b0.vy, b0.vz);
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaFree(d_h_prev));
    CUDA_CHECK(cudaFree(d_h_cur));
    CUDA_CHECK(cudaFree(d_h_adv));
    CUDA_CHECK(cudaFree(d_h_rhs));
    CUDA_CHECK(cudaFree(d_h_a));
    CUDA_CHECK(cudaFree(d_h_b));
    CUDA_CHECK(cudaFree(d_u));
    CUDA_CHECK(cudaFree(d_v));
    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaFree(d_cw));
    CUDA_CHECK(cudaFree(d_ct));

    if (win) glfwDestroyWindow(win);
    glfwTerminate();

    std::fprintf(stderr, "[Done] Top frames: %s\n", out_top.c_str());
    return 0;
}
