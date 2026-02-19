// Build (Debian + CUDA 12.9, RTX 4060 example):
//   nvcc -O3 -std=c++17 -arch=sm_89 chocolate_bunny_cuda.cu -o chocolate_bunny_cuda -lglfw -ldl -lpthread -lm
//
// Run:
//   ./chocolate_bunny_cuda stanford-bunny.obj 300 chocolate_frames
//
// Notes:
//   - This is a single-file CUDA implementation inspired by
//     "Melting and Flowing" (Carlson, Mucha, Van Horn, Turk).
//   - No CUDA-OpenGL interop is used.
//   - GLFW is used only for an optional progress window/events.
//   - Frames are saved as PNG via stb_image_write.
//   - 4 fixed views are exported: front/back/side/top.

#include <cuda_runtime.h>

#include <GLFW/glfw3.h>

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
#include <queue>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
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
} while (0)

struct Vec3 {
    float x, y, z;
    __host__ __device__ Vec3() : x(0), y(0), z(0) {}
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
__host__ __device__ inline float dot(const Vec3& a, const Vec3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
__host__ __device__ inline Vec3 cross(const Vec3& a, const Vec3& b) {
    return Vec3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}
__host__ __device__ inline float length(const Vec3& v) {
    return sqrtf(dot(v, v));
}
__host__ __device__ inline Vec3 normalize(const Vec3& v) {
    float len = length(v);
    return (len > 0.0f) ? (v / len) : Vec3(0, 0, 0);
}
__host__ __device__ inline float clamp01(float x) {
    return fminf(fmaxf(x, 0.0f), 1.0f);
}
__host__ __device__ inline float lerpf(float a, float b, float t) {
    return a + (b - a) * t;
}
__host__ __device__ inline Vec3 lerp3(const Vec3& a, const Vec3& b, float t) {
    return Vec3(lerpf(a.x, b.x, t), lerpf(a.y, b.y, t), lerpf(a.z, b.z, t));
}

struct Tri {
    Vec3 a, b, c;
};

struct TriYZ {
    Vec3 a, b, c;
    float min_y, max_y;
    float min_z, max_z;
    float det;
    bool valid;
};

static inline uint32_t xorshift32(uint32_t& s) {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    return s;
}

static inline float rand01(uint32_t& s) {
    return (xorshift32(s) & 0x00FFFFFFu) * (1.0f / 16777216.0f);
}

static inline int idx3_h(int x, int y, int z, int nx, int ny, int nz) {
    (void)nz;
    return (z * ny + y) * nx + x;
}

static std::string trim(const std::string& s) {
    size_t b = 0;
    while (b < s.size() && std::isspace((unsigned char)s[b])) b++;
    size_t e = s.size();
    while (e > b && std::isspace((unsigned char)s[e - 1])) e--;
    return s.substr(b, e - b);
}

static int parse_obj_index(const std::string& tok, int vcount) {
    size_t slash = tok.find('/');
    std::string id = (slash == std::string::npos) ? tok : tok.substr(0, slash);
    int idx = std::stoi(id);
    if (idx < 0) idx = vcount + idx + 1;
    return idx - 1;
}

static bool load_obj_triangles(const std::string& path, std::vector<Tri>& tris) {
    std::ifstream in(path);
    if (!in) return false;

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
            verts.emplace_back(x, y, z);
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

    if (tris.empty()) return false;

    Vec3 bmin(1e30f, 1e30f, 1e30f), bmax(-1e30f, -1e30f, -1e30f);
    for (const auto& t : tris) {
        bmin.x = std::min(bmin.x, std::min(t.a.x, std::min(t.b.x, t.c.x)));
        bmin.y = std::min(bmin.y, std::min(t.a.y, std::min(t.b.y, t.c.y)));
        bmin.z = std::min(bmin.z, std::min(t.a.z, std::min(t.b.z, t.c.z)));
        bmax.x = std::max(bmax.x, std::max(t.a.x, std::max(t.b.x, t.c.x)));
        bmax.y = std::max(bmax.y, std::max(t.a.y, std::max(t.b.y, t.c.y)));
        bmax.z = std::max(bmax.z, std::max(t.a.z, std::max(t.b.z, t.c.z)));
    }

    Vec3 ext = bmax - bmin;
    float max_ext = std::max(ext.x, std::max(ext.y, ext.z));
    float s = 0.56f / std::max(max_ext, 1e-6f);

    Vec3 scaled_ext = ext * s;
    Vec3 off(0.5f - 0.5f * scaled_ext.x, 0.12f, 0.5f - 0.5f * scaled_ext.z);

    for (auto& t : tris) {
        t.a = (t.a - bmin) * s + off;
        t.b = (t.b - bmin) * s + off;
        t.c = (t.c - bmin) * s + off;
    }

    return true;
}

static std::vector<TriYZ> build_triyz(const std::vector<Tri>& tris) {
    std::vector<TriYZ> out;
    out.reserve(tris.size());
    for (const auto& t : tris) {
        TriYZ q{};
        q.a = t.a; q.b = t.b; q.c = t.c;
        q.min_y = std::min(t.a.y, std::min(t.b.y, t.c.y));
        q.max_y = std::max(t.a.y, std::max(t.b.y, t.c.y));
        q.min_z = std::min(t.a.z, std::min(t.b.z, t.c.z));
        q.max_z = std::max(t.a.z, std::max(t.b.z, t.c.z));

        float m00 = t.b.y - t.a.y;
        float m01 = t.c.y - t.a.y;
        float m10 = t.b.z - t.a.z;
        float m11 = t.c.z - t.a.z;
        q.det = m00 * m11 - m01 * m10;
        q.valid = std::fabs(q.det) > 1e-10f;
        out.push_back(q);
    }
    return out;
}

static void voxelize_bunny_fill(
    int nx, int ny, int nz,
    const std::vector<TriYZ>& tri_yz,
    std::vector<int>& fluid,
    std::vector<int>& solid)
{
    const int N = nx * ny * nz;
    fluid.assign(N, 0);

    // Solid walls on all 6 boundaries.
    for (int z = 0; z < nz; ++z) {
        for (int y = 0; y < ny; ++y) {
            for (int x = 0; x < nx; ++x) {
                int id = idx3_h(x, y, z, nx, ny, nz);
                if (x == 0 || x == nx - 1 || y == 0 || y == ny - 1 || z == 0 || z == nz - 1) {
                    solid[id] = 1;
                }
            }
        }
    }

    std::vector<float> xs;
    xs.reserve(512);

    for (int z = 1; z < nz - 1; ++z) {
        float pz = (z + 0.5f) / (float)nz;
        for (int y = 1; y < ny - 1; ++y) {
            float py = (y + 0.5f) / (float)ny;
            xs.clear();

            for (const auto& t : tri_yz) {
                if (!t.valid) continue;
                if (py < t.min_y - 1e-6f || py > t.max_y + 1e-6f) continue;
                if (pz < t.min_z - 1e-6f || pz > t.max_z + 1e-6f) continue;

                float ay = t.a.y, az = t.a.z;
                float by = t.b.y, bz = t.b.z;
                float cy = t.c.y, cz = t.c.z;

                float u = ((py - ay) * (cz - az) - (pz - az) * (cy - ay)) / t.det;
                float v = ((by - ay) * (pz - az) - (bz - az) * (py - ay)) / t.det;
                float w = 1.0f - u - v;

                if (u >= -1e-6f && v >= -1e-6f && w >= -1e-6f) {
                    float px = w * t.a.x + u * t.b.x + v * t.c.x;
                    if (px >= 0.0f && px <= 1.0f) xs.push_back(px);
                }
            }

            if (xs.size() < 2) continue;

            std::sort(xs.begin(), xs.end());
            std::vector<float> uniq;
            uniq.reserve(xs.size());
            for (float x : xs) {
                if (uniq.empty() || std::fabs(x - uniq.back()) > 1e-5f) {
                    uniq.push_back(x);
                }
            }

            for (size_t p = 0; p + 1 < uniq.size(); p += 2) {
                float x0 = std::min(uniq[p], uniq[p + 1]);
                float x1 = std::max(uniq[p], uniq[p + 1]);
                if (x1 <= x0) continue;

                int i0 = std::max(1, (int)std::floor(x0 * nx));
                int i1 = std::min(nx - 2, (int)std::floor(x1 * nx));
                for (int x = i0; x <= i1; ++x) {
                    float px = (x + 0.5f) / (float)nx;
                    if (px >= x0 && px <= x1) {
                        int id = idx3_h(x, y, z, nx, ny, nz);
                        if (!solid[id]) fluid[id] = 1;
                    }
                }
            }
        }
    }

    // Remove tiny isolated islands from scan conversion noise.
    std::vector<int> visited(N, 0);
    std::queue<int> q;
    const int dx[6] = {1, -1, 0, 0, 0, 0};
    const int dy[6] = {0, 0, 1, -1, 0, 0};
    const int dz[6] = {0, 0, 0, 0, 1, -1};

    for (int z = 1; z < nz - 1; ++z) {
        for (int y = 1; y < ny - 1; ++y) {
            for (int x = 1; x < nx - 1; ++x) {
                int s = idx3_h(x, y, z, nx, ny, nz);
                if (!fluid[s] || visited[s]) continue;

                std::vector<int> comp;
                comp.reserve(2048);
                visited[s] = 1;
                q.push(s);

                while (!q.empty()) {
                    int id = q.front(); q.pop();
                    comp.push_back(id);

                    int iz = id / (nx * ny);
                    int rem = id - iz * nx * ny;
                    int iy = rem / nx;
                    int ix = rem - iy * nx;

                    for (int k = 0; k < 6; ++k) {
                        int nxp = ix + dx[k];
                        int nyp = iy + dy[k];
                        int nzp = iz + dz[k];
                        int nid = idx3_h(nxp, nyp, nzp, nx, ny, nz);
                        if (!fluid[nid] || visited[nid]) continue;
                        visited[nid] = 1;
                        q.push(nid);
                    }
                }

                if ((int)comp.size() < 8) {
                    for (int id : comp) fluid[id] = 0;
                }
            }
        }
    }
}

static void seed_particles_from_fluid(
    int nx, int ny, int nz,
    const std::vector<int>& fluid,
    const std::vector<int>& solid,
    std::vector<Vec3>& particles)
{
    particles.clear();
    particles.reserve(350000);

    uint32_t rng = 0x12345678u;
    const int dx[6] = {1, -1, 0, 0, 0, 0};
    const int dy[6] = {0, 0, 1, -1, 0, 0};
    const int dz[6] = {0, 0, 0, 0, 1, -1};

    for (int z = 1; z < nz - 1; ++z) {
        for (int y = 1; y < ny - 1; ++y) {
            for (int x = 1; x < nx - 1; ++x) {
                int id = idx3_h(x, y, z, nx, ny, nz);
                if (!fluid[id] || solid[id]) continue;

                bool surf = false;
                for (int k = 0; k < 6; ++k) {
                    int nid = idx3_h(x + dx[k], y + dy[k], z + dz[k], nx, ny, nz);
                    if (!fluid[nid]) {
                        surf = true;
                        break;
                    }
                }

                int count = surf ? 5 : 2;
                for (int n = 0; n < count; ++n) {
                    float jx = rand01(rng);
                    float jy = rand01(rng);
                    float jz = rand01(rng);
                    Vec3 p((x + jx) / (float)nx, (y + jy) / (float)ny, (z + jz) / (float)nz);
                    particles.push_back(p);
                }
            }
        }
    }
}

__host__ __device__ inline int idx3_d(int x, int y, int z, int nx, int ny, int nz) {
    (void)nz;
    return (z * ny + y) * nx + x;
}

__device__ inline float read_scalar(const float* f, int x, int y, int z, int nx, int ny, int nz) {
    x = max(0, min(nx - 1, x));
    y = max(0, min(ny - 1, y));
    z = max(0, min(nz - 1, z));
    return f[idx3_d(x, y, z, nx, ny, nz)];
}

__device__ inline float sample_scalar_centered(const float* f, float x, float y, float z, int nx, int ny, int nz) {
    x = fminf(fmaxf(x, 0.0f), 1.0f - 1e-6f);
    y = fminf(fmaxf(y, 0.0f), 1.0f - 1e-6f);
    z = fminf(fmaxf(z, 0.0f), 1.0f - 1e-6f);

    float gx = x * nx - 0.5f;
    float gy = y * ny - 0.5f;
    float gz = z * nz - 0.5f;

    int x0 = (int)floorf(gx), y0 = (int)floorf(gy), z0 = (int)floorf(gz);
    int x1 = x0 + 1, y1 = y0 + 1, z1 = z0 + 1;

    float tx = gx - x0;
    float ty = gy - y0;
    float tz = gz - z0;

    float c000 = read_scalar(f, x0, y0, z0, nx, ny, nz);
    float c100 = read_scalar(f, x1, y0, z0, nx, ny, nz);
    float c010 = read_scalar(f, x0, y1, z0, nx, ny, nz);
    float c110 = read_scalar(f, x1, y1, z0, nx, ny, nz);
    float c001 = read_scalar(f, x0, y0, z1, nx, ny, nz);
    float c101 = read_scalar(f, x1, y0, z1, nx, ny, nz);
    float c011 = read_scalar(f, x0, y1, z1, nx, ny, nz);
    float c111 = read_scalar(f, x1, y1, z1, nx, ny, nz);

    float c00 = c000 + (c100 - c000) * tx;
    float c10 = c010 + (c110 - c010) * tx;
    float c01 = c001 + (c101 - c001) * tx;
    float c11 = c011 + (c111 - c011) * tx;

    float c0 = c00 + (c10 - c00) * ty;
    float c1 = c01 + (c11 - c01) * ty;

    return c0 + (c1 - c0) * tz;
}

__device__ inline Vec3 sample_velocity_centered(
    const float* u, const float* v, const float* w,
    float x, float y, float z,
    int nx, int ny, int nz)
{
    return Vec3(
        sample_scalar_centered(u, x, y, z, nx, ny, nz),
        sample_scalar_centered(v, x, y, z, nx, ny, nz),
        sample_scalar_centered(w, x, y, z, nx, ny, nz)
    );
}

__global__ void clear_int_kernel(int* a, int n, int value) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] = value;
}

__global__ void clear_float_kernel(float* a, int n, float value) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] = value;
}

__global__ void mark_fluid_from_particles(
    const Vec3* p, int count,
    int* fluid,
    int nx, int ny, int nz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    Vec3 q = p[i];
    int ix = min(nx - 1, max(0, (int)floorf(q.x * nx)));
    int iy = min(ny - 1, max(0, (int)floorf(q.y * ny)));
    int iz = min(nz - 1, max(0, (int)floorf(q.z * nz)));

    int id = idx3_d(ix, iy, iz, nx, ny, nz);
    atomicExch(&fluid[id], 1);
}

__global__ void enforce_solid_mask(int* fluid, const int* solid, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && solid[i]) fluid[i] = 0;
}

__global__ void advect_velocity_semi_lagrangian(
    const float* u, const float* v, const float* w,
    float* out_u, float* out_v, float* out_w,
    const int* fluid,
    const int* solid,
    float dt,
    int nx, int ny, int nz)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int N = nx * ny * nz;
    if (id >= N) return;

    if (!fluid[id] || solid[id]) {
        out_u[id] = 0.0f;
        out_v[id] = 0.0f;
        out_w[id] = 0.0f;
        return;
    }

    int z = id / (nx * ny);
    int rem = id - z * nx * ny;
    int y = rem / nx;
    int x = rem - y * nx;

    float px = (x + 0.5f) / (float)nx;
    float py = (y + 0.5f) / (float)ny;
    float pz = (z + 0.5f) / (float)nz;

    Vec3 vel = sample_velocity_centered(u, v, w, px, py, pz, nx, ny, nz);
    Vec3 prev(px - dt * vel.x, py - dt * vel.y, pz - dt * vel.z);

    out_u[id] = sample_scalar_centered(u, prev.x, prev.y, prev.z, nx, ny, nz);
    out_v[id] = sample_scalar_centered(v, prev.x, prev.y, prev.z, nx, ny, nz);
    out_w[id] = sample_scalar_centered(w, prev.x, prev.y, prev.z, nx, ny, nz);
}

__global__ void add_gravity(float* v, const int* fluid, const int* solid, float dt, float g, int n) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n && fluid[id] && !solid[id]) {
        v[id] += dt * g;
    }
}

__global__ void temp_advect(
    const float* temp,
    const float* u, const float* v, const float* w,
    float* temp_out,
    const int* fluid,
    float ambient,
    float dt,
    int nx, int ny, int nz)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int N = nx * ny * nz;
    if (id >= N) return;

    if (!fluid[id]) {
        temp_out[id] = ambient;
        return;
    }

    int z = id / (nx * ny);
    int rem = id - z * nx * ny;
    int y = rem / nx;
    int x = rem - y * nx;

    float px = (x + 0.5f) / (float)nx;
    float py = (y + 0.5f) / (float)ny;
    float pz = (z + 0.5f) / (float)nz;

    Vec3 vel = sample_velocity_centered(u, v, w, px, py, pz, nx, ny, nz);
    Vec3 prev(px - dt * vel.x, py - dt * vel.y, pz - dt * vel.z);

    temp_out[id] = sample_scalar_centered(temp, prev.x, prev.y, prev.z, nx, ny, nz);
}

__global__ void temp_apply_sources(
    float* temp,
    const int* fluid,
    int nx, int ny, int nz,
    float dt,
    float ambient,
    float cool_rate,
    Vec3 src_pos,
    float src_radius,
    float src_temp,
    float src_gain)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int N = nx * ny * nz;
    if (id >= N) return;

    if (!fluid[id]) {
        temp[id] = ambient;
        return;
    }

    int z = id / (nx * ny);
    int rem = id - z * nx * ny;
    int y = rem / nx;
    int x = rem - y * nx;

    float px = (x + 0.5f) / (float)nx;
    float py = (y + 0.5f) / (float)ny;
    float pz = (z + 0.5f) / (float)nz;

    float T = temp[id];
    T += dt * cool_rate * (ambient - T);

    Vec3 d(px - src_pos.x, py - src_pos.y, pz - src_pos.z);
    float r = length(d);
    if (r < src_radius) {
        float w = 1.0f - r / src_radius;
        T += dt * src_gain * w * (src_temp - T);
    }

    temp[id] = T;
}

__global__ void temp_diffuse_jacobi(
    const float* rhs,
    const float* in,
    float* out,
    const int* fluid,
    int nx, int ny, int nz,
    float dt,
    float kappa,
    float h2inv)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int N = nx * ny * nz;
    if (id >= N) return;

    if (!fluid[id]) {
        out[id] = rhs[id];
        return;
    }

    int z = id / (nx * ny);
    int rem = id - z * nx * ny;
    int y = rem / nx;
    int x = rem - y * nx;

    float coef = dt * kappa * h2inv;

    float num = rhs[id];
    float den = 1.0f;

    const int ox[6] = {1, -1, 0, 0, 0, 0};
    const int oy[6] = {0, 0, 1, -1, 0, 0};
    const int oz[6] = {0, 0, 0, 0, 1, -1};

    for (int k = 0; k < 6; ++k) {
        int nxp = x + ox[k];
        int nyp = y + oy[k];
        int nzp = z + oz[k];
        if (nxp < 0 || nxp >= nx || nyp < 0 || nyp >= ny || nzp < 0 || nzp >= nz) continue;
        int nid = idx3_d(nxp, nyp, nzp, nx, ny, nz);
        if (fluid[nid]) {
            num += coef * in[nid];
            den += coef;
        }
    }

    out[id] = num / den;
}

__global__ void viscosity_from_temperature(
    const float* temp,
    float* visc,
    const int* fluid,
    int n,
    float melt_center,
    float melt_half,
    float visc_cold,
    float visc_hot)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    if (!fluid[i]) {
        visc[i] = visc_hot;
        return;
    }

    float T = temp[i];
    float t0 = melt_center - melt_half;
    float t1 = melt_center + melt_half;
    float s = (T - t0) / fmaxf(t1 - t0, 1e-6f);
    s = clamp01(s);
    // Quadratic/smooth transition.
    s = s * s * (3.0f - 2.0f * s);

    float ln_cold = logf(fmaxf(visc_cold, 1e-8f));
    float ln_hot = logf(fmaxf(visc_hot, 1e-8f));
    visc[i] = expf(lerpf(ln_cold, ln_hot, s));
}

__global__ void diffuse_velocity_jacobi(
    const float* rhs,
    const float* in,
    float* out,
    const float* visc,
    const int* fluid,
    const int* solid,
    int nx, int ny, int nz,
    float dt,
    float h2inv)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int N = nx * ny * nz;
    if (id >= N) return;

    if (!fluid[id] || solid[id]) {
        out[id] = 0.0f;
        return;
    }

    int z = id / (nx * ny);
    int rem = id - z * nx * ny;
    int y = rem / nx;
    int x = rem - y * nx;

    float num = rhs[id];
    float den = 1.0f;

    const int ox[6] = {1, -1, 0, 0, 0, 0};
    const int oy[6] = {0, 0, 1, -1, 0, 0};
    const int oz[6] = {0, 0, 0, 0, 1, -1};

    float nu0 = fmaxf(visc[id], 1e-8f);

    for (int k = 0; k < 6; ++k) {
        int nxp = x + ox[k];
        int nyp = y + oy[k];
        int nzp = z + oz[k];

        if (nxp < 0 || nxp >= nx || nyp < 0 || nyp >= ny || nzp < 0 || nzp >= nz) {
            float c = dt * nu0 * h2inv;
            den += c;
            continue;
        }

        int nid = idx3_d(nxp, nyp, nzp, nx, ny, nz);

        if (solid[nid]) {
            // No-slip wall.
            float c = dt * nu0 * h2inv;
            den += c;
            continue;
        }

        if (fluid[nid]) {
            float nu1 = fmaxf(visc[nid], 1e-8f);
            float face_nu = sqrtf(nu0 * nu1); // Geometric mean from paper discussion.
            float c = dt * face_nu * h2inv;
            num += c * in[nid];
            den += c;
        } else {
            // Air neighbor: skip (free-surface style decoupling for diffusion).
        }
    }

    out[id] = num / den;
}

__global__ void compute_divergence(
    const float* u, const float* v, const float* w,
    float* div,
    const int* fluid,
    const int* solid,
    int nx, int ny, int nz,
    float inv2h)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int N = nx * ny * nz;
    if (id >= N) return;

    if (!fluid[id] || solid[id]) {
        div[id] = 0.0f;
        return;
    }

    int z = id / (nx * ny);
    int rem = id - z * nx * ny;
    int y = rem / nx;
    int x = rem - y * nx;

    auto sample_vel = [&](const float* f, int sx, int sy, int sz) {
        if (sx < 0 || sx >= nx || sy < 0 || sy >= ny || sz < 0 || sz >= nz) return 0.0f;
        int nid = idx3_d(sx, sy, sz, nx, ny, nz);
        if (solid[nid]) return 0.0f;
        if (!fluid[nid]) return 0.0f;
        return f[nid];
    };

    float ux1 = sample_vel(u, x + 1, y, z);
    float ux0 = sample_vel(u, x - 1, y, z);
    float vy1 = sample_vel(v, x, y + 1, z);
    float vy0 = sample_vel(v, x, y - 1, z);
    float wz1 = sample_vel(w, x, y, z + 1);
    float wz0 = sample_vel(w, x, y, z - 1);

    div[id] = (ux1 - ux0 + vy1 - vy0 + wz1 - wz0) * inv2h;
}

__global__ void pressure_jacobi(
    const float* rhs,
    const float* in,
    float* out,
    const int* fluid,
    const int* solid,
    int nx, int ny, int nz,
    float h2)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int N = nx * ny * nz;
    if (id >= N) return;

    if (!fluid[id] || solid[id]) {
        out[id] = 0.0f;
        return;
    }

    int z = id / (nx * ny);
    int rem = id - z * nx * ny;
    int y = rem / nx;
    int x = rem - y * nx;

    float sum = 0.0f;
    float cnt = 0.0f;

    const int ox[6] = {1, -1, 0, 0, 0, 0};
    const int oy[6] = {0, 0, 1, -1, 0, 0};
    const int oz[6] = {0, 0, 0, 0, 1, -1};

    for (int k = 0; k < 6; ++k) {
        int nxp = x + ox[k];
        int nyp = y + oy[k];
        int nzp = z + oz[k];

        if (nxp < 0 || nxp >= nx || nyp < 0 || nyp >= ny || nzp < 0 || nzp >= nz) {
            sum += in[id];
            cnt += 1.0f;
            continue;
        }

        int nid = idx3_d(nxp, nyp, nzp, nx, ny, nz);
        if (solid[nid]) {
            sum += in[id]; // Neumann wall.
            cnt += 1.0f;
        } else if (fluid[nid]) {
            sum += in[nid];
            cnt += 1.0f;
        } else {
            sum += 0.0f; // Air pressure = 0.
            cnt += 1.0f;
        }
    }

    out[id] = (sum - rhs[id] * h2) / fmaxf(cnt, 1.0f);
}

__global__ void project_subtract_gradient(
    float* u, float* v, float* w,
    const float* p,
    const int* fluid,
    const int* solid,
    int nx, int ny, int nz,
    float dt,
    float inv2h)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int N = nx * ny * nz;
    if (id >= N) return;

    if (!fluid[id] || solid[id]) {
        u[id] = v[id] = w[id] = 0.0f;
        return;
    }

    int z = id / (nx * ny);
    int rem = id - z * nx * ny;
    int y = rem / nx;
    int x = rem - y * nx;

    auto sample_p = [&](int sx, int sy, int sz) {
        if (sx < 0 || sx >= nx || sy < 0 || sy >= ny || sz < 0 || sz >= nz) return p[id];
        int nid = idx3_d(sx, sy, sz, nx, ny, nz);
        if (solid[nid]) return p[id];
        if (!fluid[nid]) return 0.0f;
        return p[nid];
    };

    float px1 = sample_p(x + 1, y, z);
    float px0 = sample_p(x - 1, y, z);
    float py1 = sample_p(x, y + 1, z);
    float py0 = sample_p(x, y - 1, z);
    float pz1 = sample_p(x, y, z + 1);
    float pz0 = sample_p(x, y, z - 1);

    u[id] -= dt * (px1 - px0) * inv2h;
    v[id] -= dt * (py1 - py0) * inv2h;
    w[id] -= dt * (pz1 - pz0) * inv2h;
}

__global__ void advect_particles(
    Vec3* p,
    int count,
    const float* u,
    const float* v,
    const float* w,
    float dt,
    int nx, int ny, int nz,
    float floor_y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    Vec3 q = p[i];
    Vec3 vel = sample_velocity_centered(u, v, w, q.x, q.y, q.z, nx, ny, nz);
    q += vel * dt;

    const float eps = 1e-4f;
    if (q.x < eps) q.x = eps;
    if (q.x > 1.0f - eps) q.x = 1.0f - eps;
    if (q.y < floor_y + eps) q.y = floor_y + eps;
    if (q.y > 1.0f - eps) q.y = 1.0f - eps;
    if (q.z < eps) q.z = eps;
    if (q.z > 1.0f - eps) q.z = 1.0f - eps;

    p[i] = q;
}

__global__ void splat_particles_volume(
    const Vec3* p,
    int count,
    float* vol,
    int vx, int vy, int vz,
    float radius_vox,
    float gain)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    Vec3 q = p[i];
    float gx = q.x * vx - 0.5f;
    float gy = q.y * vy - 0.5f;
    float gz = q.z * vz - 0.5f;

    int rx = (int)ceilf(radius_vox);
    int ix = (int)floorf(gx);
    int iy = (int)floorf(gy);
    int iz = (int)floorf(gz);

    for (int dz = -rx; dz <= rx; ++dz) {
        int z = iz + dz;
        if (z < 0 || z >= vz) continue;
        float wz = fmaxf(0.0f, 1.0f - fabsf(z - gz) / radius_vox);
        if (wz <= 0.0f) continue;
        for (int dy = -rx; dy <= rx; ++dy) {
            int y = iy + dy;
            if (y < 0 || y >= vy) continue;
            float wy = fmaxf(0.0f, 1.0f - fabsf(y - gy) / radius_vox);
            if (wy <= 0.0f) continue;
            for (int dx = -rx; dx <= rx; ++dx) {
                int x = ix + dx;
                if (x < 0 || x >= vx) continue;
                float wx = fmaxf(0.0f, 1.0f - fabsf(x - gx) / radius_vox);
                if (wx <= 0.0f) continue;
                float wsum = wx * wy * wz * gain;
                atomicAdd(&vol[(z * vy + y) * vx + x], wsum);
            }
        }
    }
}

__global__ void clamp_volume(float* vol, int n, float vmax) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) vol[i] = fminf(vol[i], vmax);
}

__global__ void blur_volume_6n(const float* in, float* out, int vx, int vy, int vz) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int N = vx * vy * vz;
    if (id >= N) return;

    int z = id / (vx * vy);
    int rem = id - z * vx * vy;
    int y = rem / vx;
    int x = rem - y * vx;

    auto at = [&](int sx, int sy, int sz) {
        sx = max(0, min(vx - 1, sx));
        sy = max(0, min(vy - 1, sy));
        sz = max(0, min(vz - 1, sz));
        return in[(sz * vy + sy) * vx + sx];
    };

    float c = at(x, y, z);
    float s = at(x + 1, y, z) + at(x - 1, y, z)
            + at(x, y + 1, z) + at(x, y - 1, z)
            + at(x, y, z + 1) + at(x, y, z - 1);

    out[id] = c * 0.40f + s * 0.10f;
}

__device__ inline float sample_volume(const float* vol, float x, float y, float z, int vx, int vy, int vz) {
    x = fminf(fmaxf(x, 0.0f), 1.0f - 1e-6f);
    y = fminf(fmaxf(y, 0.0f), 1.0f - 1e-6f);
    z = fminf(fmaxf(z, 0.0f), 1.0f - 1e-6f);

    float gx = x * vx - 0.5f;
    float gy = y * vy - 0.5f;
    float gz = z * vz - 0.5f;

    int x0 = (int)floorf(gx), y0 = (int)floorf(gy), z0 = (int)floorf(gz);
    int x1 = x0 + 1, y1 = y0 + 1, z1 = z0 + 1;

    float tx = gx - x0;
    float ty = gy - y0;
    float tz = gz - z0;

    auto at = [&](int sx, int sy, int sz) {
        sx = max(0, min(vx - 1, sx));
        sy = max(0, min(vy - 1, sy));
        sz = max(0, min(vz - 1, sz));
        return vol[(sz * vy + sy) * vx + sx];
    };

    float c000 = at(x0, y0, z0);
    float c100 = at(x1, y0, z0);
    float c010 = at(x0, y1, z0);
    float c110 = at(x1, y1, z0);
    float c001 = at(x0, y0, z1);
    float c101 = at(x1, y0, z1);
    float c011 = at(x0, y1, z1);
    float c111 = at(x1, y1, z1);

    float c00 = c000 + (c100 - c000) * tx;
    float c10 = c010 + (c110 - c010) * tx;
    float c01 = c001 + (c101 - c001) * tx;
    float c11 = c011 + (c111 - c011) * tx;

    float c0 = c00 + (c10 - c00) * ty;
    float c1 = c01 + (c11 - c01) * ty;
    return c0 + (c1 - c0) * tz;
}

struct RenderCam {
    Vec3 pos;
    Vec3 fwd;
    Vec3 right;
    Vec3 up;
    float tan_half_fov;
    float aspect;
};

__global__ void raymarch_render(
    uchar4* out,
    int W, int H,
    RenderCam cam,
    const float* vol,
    int vx, int vy, int vz,
    const float* temp,
    int nx, int ny, int nz,
    float melt_center,
    float melt_half,
    float floor_y)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    float sx = ((x + 0.5f) / (float)W) * 2.0f - 1.0f;
    float sy = 1.0f - ((y + 0.5f) / (float)H) * 2.0f;

    Vec3 dir = normalize(cam.fwd + cam.right * (sx * cam.aspect * cam.tan_half_fov) + cam.up * (sy * cam.tan_half_fov));

    Vec3 skyTop(0.92f, 0.96f, 1.00f);
    Vec3 skyBot(0.74f, 0.82f, 0.92f);
    Vec3 col = lerp3(skyBot, skyTop, clamp01(0.5f * (dir.y + 1.0f)));

    float t = 0.0f;
    float t_max = 2.5f;
    float step = 0.0032f;
    float iso = 0.34f;

    bool hit = false;
    Vec3 hp(0, 0, 0);

    for (int i = 0; i < 900 && t < t_max; ++i, t += step) {
        Vec3 p = cam.pos + dir * t;
        if (p.x < 0.0f || p.x > 1.0f || p.y < 0.0f || p.y > 1.0f || p.z < 0.0f || p.z > 1.0f) {
            continue;
        }
        float d = sample_volume(vol, p.x, p.y, p.z, vx, vy, vz);
        if (d >= iso) {
            hit = true;

            float a = t - step;
            float b = t;
            for (int r = 0; r < 6; ++r) {
                float m = 0.5f * (a + b);
                Vec3 mp = cam.pos + dir * m;
                float md = sample_volume(vol, mp.x, mp.y, mp.z, vx, vy, vz);
                if (md >= iso) b = m;
                else a = m;
            }
            t = b;
            hp = cam.pos + dir * t;
            break;
        }
    }

    if (hit) {
        float eps = 1.0f / fmaxf((float)vx, fmaxf((float)vy, (float)vz));
        float dx = sample_volume(vol, hp.x + eps, hp.y, hp.z, vx, vy, vz) - sample_volume(vol, hp.x - eps, hp.y, hp.z, vx, vy, vz);
        float dy = sample_volume(vol, hp.x, hp.y + eps, hp.z, vx, vy, vz) - sample_volume(vol, hp.x, hp.y - eps, hp.z, vx, vy, vz);
        float dz = sample_volume(vol, hp.x, hp.y, hp.z + eps, vx, vy, vz) - sample_volume(vol, hp.x, hp.y, hp.z - eps, vx, vy, vz);

        Vec3 n = normalize(Vec3(dx, dy, dz));
        if (dot(n, dir) > 0.0f) n = n * -1.0f;

        float T = sample_scalar_centered(temp, hp.x, hp.y, hp.z, nx, ny, nz);
        float m = clamp01((T - (melt_center - melt_half)) / fmaxf(2.0f * melt_half, 1e-6f));
        m = m * m * (3.0f - 2.0f * m);

        Vec3 light = normalize(Vec3(0.52f, 1.25f, 0.72f));
        float diff = fmaxf(dot(n, light), 0.0f);
        Vec3 h = normalize(light - dir);
        float gloss = 0.18f + 0.72f * m;
        float spec_pow = lerpf(18.0f, 78.0f, gloss);
        float spec = powf(fmaxf(dot(n, h), 0.0f), spec_pow) * (0.18f + 0.62f * m);

        Vec3 cold(0.25f, 0.14f, 0.08f);
        Vec3 hot(0.40f, 0.23f, 0.13f);
        Vec3 base = lerp3(cold, hot, m);

        float fres = powf(1.0f - fmaxf(dot(n, -1.0f * dir), 0.0f), 5.0f);
        Vec3 edge_tint(0.24f, 0.16f, 0.10f);
        col = base * (0.16f + 0.84f * diff) + Vec3(spec, spec, spec) + edge_tint * (0.25f * fres);
    } else {
        // Ground plane shading.
        if (dir.y < -1e-4f) {
            float tg = (floor_y - cam.pos.y) / dir.y;
            if (tg > 0.0f && tg < t_max) {
                Vec3 gp = cam.pos + dir * tg;
                if (gp.x >= 0.0f && gp.x <= 1.0f && gp.z >= 0.0f && gp.z <= 1.0f) {
                    float chk = (fmodf(floorf(gp.x * 26.0f) + floorf(gp.z * 26.0f), 2.0f) < 1.0f) ? 0.0f : 1.0f;
                    Vec3 g0(0.69f, 0.63f, 0.58f);
                    Vec3 g1(0.62f, 0.56f, 0.51f);
                    Vec3 gcol = lerp3(g0, g1, chk * 0.45f);
                    float fog = expf(-0.35f * tg);
                    col = lerp3(gcol, col, clamp01(1.0f - fog));
                }
            }
        }
    }

    col.x = powf(clamp01(col.x), 1.0f / 2.2f);
    col.y = powf(clamp01(col.y), 1.0f / 2.2f);
    col.z = powf(clamp01(col.z), 1.0f / 2.2f);

    out[y * W + x] = make_uchar4(
        (unsigned char)(clamp01(col.x) * 255.0f + 0.5f),
        (unsigned char)(clamp01(col.y) * 255.0f + 0.5f),
        (unsigned char)(clamp01(col.z) * 255.0f + 0.5f),
        255
    );
}

static int reinject_free_flight_bulk_velocity(
    int nx, int ny, int nz,
    const std::vector<int>& fluid,
    const std::vector<int>& solid,
    const std::vector<float>& u_before,
    const std::vector<float>& v_before,
    const std::vector<float>& w_before,
    std::vector<float>& u_after,
    std::vector<float>& v_after,
    std::vector<float>& w_after)
{
    const int N = nx * ny * nz;
    std::vector<uint8_t> vis((size_t)N, 0);
    std::queue<int> q;

    const int ox[6] = {1, -1, 0, 0, 0, 0};
    const int oy[6] = {0, 0, 1, -1, 0, 0};
    const int oz[6] = {0, 0, 0, 0, 1, -1};

    int corrected = 0;

    for (int z = 1; z < nz - 1; ++z) {
        for (int y = 1; y < ny - 1; ++y) {
            for (int x = 1; x < nx - 1; ++x) {
                int s = idx3_h(x, y, z, nx, ny, nz);
                if (!fluid[s] || solid[s] || vis[s]) continue;

                std::vector<int> comp;
                comp.reserve(512);
                bool touches_obstacle = false;

                vis[s] = 1;
                q.push(s);

                while (!q.empty()) {
                    int id = q.front(); q.pop();
                    comp.push_back(id);

                    int cz = id / (nx * ny);
                    int rem = id - cz * nx * ny;
                    int cy = rem / nx;
                    int cx = rem - cy * nx;

                    for (int k = 0; k < 6; ++k) {
                        int nxp = cx + ox[k];
                        int nyp = cy + oy[k];
                        int nzp = cz + oz[k];

                        if (nxp < 0 || nxp >= nx || nyp < 0 || nyp >= ny || nzp < 0 || nzp >= nz) {
                            touches_obstacle = true;
                            continue;
                        }

                        int nid = idx3_h(nxp, nyp, nzp, nx, ny, nz);
                        if (solid[nid]) touches_obstacle = true;
                        if (fluid[nid] && !solid[nid] && !vis[nid]) {
                            vis[nid] = 1;
                            q.push(nid);
                        }
                    }

                    if (cy <= 1) touches_obstacle = true;
                }

                if (touches_obstacle || (int)comp.size() < 16) {
                    continue;
                }

                Vec3 pre(0, 0, 0), post(0, 0, 0);
                for (int id : comp) {
                    pre += Vec3(u_before[id], v_before[id], w_before[id]);
                    post += Vec3(u_after[id], v_after[id], w_after[id]);
                }
                float invn = 1.0f / (float)comp.size();
                pre = pre * invn;
                post = post * invn;
                Vec3 delta = pre - post;

                for (int id : comp) {
                    u_after[id] += delta.x;
                    v_after[id] += delta.y;
                    w_after[id] += delta.z;
                }
                corrected++;
            }
        }
    }

    return corrected;
}

int main(int argc, char** argv) {
    std::string obj_path = (argc >= 2) ? argv[1] : "stanford-bunny.obj";
    int total_frames = (argc >= 3) ? std::max(1, std::atoi(argv[2])) : 300;
    std::string out_dir = (argc >= 4) ? argv[3] : "chocolate_frames";

    const int NX = 72;
    const int NY = 96;
    const int NZ = 72;
    const int N = NX * NY * NZ;

    const int VX = NX * 2;
    const int VY = NY * 2;
    const int VZ = NZ * 2;
    const int VN = VX * VY * VZ;

    const int W = 960;
    const int H = 720;

    const float h = 1.0f / (float)std::max(NX, std::max(NY, NZ));
    const float h2 = h * h;
    const float h2inv = 1.0f / h2;
    const float inv2h = 0.5f / h;

    const int substeps = 2;
    const float dt = (1.0f / 30.0f) / (float)substeps;

    // Material parameters (chocolate bunny).
    const float ambient_temp = 20.0f;
    const float initial_temp = 19.0f;
    const float melt_center = 31.5f;
    const float melt_half = 1.9f;
    const float visc_cold = 14000.0f;
    const float visc_hot = 0.22f;

    const float thermal_kappa = 0.0007f;
    const int temp_jacobi_iters = 20;

    const int vel_diffuse_iters = 48;
    const int pressure_iters = 90;

    const float gravity = -1.8f;
    const float floor_y = 1.5f / (float)NY;

    Vec3 heat_src(0.66f, 0.73f, 0.46f);
    const float heat_radius = 0.19f;
    const float heat_temp = 45.0f;
    const float heat_gain = 8.8f;
    const float cool_rate = 0.10f;

    std::vector<Tri> tris;
    if (!load_obj_triangles(obj_path, tris)) {
        std::fprintf(stderr, "[Error] Failed to load OBJ: %s\n", obj_path.c_str());
        return 1;
    }
    std::printf("[Init] Loaded triangles: %zu\n", tris.size());

    std::vector<int> h_fluid0(N, 0), h_solid(N, 0);
    auto tri_yz = build_triyz(tris);
    voxelize_bunny_fill(NX, NY, NZ, tri_yz, h_fluid0, h_solid);

    int init_fluid_cells = 0;
    for (int i = 0; i < N; ++i) if (h_fluid0[i]) init_fluid_cells++;
    std::printf("[Init] Initial fluid cells: %d\n", init_fluid_cells);

    std::vector<Vec3> h_particles;
    seed_particles_from_fluid(NX, NY, NZ, h_fluid0, h_solid, h_particles);
    std::printf("[Init] Seed particles: %zu\n", h_particles.size());

    if (h_particles.empty()) {
        std::fprintf(stderr, "[Error] No particles generated from bunny voxelization.\n");
        return 1;
    }

    std::filesystem::create_directories(out_dir);
    const std::string out_front = out_dir + "/front";
    const std::string out_back = out_dir + "/back";
    const std::string out_side = out_dir + "/side";
    const std::string out_top = out_dir + "/top";
    std::filesystem::create_directories(out_front);
    std::filesystem::create_directories(out_back);
    std::filesystem::create_directories(out_side);
    std::filesystem::create_directories(out_top);

    GLFWwindow* win = nullptr;
    if (glfwInit()) {
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
        win = glfwCreateWindow(640, 360, "CUDA Chocolate Bunny Melt (no OpenGL interop)", nullptr, nullptr);
        if (!win) {
            std::fprintf(stderr, "[Warn] GLFW window creation failed. Running headless.\n");
            glfwTerminate();
        }
    } else {
        std::fprintf(stderr, "[Warn] glfwInit failed. Running headless.\n");
    }

    CUDA_CHECK(cudaSetDevice(0));

    // Device arrays.
    int* d_fluid = nullptr;
    int* d_solid = nullptr;

    float *d_u = nullptr, *d_v = nullptr, *d_w = nullptr;
    float *d_u_tmp = nullptr, *d_v_tmp = nullptr, *d_w_tmp = nullptr;
    float *d_u_rhs = nullptr, *d_v_rhs = nullptr, *d_w_rhs = nullptr;

    float *d_temp = nullptr, *d_temp_tmp = nullptr, *d_temp_rhs = nullptr;
    float *d_visc = nullptr;

    float *d_div = nullptr;
    float *d_p = nullptr, *d_p_tmp = nullptr;

    Vec3* d_particles = nullptr;

    float *d_vol = nullptr, *d_vol_tmp = nullptr;
    uchar4* d_image = nullptr;

    CUDA_CHECK(cudaMalloc(&d_fluid, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_solid, N * sizeof(int)));

    CUDA_CHECK(cudaMalloc(&d_u, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_w, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_u_tmp, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v_tmp, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_w_tmp, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_u_rhs, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v_rhs, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_w_rhs, N * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_temp, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_temp_tmp, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_temp_rhs, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_visc, N * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_div, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_p, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_p_tmp, N * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_particles, h_particles.size() * sizeof(Vec3)));

    CUDA_CHECK(cudaMalloc(&d_vol, VN * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_vol_tmp, VN * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_image, (size_t)W * H * sizeof(uchar4)));

    CUDA_CHECK(cudaMemcpy(d_solid, h_solid.data(), N * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_particles, h_particles.data(), h_particles.size() * sizeof(Vec3), cudaMemcpyHostToDevice));

    int tpb = 256;
    int blocksN = (N + tpb - 1) / tpb;
    int blocksP = ((int)h_particles.size() + tpb - 1) / tpb;
    int blocksV = (VN + tpb - 1) / tpb;

    clear_float_kernel<<<blocksN, tpb>>>(d_u, N, 0.0f);
    clear_float_kernel<<<blocksN, tpb>>>(d_v, N, 0.0f);
    clear_float_kernel<<<blocksN, tpb>>>(d_w, N, 0.0f);

    // Initial fluid from particles.
    clear_int_kernel<<<blocksN, tpb>>>(d_fluid, N, 0);
    mark_fluid_from_particles<<<blocksP, tpb>>>(d_particles, (int)h_particles.size(), d_fluid, NX, NY, NZ);
    enforce_solid_mask<<<blocksN, tpb>>>(d_fluid, d_solid, N);

    std::vector<float> h_temp(N, ambient_temp);
    for (int i = 0; i < N; ++i) {
        if (h_fluid0[i]) h_temp[i] = initial_temp;
    }
    CUDA_CHECK(cudaMemcpy(d_temp, h_temp.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    viscosity_from_temperature<<<blocksN, tpb>>>(d_temp, d_visc, d_fluid, N,
                                                 melt_center, melt_half,
                                                 visc_cold, visc_hot);

    // Host buffers for free-flight reinjection.
    std::vector<int> h_fluid(N, 0);
    std::vector<float> h_ub(N), h_vb(N), h_wb(N);
    std::vector<float> h_ua(N), h_va(N), h_wa(N);

    std::vector<uchar4> h_img((size_t)W * H);

    auto swap_ptr = [](float*& a, float*& b) {
        float* t = a;
        a = b;
        b = t;
    };

    std::printf("[Run] Chocolate melt frames: %d, substeps/frame: %d, dt: %.6f\n", total_frames, substeps, dt);

    for (int frame = 0; frame < total_frames; ++frame) {
        if (win) {
            glfwPollEvents();
            if (glfwWindowShouldClose(win)) {
                std::printf("[Run] Window closed by user, stopping at frame %d\n", frame);
                break;
            }
            char title[256];
            std::snprintf(title, sizeof(title), "CUDA Chocolate Bunny Melt - frame %d/%d", frame + 1, total_frames);
            glfwSetWindowTitle(win, title);
        }

        // Move heater slowly around upper bunny region.
        float ft = (total_frames > 1) ? (frame / (float)(total_frames - 1)) : 0.0f;
        float heat_ang = -0.9f + 1.8f * ft;
        heat_src.x = 0.5f + 0.22f * cosf(heat_ang);
        heat_src.z = 0.5f + 0.16f * sinf(heat_ang);
        heat_src.y = 0.73f;

        for (int sub = 0; sub < substeps; ++sub) {
            // Rebuild fluid marker from particles (MAC marker step).
            clear_int_kernel<<<blocksN, tpb>>>(d_fluid, N, 0);
            mark_fluid_from_particles<<<blocksP, tpb>>>(d_particles, (int)h_particles.size(), d_fluid, NX, NY, NZ);
            enforce_solid_mask<<<blocksN, tpb>>>(d_fluid, d_solid, N);

            // Temperature: advection + source/cooling + implicit diffusion.
            temp_advect<<<blocksN, tpb>>>(d_temp, d_u, d_v, d_w, d_temp_rhs,
                                          d_fluid, ambient_temp, dt, NX, NY, NZ);
            temp_apply_sources<<<blocksN, tpb>>>(d_temp_rhs, d_fluid, NX, NY, NZ, dt,
                                                 ambient_temp, cool_rate,
                                                 heat_src, heat_radius, heat_temp, heat_gain);

            CUDA_CHECK(cudaMemcpy(d_temp, d_temp_rhs, N * sizeof(float), cudaMemcpyDeviceToDevice));
            for (int it = 0; it < temp_jacobi_iters; ++it) {
                temp_diffuse_jacobi<<<blocksN, tpb>>>(d_temp_rhs, d_temp, d_temp_tmp,
                                                      d_fluid, NX, NY, NZ, dt, thermal_kappa, h2inv);
                swap_ptr(d_temp, d_temp_tmp);
            }

            viscosity_from_temperature<<<blocksN, tpb>>>(d_temp, d_visc, d_fluid, N,
                                                         melt_center, melt_half,
                                                         visc_cold, visc_hot);

            // Velocity: advection + gravity.
            advect_velocity_semi_lagrangian<<<blocksN, tpb>>>(
                d_u, d_v, d_w,
                d_u_tmp, d_v_tmp, d_w_tmp,
                d_fluid, d_solid,
                dt, NX, NY, NZ);
            swap_ptr(d_u, d_u_tmp);
            swap_ptr(d_v, d_v_tmp);
            swap_ptr(d_w, d_w_tmp);

            add_gravity<<<blocksN, tpb>>>(d_v, d_fluid, d_solid, dt, gravity, N);

            // Store RHS and pre-diffusion velocity for free-flight reinjection.
            CUDA_CHECK(cudaMemcpy(d_u_rhs, d_u, N * sizeof(float), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(d_v_rhs, d_v, N * sizeof(float), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(d_w_rhs, d_w, N * sizeof(float), cudaMemcpyDeviceToDevice));

            // Implicit variable-viscosity diffusion (Jacobi iterations).
            for (int it = 0; it < vel_diffuse_iters; ++it) {
                diffuse_velocity_jacobi<<<blocksN, tpb>>>(d_u_rhs, d_u, d_u_tmp, d_visc,
                                                          d_fluid, d_solid, NX, NY, NZ, dt, h2inv);
                diffuse_velocity_jacobi<<<blocksN, tpb>>>(d_v_rhs, d_v, d_v_tmp, d_visc,
                                                          d_fluid, d_solid, NX, NY, NZ, dt, h2inv);
                diffuse_velocity_jacobi<<<blocksN, tpb>>>(d_w_rhs, d_w, d_w_tmp, d_visc,
                                                          d_fluid, d_solid, NX, NY, NZ, dt, h2inv);
                swap_ptr(d_u, d_u_tmp);
                swap_ptr(d_v, d_v_tmp);
                swap_ptr(d_w, d_w_tmp);
            }

            // Free-flight momentum reintroduction (host-side connected components).
            CUDA_CHECK(cudaMemcpy(h_fluid.data(), d_fluid, N * sizeof(int), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_ub.data(), d_u_rhs, N * sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_vb.data(), d_v_rhs, N * sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_wb.data(), d_w_rhs, N * sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_ua.data(), d_u, N * sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_va.data(), d_v, N * sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_wa.data(), d_w, N * sizeof(float), cudaMemcpyDeviceToHost));

            (void)reinject_free_flight_bulk_velocity(NX, NY, NZ,
                                                     h_fluid, h_solid,
                                                     h_ub, h_vb, h_wb,
                                                     h_ua, h_va, h_wa);

            CUDA_CHECK(cudaMemcpy(d_u, h_ua.data(), N * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_v, h_va.data(), N * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_w, h_wa.data(), N * sizeof(float), cudaMemcpyHostToDevice));

            // Pressure projection.
            compute_divergence<<<blocksN, tpb>>>(d_u, d_v, d_w, d_div,
                                                 d_fluid, d_solid,
                                                 NX, NY, NZ, inv2h);

            clear_float_kernel<<<blocksN, tpb>>>(d_p, N, 0.0f);
            for (int it = 0; it < pressure_iters; ++it) {
                pressure_jacobi<<<blocksN, tpb>>>(d_div, d_p, d_p_tmp,
                                                  d_fluid, d_solid,
                                                  NX, NY, NZ, h2 / dt);
                swap_ptr(d_p, d_p_tmp);
            }

            project_subtract_gradient<<<blocksN, tpb>>>(d_u, d_v, d_w, d_p,
                                                        d_fluid, d_solid,
                                                        NX, NY, NZ, dt, inv2h);

            // Advect particles with projected velocity.
            advect_particles<<<blocksP, tpb>>>(d_particles, (int)h_particles.size(),
                                               d_u, d_v, d_w,
                                               dt, NX, NY, NZ, floor_y);
        }

        // Build high-resolution volume from particles (splat + blur).
        clear_float_kernel<<<blocksV, tpb>>>(d_vol, VN, 0.0f);
        splat_particles_volume<<<blocksP, tpb>>>(d_particles, (int)h_particles.size(),
                                                 d_vol, VX, VY, VZ,
                                                 2.5f, 0.42f);
        clamp_volume<<<blocksV, tpb>>>(d_vol, VN, 1.0f);
        for (int i = 0; i < 3; ++i) {
            blur_volume_6n<<<blocksV, tpb>>>(d_vol, d_vol_tmp, VX, VY, VZ);
            CUDA_CHECK(cudaMemcpy(d_vol, d_vol_tmp, VN * sizeof(float), cudaMemcpyDeviceToDevice));
        }

        dim3 b2(16, 16);
        dim3 g2((W + b2.x - 1) / b2.x, (H + b2.y - 1) / b2.y);
        Vec3 target(0.5f, 0.33f, 0.5f);

        auto build_cam = [&](const Vec3& cam_pos, const Vec3& prefer_up) {
            RenderCam cam{};
            cam.pos = cam_pos;
            cam.fwd = normalize(target - cam_pos);
            Vec3 up_ref = prefer_up;
            Vec3 right = cross(cam.fwd, up_ref);
            if (length(right) < 1e-5f) {
                up_ref = Vec3(0, 0, 1);
                right = cross(cam.fwd, up_ref);
                if (length(right) < 1e-5f) {
                    up_ref = Vec3(1, 0, 0);
                    right = cross(cam.fwd, up_ref);
                }
            }
            cam.right = normalize(right);
            cam.up = normalize(cross(cam.right, cam.fwd));
            cam.tan_half_fov = tanf(37.0f * (float)M_PI / 180.0f);
            cam.aspect = W / (float)H;
            return cam;
        };

        struct ViewDesc {
            const char* name;
            std::string dir;
            Vec3 pos;
            Vec3 up;
        };

        const float view_r = 1.18f;
        const float side_h = 0.53f;
        std::array<ViewDesc, 4> views = {{
            {"front", out_front, Vec3(target.x, side_h, target.z + view_r), Vec3(0, 1, 0)},
            {"back",  out_back,  Vec3(target.x, side_h, target.z - view_r), Vec3(0, 1, 0)},
            {"side",  out_side,  Vec3(target.x + view_r, side_h, target.z), Vec3(0, 1, 0)},
            {"top",   out_top,   Vec3(target.x, target.y + 1.20f, target.z), Vec3(0, 0, -1)},
        }};

        for (const auto& vdesc : views) {
            RenderCam cam = build_cam(vdesc.pos, vdesc.up);
            raymarch_render<<<g2, b2>>>(d_image, W, H, cam,
                                        d_vol, VX, VY, VZ,
                                        d_temp, NX, NY, NZ,
                                        melt_center, melt_half, floor_y);

            CUDA_CHECK(cudaMemcpy(h_img.data(), d_image, (size_t)W * H * sizeof(uchar4), cudaMemcpyDeviceToHost));

            char name[512];
            std::snprintf(name, sizeof(name), "%s/frame_%04d.png", vdesc.dir.c_str(), frame);
            if (!stbi_write_png(name, W, H, 4, h_img.data(), W * 4)) {
                std::fprintf(stderr, "[Warn] Failed to write frame: %s\n", name);
            }
        }

        if ((frame + 1) % 10 == 0 || frame == total_frames - 1) {
            std::printf("[Run] Saved frame %d / %d (front/back/side/top)\n", frame + 1, total_frames);
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    if (win) {
        glfwDestroyWindow(win);
        glfwTerminate();
    }

    CUDA_CHECK(cudaFree(d_fluid));
    CUDA_CHECK(cudaFree(d_solid));

    CUDA_CHECK(cudaFree(d_u));
    CUDA_CHECK(cudaFree(d_v));
    CUDA_CHECK(cudaFree(d_w));
    CUDA_CHECK(cudaFree(d_u_tmp));
    CUDA_CHECK(cudaFree(d_v_tmp));
    CUDA_CHECK(cudaFree(d_w_tmp));
    CUDA_CHECK(cudaFree(d_u_rhs));
    CUDA_CHECK(cudaFree(d_v_rhs));
    CUDA_CHECK(cudaFree(d_w_rhs));

    CUDA_CHECK(cudaFree(d_temp));
    CUDA_CHECK(cudaFree(d_temp_tmp));
    CUDA_CHECK(cudaFree(d_temp_rhs));
    CUDA_CHECK(cudaFree(d_visc));

    CUDA_CHECK(cudaFree(d_div));
    CUDA_CHECK(cudaFree(d_p));
    CUDA_CHECK(cudaFree(d_p_tmp));

    CUDA_CHECK(cudaFree(d_particles));

    CUDA_CHECK(cudaFree(d_vol));
    CUDA_CHECK(cudaFree(d_vol_tmp));

    CUDA_CHECK(cudaFree(d_image));

    std::printf("[Done] Output frames:\n");
    std::printf("  - %s\n", out_front.c_str());
    std::printf("  - %s\n", out_back.c_str());
    std::printf("  - %s\n", out_side.c_str());
    std::printf("  - %s\n", out_top.c_str());
    return 0;
}
