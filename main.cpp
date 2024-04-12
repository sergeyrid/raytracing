#include <algorithm>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include <omp.h>

using namespace std;

const float EPS = 1e-5f;
const float INF = 10000.f;
const float BIG_INF = 20000.f;
const float NEG_INF = -10000.f;

struct ColorInt {
    uint8_t red;
    uint8_t green;
    uint8_t blue;
};

struct SceneInt {
    uint16_t width;
    uint16_t height;
    vector<ColorInt> data;

    SceneInt(const uint16_t &width, const uint16_t &height) : width(width), height(height) {
        data = vector<ColorInt>();
        data.reserve(width * height);
    }
};

struct SceneFloat {
    uint16_t width;
    uint16_t height;
    vector<glm::vec3> data;

    SceneFloat(const uint16_t &width, const uint16_t &height) : width(width), height(height) {
        data = vector<glm::vec3>();
        data.resize(width * height);
    }
};

enum class Material {
    DIFFUSER,
    METALLIC,
    DIELECTRIC,
};

struct Primitive;

struct Intersection {
    bool isIntersected = false;
    bool isInside = false;

    float t = 0.;
    glm::vec3 p{0., 0., 0.};
    glm::vec3 d{0., 0., 0.};
    float nl = 0.;

    glm::vec3 normal{0., 0., 0.};
    glm::vec3 pPlusNormalEps{0., 0., 0.};
    glm::vec3 pMinusNormalEps{0., 0., 0.};
    const Primitive *primitive = nullptr;

    void update(const glm::vec3 &o, const glm::vec3 &newD, const Primitive *newPrimitive) {
        if (!isIntersected) {
            return;
        }
        d = newD;
        nl = -glm::dot(normal, d);
        p = o + t * d;
        primitive = newPrimitive;
        glm::vec3 normalEps = normal * EPS;
        pPlusNormalEps = p + normalEps;
        pMinusNormalEps = p - normalEps;
    }
};

uint16_t sampleUInt(minstd_rand &RNG, uint16_t a = 0, uint16_t b = 0) {
    uniform_int_distribution<uint16_t> dis{a, b};
    return dis(RNG);
}

float sampleUniform(minstd_rand &RNG, float a = 0., float b = 1.) {
    uniform_real_distribution<float> dis{a, b};
    return dis(RNG);
}

float sampleNormal(minstd_rand &RNG, float m = 0., float s = 1.) {
    normal_distribution<float> dis{m, s};
    return dis(RNG);
}

struct AABB {
    glm::vec3 minPoint{INF, INF, INF};
    glm::vec3 maxPoint{NEG_INF, NEG_INF, NEG_INF};
    glm::vec3 center{0., 0., 0.};
    glm::vec3 sizePlusCenter{0., 0., 0.};
    glm::vec3 negSizePlusCenter{0., 0., 0.};

    float area() const {
        if (minPoint.x > maxPoint.x || minPoint.y > maxPoint.y || minPoint.z > maxPoint.z) {
            return 0.f;
        }

        float a = maxPoint.x - minPoint.x;
        float b = maxPoint.y - minPoint.y;
        float c = maxPoint.z - minPoint.z;

        return a * b + a * c + b * c;
    }

    bool isInside(const glm::vec3 &p) const {
        return maxPoint.x > p.x && maxPoint.y > p.y && maxPoint.z > p.z &&
                minPoint.x < p.x && minPoint.y < p.y && minPoint.z < p.z;
    }

    void extend(glm::vec3 p)
    {
        minPoint = glm::min(minPoint, p);
        maxPoint = glm::max(maxPoint, p);
    }

    void extend(const AABB &aabb)
    {
        minPoint = glm::min(minPoint, aabb.minPoint);
        maxPoint = glm::max(maxPoint, aabb.maxPoint);
    }

    void rotateAndTranslate(const glm::quat &rotation, const glm::vec3 &translation) {
        AABB zeroAABB = *this;

        minPoint = {INF, INF, INF};
        maxPoint = {-INF, -INF, -INF};

        extend(rotation * zeroAABB.maxPoint);
        extend(rotation * glm::vec3{zeroAABB.maxPoint.x, zeroAABB.maxPoint.y, zeroAABB.minPoint.z});
        extend(rotation * glm::vec3{zeroAABB.maxPoint.x, zeroAABB.minPoint.y, zeroAABB.maxPoint.z});
        extend(rotation * glm::vec3{zeroAABB.minPoint.x, zeroAABB.maxPoint.y, zeroAABB.maxPoint.z});
        extend(rotation * glm::vec3{zeroAABB.maxPoint.x, zeroAABB.minPoint.y, zeroAABB.minPoint.z});
        extend(rotation * glm::vec3{zeroAABB.minPoint.x, zeroAABB.maxPoint.y, zeroAABB.minPoint.z});
        extend(rotation * glm::vec3{zeroAABB.minPoint.x, zeroAABB.minPoint.y, zeroAABB.maxPoint.z});
        extend(rotation * zeroAABB.minPoint);

        minPoint += translation;
        maxPoint += translation;
    }

    void calculateEverything() {
        center = glm::vec3 {
                (maxPoint.x + minPoint.x) / 2.f,
                (maxPoint.y + minPoint.y) / 2.f,
                (maxPoint.z + minPoint.z) / 2.f,
        };
        glm::vec3 size = maxPoint - minPoint;
        glm::vec3 negSize = -size;
        sizePlusCenter = size + center;
        negSizePlusCenter = negSize + center;
    }

    float intersect(const glm::vec3 &o, const glm::vec3 &d) const {
        if (isInside(o)) {
            return 0.f;
        }

        glm::vec3 t1 = (sizePlusCenter - o) / d;
        glm::vec3 t2 = (negSizePlusCenter - o) / d;

        float d1 = max(max(min(t1.x, t2.x), min(t1.y, t2.y)), min(t1.z, t2.z));
        float d2 = min(min(max(t1.x, t2.x), max(t1.y, t2.y)), max(t1.z, t2.z));

        if (d1 > d2 || d2 < 0.f) {
            return BIG_INF;
        } else {
            return d1;
        }
    }
};

struct Primitive {
    glm::vec3 position{0., 0., 0.};
    glm::quat rotation{1., 0., 0., 0.};
    glm::quat conjRotation{1., 0., 0., 0.};
    glm::vec3 color{0., 0., 0.};
    glm::vec3 emission{0., 0., 0.};
    Material material = Material::DIFFUSER;
    float ior = 1.;
    bool isPlane = false;
    AABB aabb;

    virtual Intersection intersectPrimitive(const glm::vec3 &o, const glm::vec3 &d) = 0;

    virtual glm::vec3 samplePoint(minstd_rand &RNG) = 0;
    virtual float pdf(glm::vec3 o, glm::vec3 d) = 0;

    virtual void calculateAABB() = 0;

    virtual void update() {
        calculateAABB();
    }

    pair<Intersection, Intersection> intersectFull(const glm::vec3 &o, const glm::vec3 &d) {
        Intersection i1 = intersectPrimitive(o, d);
        if (!i1.isIntersected) {
            return {i1, i1};
        }
        Intersection i2;
        if (i1.isInside) {
            i2 = intersectPrimitive(i1.pPlusNormalEps, -d);
        } else {
            i2 = intersectPrimitive(i1.pMinusNormalEps, d);
        }
        return {i1, i2};
    }

    float pdfFull(const glm::vec3 &o, const Intersection &i1, const Intersection &i2,
                  const float &p1, const float &p2) {
        glm::vec3 r1 = i1.p - o;
        glm::vec3 r2 = i2.p - o;
        float fp1 = p1 * glm::dot(r1, r1) / glm::abs(glm::dot(glm::normalize(r1), i1.normal));
        float fp2 = p2 * glm::dot(r2, r2) / glm::abs(glm::dot(glm::normalize(r2), i2.normal));
        return fp1 + fp2;
    }
};

struct Plane : Primitive {
    glm::vec3 normal{0., 0., 1.};
    glm::vec3 rotatedNormal{0., 0., 1.};

    Intersection intersectPrimitive(const glm::vec3 &o, const glm::vec3 &d) override {
        glm::vec3 no = conjRotation * (o - position);
        glm::vec3 nd = conjRotation * d;

        Intersection intersection;
        intersection.normal = rotatedNormal;
        intersection.t = -glm::dot(no, normal) / glm::dot(nd, normal);
        intersection.isIntersected = intersection.t > 0.f;
        intersection.update(o, d, this);
        return intersection;
    }

    glm::vec3 samplePoint(minstd_rand &RNG) override {
        throw bad_function_call();
    }

    float pdf(glm::vec3 o, glm::vec3 d) override {
        throw bad_function_call();
    }

    void calculateAABB() override {
        throw bad_function_call();
    }

    void update() override {
        rotatedNormal = rotation * normal;
    }
};

struct Ellipsoid : Primitive {
    glm::vec3 radius{1., 1., 1.};
    glm::vec3 radiusSq{1., 1., 1.};

    Intersection intersectPrimitive(const glm::vec3 &o, const glm::vec3 &d) override {
        glm::vec3 no = conjRotation * (o - position);
        glm::vec3 nd = conjRotation * d;

        glm::vec3 ndr = nd / radius;
        glm::vec3 nor = no / radius;

        float a = glm::dot(ndr, ndr);
        float b = glm::dot(nor, ndr) / a;
        float c = (glm::dot(nor, nor) - 1) / a;

        Intersection intersection;

        float discr = b * b - c;
        if (discr < 0.f) {
            return intersection;
        }
        float sqrtDiscr = glm::sqrt(discr);

        float mind = -b - sqrtDiscr;
        float maxd = -b + sqrtDiscr;

        if (mind < 0.f && maxd > 0.f) {
            intersection.t = maxd;
            intersection.normal = -glm::normalize(rotation * ((no + maxd * nd) / radiusSq));
            intersection.isInside = true;
            intersection.isIntersected = true;
        } else if (mind > 0.f) {
            intersection.t = mind;
            intersection.normal = glm::normalize(rotation * ((no + mind * nd) / radiusSq));
            intersection.isIntersected = true;
        }

        intersection.update(o, d, this);

        return intersection;
    }

    glm::vec3 samplePoint(minstd_rand &RNG) override {
        float x = sampleNormal(RNG);
        float y = sampleNormal(RNG);
        float z = sampleNormal(RNG);
        glm::vec3 point{x, y, z};
        point = glm::normalize(point);
        point.x *= radius.x;
        point.y *= radius.y;
        point.z *= radius.z;
        return rotation * point + position;
    }

    float pdf(glm::vec3 o, glm::vec3 d) override {
        pair<Intersection, Intersection> is = intersectFull(o, d);
        Intersection &i1 = is.first;
        Intersection &i2 = is.second;

        if (!i1.isIntersected) {
            return 0.f;
        }

        float rs = radius.x * radius.y * radius.z;
        rs = rs * rs;
        float rxs = radius.x * radius.x;
        float rys = radius.y * radius.y;
        float rzs = radius.z * radius.z;

        glm::vec3 pos1 = conjRotation * (i1.p - position);
        glm::vec3 pos2 = conjRotation * (i1.p - position);

        float p1 = glm::one_over_pi<float>() / 4.f / glm::sqrt(
                pos1.x * pos1.x * rs / rxs + pos1.y * pos1.y * rs / rys + pos1.z * pos1.z * rs / rzs);
        float p2 = glm::one_over_pi<float>() / 4.f / glm::sqrt(
                pos2.x * pos2.x * rs / rxs + pos2.y * pos2.y * rs / rys + pos2.z * pos2.z * rs / rzs);

        return pdfFull(o, i1, i2, p1, p2);
    }

    void calculateAABB() override {
        aabb = AABB{};
        aabb.extend(radius);
        aabb.extend(-radius);
        aabb.rotateAndTranslate(rotation, position);
        aabb.calculateEverything();
    }
};

struct Box : Primitive {
    glm::vec3 size{1., 1., 1.};

    Intersection intersectPrimitive(const glm::vec3 &o, const glm::vec3 &d) override {
        glm::vec3 no = conjRotation * (o - position);
        glm::vec3 nd = conjRotation * d;

        glm::vec3 t1 = (size - no) / nd;
        glm::vec3 t2 = (-size - no) / nd;

        float d1 = max(max(min(t1.x, t2.x), min(t1.y, t2.y)), min(t1.z, t2.z));
        float d2 = min(min(max(t1.x, t2.x), max(t1.y, t2.y)), max(t1.z, t2.z));

        Intersection intersection;

        if (d1 > d2 || d2 < 0.f) {
            return intersection;
        }

        intersection.isIntersected = true;
        if (d1 < 0.f) {
            intersection.t = d2;
            intersection.isInside = true;
        } else {
            intersection.t = d1;
        }

        glm::vec3 ps = (no + intersection.t * nd) / size;
        glm::vec3 aps {fabs(ps.x), fabs(ps.y), fabs(ps.z)};
        if (aps.x > aps.y && aps.x > aps.z) {
            if (ps.x < 0) {
                intersection.normal = {-1., 0., 0.};
            } else {
                intersection.normal = {1., 0., 0.};
            }
        } else if (aps.y > aps.x && aps.y > aps.z) {
            if (ps.y < 0) {
                intersection.normal = {0., -1., 0.};
            } else {
                intersection.normal = {0., 1., 0.};
            }
        } else {
            if (ps.z < 0) {
                intersection.normal = {0., 0., -1.};
            } else {
                intersection.normal = {0., 0., 1.};
            }
        }
        intersection.normal = rotation * intersection.normal;
        if (intersection.isInside) {
            intersection.normal = -intersection.normal;
        }

        intersection.update(o, d, this);

        return intersection;
    }

    glm::vec3 samplePoint(minstd_rand &RNG) override {
        float wx = size.y * size.z;
        float wy = size.x * size.z;
        float wz = size.x * size.y;
        float u = sampleUniform(RNG, 0.f, wx + wy + wz);
        float side = 2 * (float(sampleUniform(RNG) < 0.5f) - 0.5f);
        float c1 = sampleUniform(RNG, 0.f, 2.f) - 1.f, c2 = sampleUniform(RNG, 0.f, 2.f) - 1.f;
        glm::vec3 point;
        if (u < wx) {
            point = {side * size.x, c1 * size.y, c2 * size.z};
        } else if (u < wx + wy) {
            point = {c1 * size.x, side * size.y, c2 * size.z};
        } else {
            point = {c1 * size.x, c2 * size.y, side * size.z};
        }
        return rotation * point + position;
    }

    float pdf(glm::vec3 o, glm::vec3 d) override {
        pair<Intersection, Intersection> is = intersectFull(o, d);
        Intersection &i1 = is.first;
        Intersection &i2 = is.second;

        if (!i1.isIntersected) {
            return 0.f;
        }

        float p = 1.f / 8.f / (size.y * size.z + size.x * size.z + size.x * size.y);

        return pdfFull(o, i1, i2, p, p);
    }

    void calculateAABB() override {
        aabb = AABB{};
        aabb.extend(size / 2.f);
        aabb.extend(-size / 2.f);
        aabb.rotateAndTranslate(rotation, position);
        aabb.calculateEverything();
    }
};

struct Triangle : Primitive {
    glm::vec3 pointA{0., 0., 0.};
    glm::vec3 pointB{0., 0., 0.};
    glm::vec3 pointC{0., 0., 0.};
    glm::vec3 normal{0., 0., 1.};
    glm::vec3 sideAB{0., 0., 0.};
    glm::vec3 sideAC{0., 0., 0.};
    float pdfConst = 0.;

    Triangle(float ax, float ay, float az, float bx, float by, float bz, float cx, float cy, float cz)
            : pointA(ax, ay, az), pointB(bx, by, bz), pointC(cx, cy, cz) {}

    Intersection intersectPrimitive(const glm::vec3 &o, const glm::vec3 &d) override {
        Intersection intersection;

        glm::vec3 uvt = glm::inverse(glm::mat3x3{sideAB, sideAC, -d}) * (o - pointA);

        if (uvt[0] < 0.f || uvt[1] < 0.f || uvt[2] < 0.f || uvt[0] + uvt[1] > 1.f) {
            return intersection;
        }

        intersection.isIntersected = true;
        intersection.t = uvt[2];
        intersection.normal = normal;
        if (glm::dot(normal, d) > 0.f) {
            intersection.isInside = true;
            intersection.normal *= -1.f;
        }

        intersection.update(o, d, this);

        return intersection;
    }

    glm::vec3 samplePoint(minstd_rand &RNG) override {
        float u = sampleUniform(RNG);
        float v = sampleUniform(RNG);
        if (u + v > 1.f) {
            u = 1.f - u;
            v = 1.f - v;
        }
        return rotation * (pointA + u * sideAB + v * sideAC) + position;
    }

    float pdf(glm::vec3 o, glm::vec3 d) override {
        Intersection i = intersectPrimitive(o, d);
        if (!i.isIntersected){
            return 0.f;
        }
        glm::vec3 r = i.p - o;
        return pdfConst * glm::dot(r, r) / glm::abs(glm::dot(glm::normalize(r), i.normal));
    }

    void calculateAABB() override {
        aabb = AABB{};
        aabb.extend(pointA);
        aabb.extend(pointB);
        aabb.extend(pointC);
        aabb.calculateEverything();
    }

    void update() override {
        normal = rotation * normal;
        pointA = rotation * pointA + position;
        pointB = rotation * pointB + position;
        pointC = rotation * pointC + position;
        sideAB = pointB - pointA;
        sideAC = pointC - pointA;
        glm::vec3 crossProduct = glm::cross(sideAB, sideAC);
        normal = glm::normalize(crossProduct);
        pdfConst = 2.f / glm::length(crossProduct);
        rotation = glm::quat {0.f, 0.f, 0.f, 1.f};
        position = glm::vec3 {};
        calculateAABB();
    }
};

struct BVHNode {
    AABB aabb;
    uint16_t left = 0;
    uint16_t right = 0;
    uint16_t firstPrimitiveId = 0;
    uint16_t primitiveCount = 0;
    uint8_t division = 0; // 0 - x, 1 - y, 2 - z
};

struct BVH {
    vector<BVHNode> nodes;
    uint16_t root = 0;

    void build(vector<shared_ptr<Primitive>> &primitives) {
        auto begin = partition(primitives.begin(), primitives.end(),
                               [](shared_ptr<Primitive> &primitive) { return primitive->isPlane; });
        buildRecursive(primitives, begin, primitives.end());
        nodes[root].firstPrimitiveId = 0;
        nodes[root].primitiveCount += distance(primitives.begin(), begin);
    }

    Intersection intersect(const vector<shared_ptr<Primitive>> &primitives, const glm::vec3 &o, const glm::vec3 &d,
                           uint16_t curNode = 0) const {
        Intersection closestIntersection;

        for (uint16_t i = nodes[curNode].firstPrimitiveId; i < nodes[curNode].firstPrimitiveId + nodes[curNode].primitiveCount; ++i) {
            Intersection newIntersection = primitives[i]->intersectPrimitive(o, d);
            if (newIntersection.isIntersected &&
                (!closestIntersection.isIntersected || newIntersection.t < closestIntersection.t)) {
                closestIntersection = newIntersection;
                closestIntersection.primitive = primitives[i].get();
            }
        }

        uint16_t left = nodes[curNode].left;
        uint16_t right = nodes[curNode].right;

        float leftDistance = nodes[left].aabb.intersect(o, d);
        float rightDistance = nodes[right].aabb.intersect(o, d);

        if (left != root && leftDistance < INF) {
            Intersection newIntersection = intersect(primitives, o, d, left);
            if (newIntersection.isIntersected &&
                (!closestIntersection.isIntersected || newIntersection.t < closestIntersection.t)) {
                closestIntersection = newIntersection;
            }
        }

        if (right != root && rightDistance < INF) {
            Intersection newIntersection = intersect(primitives, o, d, right);
            if (newIntersection.isIntersected &&
                (!closestIntersection.isIntersected || newIntersection.t < closestIntersection.t)) {
                closestIntersection = newIntersection;
            }
        }

        return closestIntersection;
    }

private:
    void buildRecursive(vector<shared_ptr<Primitive>> &primitives, auto begin, auto end) {
        uint16_t curNode = nodes.size();
        nodes.push_back(BVHNode{});

        for (auto i = begin; i != end; ++i) {
            nodes[curNode].aabb.extend((*i)->aabb);
        }
        nodes[curNode].aabb.calculateEverything();
        uint16_t totalCount = distance(begin, end);

        if (totalCount <= 4) {
            nodes[curNode].firstPrimitiveId = distance(primitives.begin(), begin);
            nodes[curNode].primitiveCount = totalCount;
            return;
        }

        float bestCost = (float)totalCount * nodes[curNode].aabb.area();
        auto middle = end;

        for (uint8_t division = 0; division < 3; ++division) {
            sort(begin, end, [division](shared_ptr<Primitive> &p1, shared_ptr<Primitive> &p2) {
                if (division == 0) {
                    return p1->aabb.center.x < p2->aabb.center.x;
                } else if (division == 1) {
                    return p1->aabb.center.y < p2->aabb.center.y;
                } else {
                    return p1->aabb.center.z < p2->aabb.center.z;
                }
            });

            AABB totalAABB;
            vector<AABB> leftAABBs(primitives.size() + 1);
            vector<AABB> rightAABBs(primitives.size() + 1);
            uint16_t j = 0;
            for (auto i = begin; i != end; ++i) {
                ++j;
                totalAABB.extend((*i)->aabb);
                leftAABBs[j] = totalAABB;
            }
            totalAABB = AABB{};
            for (auto i = end - 1; i != begin; --i) {
                --j;
                totalAABB.extend((*i)->aabb);
                rightAABBs[j] = totalAABB;
            }
            rightAABBs[0] = leftAABBs.back();

            for (uint16_t i = 1; i < totalCount; ++i) {
                float newCost = leftAABBs[i].area() * (float)i + rightAABBs[i].area() * (float)(totalCount - i);
                if (newCost < bestCost) {
                    bestCost = newCost;
                    middle = begin + i;
                    nodes[curNode].division = division;
                }
            }
        }

        sort(begin, end, [this, curNode](shared_ptr<Primitive> &p1, shared_ptr<Primitive> &p2) {
            if (nodes[curNode].division == 0) {
                return p1->aabb.center.x < p2->aabb.center.x;
            } else if (nodes[curNode].division == 1) {
                return p1->aabb.center.y < p2->aabb.center.y;
            } else {
                return p1->aabb.center.z < p2->aabb.center.z;
            }
        });

        if (middle == end) {
            nodes[curNode].firstPrimitiveId = distance(primitives.begin(), end);
            nodes[curNode].primitiveCount = totalCount;
            return;
        }

        nodes[curNode].left = curNode + 1;
        buildRecursive(primitives, begin, middle);

        nodes[curNode].right = nodes.size();
        buildRecursive(primitives, middle, end);
    }
};

struct InputData {
    uint16_t width = 0;
    uint16_t height = 0;
    uint16_t rayDepth = 0;
    uint16_t samples = 0;
    glm::vec3 backgroundColor{0., 0., 0.};

    glm::vec3 cameraPosition{0., 0., 0.};
    glm::vec3 cameraRight{1., 0., 0.};
    glm::vec3 cameraUp{0., 1., 0.};
    glm::vec3 cameraForward{0., 0., 1.};
    glm::vec2 cameraFovTan{0., 0.};

    vector<shared_ptr<Primitive>> primitives;
    vector<Primitive *> lights;

    BVH bvh;
};

struct Distribution {
    glm::vec3 x{0., 0., 0.};
    glm::vec3 n{0., 0., 0.};

    Distribution(glm::vec3 x, glm::vec3 n) : x(x), n(n) {}

    virtual glm::vec3 sample(minstd_rand &RNG) = 0;
    virtual float pdf(glm::vec3 d) = 0;

    virtual ~Distribution() = default;
};

struct Uniform : Distribution {
    using Distribution::Distribution;

    glm::vec3 sample(minstd_rand &RNG) override {
        float x = sampleNormal(RNG);
        float y = sampleNormal(RNG);
        float z = sampleNormal(RNG);
        glm::vec3 d{x, y, z};
        return glm::normalize(d);
    }

    float pdf(glm::vec3) override {
        return glm::four_over_pi<float>();
    }
};

struct Cosine : Distribution {
    using Distribution::Distribution;

    glm::vec3 sample(minstd_rand &RNG) override {
        float x = sampleNormal(RNG);
        float y = sampleNormal(RNG);
        float z = sampleNormal(RNG);
        glm::vec3 d{x, y, z};
        d = glm::normalize(d) + n * (1.f + EPS);
        return glm::normalize(d);
    }

    float pdf(glm::vec3 d) override {
        float dn = glm::dot(d, n);
        if (dn < EPS) {
            return 0.f;
        }
        return dn * glm::one_over_pi<float>();
    }
};

struct LightSurface : Distribution {
    const vector<Primitive *> *lights;
    const Primitive *primitive;

    LightSurface(glm::vec3 x, glm::vec3 n, const vector<Primitive *> *lights, const Primitive *primitive)
            : Distribution(x, n), lights(lights), primitive(primitive) {}

    glm::vec3 sample(minstd_rand &RNG) override {
        if (lights->empty()) {
            return {0.f, 0.f, 0.f};
        }
        return glm::normalize((*lights)[sampleUInt(RNG, 0, lights->size() - 1)]->samplePoint(RNG) - x);
    }

    float pdf(glm::vec3 d) override {
        float ps = 0.f;
        for (auto &light: *lights) {
            if (light == primitive) {
                continue;
            }
            ps += light->pdf(x, d);
        }
        if (ps < EPS || isnan(ps) || isinf(ps)) {
            return 0.f;
        }
        return ps / float(lights->size());
    }
};

struct Mix : Distribution {
    Cosine cosine;
    LightSurface lightSurface;

    Mix(glm::vec3 x, glm::vec3 n, const vector<Primitive *> *lights, const Primitive *primitive) :
            Distribution(x, n), cosine(x, n), lightSurface(x, n, lights, primitive) {}

    glm::vec3 sample(minstd_rand &RNG) override {
        bool c = sampleUniform(RNG) < 0.8f;
        if (c) {
            return cosine.sample(RNG);
        } else {
            return lightSurface.sample(RNG);
        }
    }

    float pdf(glm::vec3 d) override {
        return 0.8f * cosine.pdf(d) + 0.2f * lightSurface.pdf(d);
    }
};

glm::vec3 saturate(const glm::vec3 &color) {
    return glm::clamp(color, 0.f, 1.f);
}

glm::vec3 aces_tonemap(glm::vec3 const & x) {
    const float a = 2.51f;
    const float b = 0.03f;
    const float c = 2.43f;
    const float d = 0.59f;
    const float e = 0.14f;
    return saturate((x*(a*x+b))/(x*(c*x+d)+e));
}

ColorInt colorFloatToInt(glm::vec3 &color) {
    return ColorInt {
        uint8_t(color.x * 255),
        uint8_t(color.y * 255),
        uint8_t(color.z * 255),
    };
}

SceneInt sceneFloatToInt(SceneFloat &scene) {
    SceneInt sceneInt(scene.width, scene.height);
    for (auto &pixel: scene.data) {
        sceneInt.data.push_back(colorFloatToInt(pixel));
    }
    return sceneInt;
}

void printImage(SceneFloat &scene, string &outputPath) {
    SceneInt sceneInt = sceneFloatToInt(scene);
    ofstream outputFile(outputPath);
    outputFile << "P6" << endl << sceneInt.width << " " << sceneInt.height << endl << 255 << endl;
    for (auto &pixel: sceneInt.data) {
        outputFile << pixel.red << pixel.green << pixel.blue;
    }
    outputFile.close();
}

InputData parseInput(string &inputPath) {
    double start = omp_get_wtime();;
    ifstream inputFile(inputPath);

    InputData inputData;

    string command;
    shared_ptr<Primitive> lastPrimitive = nullptr;
    while (inputFile >> command) {
        if (command == "DIMENSIONS") {
            inputFile >> inputData.width >> inputData.height;
        } else if (command == "RAY_DEPTH") {
            inputFile >> inputData.rayDepth;
        } else if (command == "SAMPLES") {
            inputFile >> inputData.samples;
        } else if (command == "BG_COLOR") {
            inputFile >> inputData.backgroundColor.x >> inputData.backgroundColor.y >> inputData.backgroundColor.z;
        } else if (command == "CAMERA_POSITION") {
            inputFile >> inputData.cameraPosition.x >> inputData.cameraPosition.y >> inputData.cameraPosition.z;
        } else if (command == "CAMERA_RIGHT") {
            inputFile >> inputData.cameraRight.x >> inputData.cameraRight.y >> inputData.cameraRight.z;
        } else if (command == "CAMERA_UP") {
            inputFile >> inputData.cameraUp.x >> inputData.cameraUp.y >> inputData.cameraUp.z;
        } else if (command == "CAMERA_FORWARD") {
            inputFile >> inputData.cameraForward.x >> inputData.cameraForward.y >> inputData.cameraForward.z;
        } else if (command == "CAMERA_FOV_X") {
            float fovX;
            inputFile >> fovX;
            inputData.cameraFovTan.x = glm::tan(fovX / 2);
        } else if (command == "NEW_PRIMITIVE" && lastPrimitive != nullptr) {
            inputData.primitives.push_back(lastPrimitive);
            lastPrimitive->update();
            if (glm::length(lastPrimitive->emission) > EPS && !lastPrimitive->isPlane) {
                inputData.lights.push_back(lastPrimitive.get());
            }
            lastPrimitive = nullptr;
        } else if (command == "PLANE") {
            std::shared_ptr<Plane> newPlane(new Plane());
            inputFile >> newPlane->normal.x >> newPlane->normal.y >> newPlane->normal.z;
            newPlane->normal = glm::normalize(newPlane->normal);
            newPlane->isPlane = true;
            lastPrimitive = newPlane;
        } else if (command == "ELLIPSOID") {
            std::shared_ptr<Ellipsoid> newEllipsoid(new Ellipsoid());
            inputFile >> newEllipsoid->radius.x >> newEllipsoid->radius.y >> newEllipsoid->radius.z;
            newEllipsoid->radiusSq = newEllipsoid->radius * newEllipsoid->radius;
            lastPrimitive = newEllipsoid;
        } else if (command == "BOX") {
            std::shared_ptr<Box> newBox(new Box());
            inputFile >> newBox->size.x >> newBox->size.y >> newBox->size.z;
            lastPrimitive = newBox;
        } else if (command == "TRIANGLE") {
            float ax, ay, az, bx, by, bz, cx, cy, cz;
            inputFile >> ax >> ay >> az >> bx >> by >> bz >> cx >> cy >> cz;
            lastPrimitive = std::make_shared<Triangle>(ax, ay, az, bx, by, bz, cx, cy, cz);
        } else if (command == "POSITION") {
            inputFile >> lastPrimitive->position.x >> lastPrimitive->position.y >> lastPrimitive->position.z;
        } else if (command == "ROTATION") {
            inputFile >> lastPrimitive->rotation.x >> lastPrimitive->rotation.y;
            inputFile >> lastPrimitive->rotation.z >> lastPrimitive->rotation.w;
            lastPrimitive->conjRotation = glm::conjugate(lastPrimitive->rotation);
        } else if (command == "COLOR") {
            inputFile >> lastPrimitive->color.x >> lastPrimitive->color.y >> lastPrimitive->color.z;
        } else if (command == "EMISSION") {
            inputFile >> lastPrimitive->emission.x >> lastPrimitive->emission.y >> lastPrimitive->emission.z;
        } else if (command == "METALLIC") {
            lastPrimitive->material = Material::METALLIC;
        } else if (command == "DIELECTRIC") {
            lastPrimitive->material = Material::DIELECTRIC;
        } else if (command == "IOR") {
            inputFile >> lastPrimitive->ior;
        }
    }
    inputFile.close();
    if (lastPrimitive != nullptr) {
        inputData.primitives.push_back(lastPrimitive);
        lastPrimitive->update();
        if (glm::length(lastPrimitive->emission) > EPS && !lastPrimitive->isPlane) {
            inputData.lights.push_back(lastPrimitive.get());
        }
    }
    inputData.cameraFovTan.y = inputData.cameraFovTan.x * float(inputData.height) / float(inputData.width);
    inputData.bvh = BVH{};
    inputData.bvh.build(inputData.primitives);

    cout << "Processed input after " << omp_get_wtime() - start << "s" << endl;

    return inputData;
}

Intersection intersectScene(const glm::vec3 &o, const glm::vec3 &d, const InputData &inputData) {
    return inputData.bvh.intersect(inputData.primitives, o, d);
}

glm::vec3 applyLight(
        const Intersection &intersection, const InputData &inputData, const uint16_t &rayDepth, minstd_rand &RNG);

glm::vec3 getReflectedLight(
        const Intersection &intersection, const InputData &inputData, const uint16_t &rayDepth, minstd_rand &RNG) {
    glm::vec3 rd = intersection.d + 2.f * intersection.normal * intersection.nl;
    rd = glm::normalize(rd);
    Intersection reflectionIntersection = intersectScene(intersection.pPlusNormalEps, rd, inputData);
    return applyLight(reflectionIntersection, inputData, rayDepth - 1, RNG);
}

glm::vec3 applyLightDiffuser(
        const Intersection &intersection, const InputData &inputData, const uint16_t &rayDepth, minstd_rand &RNG) {
    Mix dis{intersection.p, intersection.normal, &inputData.lights, intersection.primitive};
    glm::vec3 w = dis.sample(RNG);
    float p = dis.pdf(w);
    float wn = glm::dot(w, intersection.normal);
    if (wn < 0.f || p < EPS) {
        return intersection.primitive->emission;
    }
    Intersection nextIntersection = intersectScene(intersection.pPlusNormalEps, w, inputData);
    glm::vec3 l = applyLight(nextIntersection, inputData, rayDepth - 1, RNG);
    return intersection.primitive->color * l * glm::one_over_pi<float>() * wn / p + intersection.primitive->emission;
}

glm::vec3 applyLightMetallic(
        const Intersection &intersection, const InputData &inputData, const uint16_t &rayDepth, minstd_rand &RNG) {
    glm::vec3 reflectedColor = getReflectedLight(intersection, inputData, rayDepth, RNG);
    return intersection.primitive->color * reflectedColor + intersection.primitive->emission;
}

glm::vec3 applyLightDielectric(
        const Intersection &intersection, const InputData &inputData, const uint16_t &rayDepth, minstd_rand &RNG) {
    float n1 = 1., n2 = intersection.primitive->ior;
    if (intersection.isInside) {
        n1 = n2;
        n2 = 1.;
    }
    float n12 = n1 / n2;
    float nl = intersection.nl;
    float s = n12 * glm::sqrt(1.f - nl * nl);

    if (s > 1.f) {
        return getReflectedLight(intersection, inputData, rayDepth, RNG) + intersection.primitive->emission;
    }

    float r0 = (n1 - n2) / (n1 + n2);
    r0 *= r0;
    float mnl = 1.f - nl;
    float mnlsq = mnl * mnl;
    float r = r0 + (1.f - r0) * mnlsq * mnlsq * mnl;

    if (sampleUniform(RNG) < r) {
        return getReflectedLight(intersection, inputData, rayDepth, RNG) + intersection.primitive->emission;
    }

    glm::vec3 rd = n12 * intersection.d + (n12 * nl - glm::sqrt(1 - s * s)) * intersection.normal;
    rd = glm::normalize(rd);
    Intersection refractedIntersection = intersectScene(intersection.pMinusNormalEps, rd, inputData);
    glm::vec3 refractedColor = applyLight(refractedIntersection, inputData, rayDepth - 1, RNG);

    if (!intersection.isInside) {
        refractedColor *= intersection.primitive->color;
    }

    return refractedColor + intersection.primitive->emission;
}

glm::vec3 applyLight(
        const Intersection &intersection, const InputData &inputData, const uint16_t &rayDepth, minstd_rand &RNG) {
    if (!intersection.isIntersected) {
        return inputData.backgroundColor;
    }

    glm::vec3 color{0., 0., 0.};
    if (rayDepth == 0) {
        return color;
    } else if (rayDepth == 1) {
        return intersection.primitive->emission;
    }

    if (intersection.primitive->material == Material::DIFFUSER) {
        color = applyLightDiffuser(intersection, inputData, rayDepth, RNG);
    } else if (intersection.primitive->material == Material::METALLIC) {
        color = applyLightMetallic(intersection, inputData, rayDepth, RNG);
    } else if (intersection.primitive->material == Material::DIELECTRIC) {
        color = applyLightDielectric(intersection, inputData, rayDepth, RNG);
    }

    return color;
}

glm::vec3 generateSample(uint16_t px, uint16_t py, const InputData &inputData, minstd_rand &RNG) {
    float x = (2 * (float(px) + sampleUniform(RNG)) / float(inputData.width) - 1) * inputData.cameraFovTan.x;
    float y = -(2 * (float(py) + sampleUniform(RNG)) / float(inputData.height) - 1) * inputData.cameraFovTan.y;
    glm::vec3 d = glm::normalize(x * inputData.cameraRight + y * inputData.cameraUp + inputData.cameraForward);
    Intersection intersection = intersectScene(inputData.cameraPosition, d, inputData);
    return applyLight(intersection, inputData, inputData.rayDepth, RNG);
}

glm::vec3 generatePixel(uint16_t px, uint16_t py, const InputData &inputData) {
    minstd_rand RNG{uint32_t(py * inputData.width + px)};
    glm::vec3 color{0., 0., 0.};
    for (uint16_t i = 0; i < inputData.samples; ++i) {
        color += generateSample(px, py, inputData, RNG);
    }
    color /= inputData.samples;
    color = aces_tonemap(color);
    float p = 1.f / 2.2f;
    color = {glm::pow(color.x, p),
             glm::pow(color.y, p),
             glm::pow(color.z, p)};
    return color;
}

SceneFloat generateScene(const InputData &inputData) {
    SceneFloat scene(inputData.width, inputData.height);
    #pragma omp parallel for schedule(dynamic, 8)
    for (uint32_t ij = 0; ij < uint32_t(inputData.height) * uint32_t(inputData.width); ++ij) {
        scene.data[ij] = generatePixel(ij % inputData.width, ij / inputData.width, inputData);
    }
    return scene;
}

int main(int, char *argv[]) {
    double start = omp_get_wtime();;

    string inputPath = argv[1];
    string outputPath = argv[2];

    InputData inputData = parseInput(inputPath);
    SceneFloat scene = generateScene(inputData);

    printImage(scene, outputPath);

    cout << "Finished after " << omp_get_wtime() - start << "s";
    return 0;
}
