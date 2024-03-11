#include <array>
#include <ctime>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

using namespace std;

const float EPS = 0.0001f;
minstd_rand RNG{42};

struct ColorInt {
    uint8_t red;
    uint8_t green;
    uint8_t blue;
};

struct SceneInt {
    uint32_t width;
    uint32_t height;
    vector<ColorInt> data;

    SceneInt(const uint32_t &width, const uint32_t &height) : width(width), height(height) {
        data = vector<ColorInt>();
        data.reserve(width * height);
    }
};

struct SceneFloat {
    uint32_t width;
    uint32_t height;
    vector<glm::vec3> data;

    SceneFloat(const uint32_t &width, const uint32_t &height) : width(width), height(height) {
        data = vector<glm::vec3>();
        data.reserve(width * height);
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
    const Primitive *primitive = nullptr;

    void update(const glm::vec3 &o, const glm::vec3 &newD, const Primitive *newPrimitive) {
        if (!isIntersected) {
            return;
        }
        d = newD;
        nl = -glm::dot(normal, d);
        p = o + t * d;
        primitive = newPrimitive;
    }
};

float sampleUniform(float a = 0., float b = 1.) {
    uniform_real_distribution<float> dis{a, b};
    return dis(RNG);
}

float sampleNormal(float m = 0., float s = 1.) {
    normal_distribution<float> dis{m, s};
    return dis(RNG);
}

struct Primitive {
    glm::vec3 position{0., 0., 0.};
    glm::quat rotation{1., 0., 0., 0.};
    glm::quat conjRotation{1., 0., 0., 0.};
    glm::vec3 color{0., 0., 0.};
    glm::vec3 emission{0., 0., 0.};
    Material material = Material::DIFFUSER;
    float ior = 1.;
    bool isPlane = false;

    virtual Intersection intersectPrimitive(const glm::vec3 &o, const glm::vec3 &d) = 0;

    virtual glm::vec3 samplePoint() = 0;
    virtual float pdf(glm::vec3 o, glm::vec3 d) = 0;

    pair<Intersection, Intersection> intersectFull(const glm::vec3 &o, const glm::vec3 &d) {
        Intersection i1 = intersectPrimitive(o, d);
        if (!i1.isIntersected) {
            return {i1, i1};
        }
        Intersection i2;
        if (i1.isInside) {
            i2 = intersectPrimitive(i1.p + i1.normal * EPS, glm::normalize(o - i1.p));
        } else {
            i2 = intersectPrimitive(i1.p - i1.normal * EPS, d);
        }
        i2.p = o + i2.t * d;
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

    Intersection intersectPrimitive(const glm::vec3 &o, const glm::vec3 &d) override {
        glm::vec3 no = conjRotation * (o - position);
        glm::vec3 nd = conjRotation * d;

        Intersection intersection;
        intersection.normal = rotation * normal;
        intersection.t = -glm::dot(no, normal) / glm::dot(nd, normal);
        intersection.isIntersected = intersection.t > 0;
        intersection.update(o, d, this);
        return intersection;
    }

    glm::vec3 samplePoint() override {
        throw bad_function_call();
    }

    float pdf(glm::vec3 o, glm::vec3 d) override {
        throw bad_function_call();
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
        if (discr < 0) {
            return intersection;
        }
        float sqrtDiscr = glm::sqrt(discr);

        float mind = -b - sqrtDiscr;
        float maxd = -b + sqrtDiscr;

        if (mind < 0 && maxd > 0) {
            intersection.t = maxd;
            intersection.normal = -glm::normalize(rotation * ((no + maxd * nd) / radiusSq));
            intersection.isInside = true;
            intersection.isIntersected = true;
        } else if (mind > 0) {
            intersection.t = mind;
            intersection.normal = glm::normalize(rotation * ((no + mind * nd) / radiusSq));
            intersection.isIntersected = true;
        }

        intersection.update(o, d, this);

        return intersection;
    }

    glm::vec3 samplePoint() override {
        float x = sampleNormal();
        float y = sampleNormal();
        float z = sampleNormal();
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

        if (d1 > d2 || d2 < 0) {
            return intersection;
        }

        intersection.isIntersected = true;
        if (d1 < 0) {
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

    glm::vec3 samplePoint() override {
        float wx = size.y * size.z;
        float wy = size.x * size.z;
        float wz = size.x * size.y;
        float u = sampleUniform(0.f, wx + wy + wz);
        float side = 2 * (float(sampleUniform() < 0.5f) - 0.5f);
        float c1 = sampleUniform(0.f, 2.f) - 1.f, c2 = sampleUniform(0.f, 2.f) - 1.f;
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
};

struct InputData {
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t rayDepth = 0;
    uint32_t samples = 0;
    glm::vec3 backgroundColor{0., 0., 0.};

    glm::vec3 cameraPosition{0., 0., 0.};
    glm::vec3 cameraRight{1., 0., 0.};
    glm::vec3 cameraUp{0., 1., 0.};
    glm::vec3 cameraForward{0., 0., 1.};
    glm::vec2 cameraFovTan{0., 0.};

    vector<shared_ptr<Primitive>> primitives;
    vector<Primitive *> lights;
};

struct Distribution {
    glm::vec3 x{0., 0., 0.};
    glm::vec3 n{0., 0., 0.};

    Distribution(glm::vec3 x, glm::vec3 n) : x(x), n(n) {}

    virtual glm::vec3 sample() = 0;
    virtual float pdf(glm::vec3 d) = 0;

    virtual ~Distribution() = default;
};

struct Uniform : Distribution {
    using Distribution::Distribution;

    glm::vec3 sample() override {
        float x = sampleNormal();
        float y = sampleNormal();
        float z = sampleNormal();
        glm::vec3 d{x, y, z};
        return glm::normalize(d);
    }

    float pdf(glm::vec3) override {
        return glm::four_over_pi<float>();
    }
};

struct Cosine : Distribution {
    using Distribution::Distribution;

    glm::vec3 sample() override {
        float x = sampleNormal();
        float y = sampleNormal();
        float z = sampleNormal();
        glm::vec3 d{x, y, z};
        d = glm::normalize(d) + n * (1 + EPS);
        return glm::normalize(d);
    }

    float pdf(glm::vec3 d) override {
        float dn = glm::dot(d, n);
        if (dn < 0.f) {
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

    glm::vec3 sample() override {
        int i = int(sampleUniform(0.f, float(lights->size())) - EPS);
        return glm::normalize((*lights)[i]->samplePoint() - x);
    }

    float pdf(glm::vec3 d) override {
        float ps = 0.f;
        for (auto &light: *lights) {
            if (light == primitive) {
                continue;
            }
            ps += light->pdf(x, d);
        }
        if (ps < 0.f || isnan(ps) || isinf(ps)) {
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

    glm::vec3 sample() override {
        bool c = sampleUniform() < 0.5f;
        if (c) {
            return cosine.sample();
        } else {
            return lightSurface.sample();
        }
    }

    float pdf(glm::vec3 d) override {
        return 0.5f * cosine.pdf(d) + lightSurface.pdf(d);
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
    ifstream inputFile(inputPath);

    InputData inputData;

    string command;
    shared_ptr<Primitive> lastPrimitive = nullptr;
    while (inputFile >> command) {
        if (command == "DIMENSIONS") {
            inputFile >> inputData.width >> inputData.height;
        } else if (command == "RAY_DEPTH") {
            inputFile >> inputData.rayDepth;
            if (inputData.rayDepth > 0) {
                --inputData.rayDepth;
            }
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
    }
    inputData.cameraFovTan.y = inputData.cameraFovTan.x * float(inputData.height) / float(inputData.width);

    return inputData;
}

Intersection intersectScene(const glm::vec3 &o, const glm::vec3 &d, const InputData &inputData) {
    Intersection closestIntersection;
    for (auto &primitive: inputData.primitives) {
        Intersection newIntersection = primitive->intersectPrimitive(o, d);
        if (newIntersection.isIntersected &&
                (!closestIntersection.isIntersected || newIntersection.t < closestIntersection.t)) {
            closestIntersection = newIntersection;
            closestIntersection.primitive = primitive.get();
        }
    }
    return closestIntersection;
}

glm::vec3 applyLight(const Intersection &intersection, const InputData &inputData, const uint32_t &rayDepth);

glm::vec3 getReflectedLight(const Intersection &intersection, const InputData &inputData, const uint32_t &rayDepth) {
    glm::vec3 rd = intersection.d + 2.f * intersection.normal * intersection.nl;
    rd = glm::normalize(rd);
    Intersection reflectionIntersection = intersectScene(
            intersection.p + EPS * intersection.normal, rd, inputData);
    return applyLight(reflectionIntersection, inputData, rayDepth - 1);
}

glm::vec3 applyLightDiffuser(const Intersection &intersection, const InputData &inputData, const uint32_t &rayDepth) {
    Distribution *dis;
    if (inputData.lights.empty()) {
        dis = new Cosine{intersection.p, intersection.normal};
    } else {
        dis = new Mix{intersection.p, intersection.normal, &inputData.lights, intersection.primitive};
    }
    glm::vec3 w = dis->sample();
    float p = dis->pdf(w);
    delete dis;
    float wn = glm::dot(w, intersection.normal);
    if (wn < 0.f || p <= 0.f) {
        return intersection.primitive->emission;
    }
    Intersection nextIntersection = intersectScene(intersection.p + EPS * intersection.normal, w, inputData);
    glm::vec3 l = applyLight(nextIntersection, inputData, rayDepth - 1);
    return intersection.primitive->color * l * glm::one_over_pi<float>() * wn / p + intersection.primitive->emission;
}

glm::vec3 applyLightMetallic(const Intersection &intersection, const InputData &inputData, const uint32_t &rayDepth) {
    glm::vec3 reflectedColor = getReflectedLight(intersection, inputData, rayDepth);
    return intersection.primitive->color * reflectedColor + intersection.primitive->emission;
}

glm::vec3 applyLightDielectric(const Intersection &intersection, const InputData &inputData, const uint32_t &rayDepth) {
    float n1 = 1., n2 = intersection.primitive->ior;
    if (intersection.isInside) {
        n1 = n2;
        n2 = 1.;
    }
    float n12 = n1 / n2;
    float nl = intersection.nl;
    float s = n12 * glm::sqrt(1.f - nl * nl);

    if (s > 1.f) {
        return getReflectedLight(intersection, inputData, rayDepth) + intersection.primitive->emission;
    }

    float r0 = (n1 - n2) / (n1 + n2);
    r0 *= r0;
    float mnl = 1.f - nl;
    float mnlsq = mnl * mnl;
    float r = r0 + (1.f - r0) * mnlsq * mnlsq * mnl;

    if (sampleUniform() < r) {
        return getReflectedLight(intersection, inputData, rayDepth) + intersection.primitive->emission;
    }

    glm::vec3 rd = n12 * intersection.d + (n12 * nl - glm::sqrt(1 - s * s)) * intersection.normal;
    rd = glm::normalize(rd);
    Intersection refractedIntersection = intersectScene(
            intersection.p - EPS * intersection.normal, rd, inputData);
    glm::vec3 refractedColor = applyLight(refractedIntersection, inputData, rayDepth - 1);

    if (!intersection.isInside) {
        refractedColor *= intersection.primitive->color;
    }

    return refractedColor + intersection.primitive->emission;
}

glm::vec3 applyLight(const Intersection &intersection, const InputData &inputData, const uint32_t &rayDepth) {
    if (!intersection.isIntersected) {
        return inputData.backgroundColor;
    }

    glm::vec3 color{0., 0., 0.};
    if (rayDepth == 0) {
        return color;
    }

    if (intersection.primitive->material == Material::DIFFUSER) {
        color = applyLightDiffuser(intersection, inputData, rayDepth);
    } else if (intersection.primitive->material == Material::METALLIC) {
        color = applyLightMetallic(intersection, inputData, rayDepth);
    } else if (intersection.primitive->material == Material::DIELECTRIC) {
        color = applyLightDielectric(intersection, inputData, rayDepth);
    }

    return color;
}

glm::vec3 generateSample(uint32_t px, uint32_t py, const InputData &inputData) {
    float x = (2 * (float(px) + sampleUniform()) / float(inputData.width) - 1) * inputData.cameraFovTan.x;
    float y = -(2 * (float(py) + sampleUniform()) / float(inputData.height) - 1) * inputData.cameraFovTan.y;
    glm::vec3 d = glm::normalize(x * inputData.cameraRight + y * inputData.cameraUp + inputData.cameraForward);
    Intersection intersection = intersectScene(inputData.cameraPosition, d, inputData);
    return applyLight(intersection, inputData, inputData.rayDepth);
}

glm::vec3 generatePixel(uint32_t px, uint32_t py, const InputData &inputData) {
    glm::vec3 color{0., 0., 0.};
    for (uint32_t i = 0; i < inputData.samples; ++i) {
        color += generateSample(px, py, inputData);
    }
    color /= inputData.samples;
    color = aces_tonemap(color);
    float p = 1.f / 2.2f;
    color = {glm::pow(color.x, p), glm::pow(color.y, p), glm::pow(color.z, p)};
    return color;
}

SceneFloat generateScene(const InputData &inputData) {
    SceneFloat scene(inputData.width, inputData.height);
    for (uint32_t i = 0; i < inputData.height; ++i) {
        for (uint32_t j = 0; j < inputData.width; ++j) {
            scene.data.push_back(generatePixel(j, i, inputData));
        }
    }
    return scene;
}

int main(int, char *argv[]) {
    clock_t start = clock();

    string inputPath = argv[1];
    string outputPath = argv[2];

    InputData inputData = parseInput(inputPath);
    SceneFloat scene = generateScene(inputData);

    printImage(scene, outputPath);

    cout << "Finished after " << float(clock() - start) / float(CLOCKS_PER_SEC) << "s";
    return 0;
}
