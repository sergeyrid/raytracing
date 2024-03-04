#include <array>
#include <ctime>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

using namespace std;

const float EPS = 0.001f;

struct ColorInt {
    uint8_t red;
    uint8_t green;
    uint8_t blue;
};

struct SceneInt {
    uint32_t width;
    uint32_t height;
    vector<ColorInt> data;

    SceneInt(const uint32_t &width, const uint32_t &height) {
        this->width = width;
        this->height = height;
        this->data = vector<ColorInt>();
        this->data.reserve(width * height);
    }
};

struct SceneFloat {
    uint32_t width;
    uint32_t height;
    vector<glm::vec3> data;

    SceneFloat(const uint32_t &width, const uint32_t &height) {
        this->width = width;
        this->height = height;
        this->data = vector<glm::vec3>();
        this->data.reserve(width * height);
    }
};

enum class Material {
    DIFFUSER,
    METALLIC,
    DIELECTRIC,
};

struct Intersection {
    bool isIntersected = false;
    bool isInside = false;

    float t = 0.;
    glm::vec3 p{0., 0., 0.};
    glm::vec3 d{0., 0., 0.};

    glm::vec3 normal{0., 0., 0.};
    glm::vec3 color{0., 0., 0.};
    Material material = Material::DIFFUSER;
    float ior = 1.;
};

struct Primitive {
    glm::vec3 position{0., 0., 0.};
    glm::quat rotation{1., 0., 0., 0.};
    glm::vec3 color{0., 0., 0.};
    Material material = Material::DIFFUSER;
    float ior = 1.;

    virtual Intersection intersectPrimitive(const glm::vec3 &o, const glm::vec3 &d) = 0;
};

struct Plane : Primitive {
    glm::vec3 normal{0., 0., 1.};

    Intersection intersectPrimitive(const glm::vec3 &o, const glm::vec3 &d) override;
};

struct Ellipsoid : Primitive {
    glm::vec3 radius{1., 1., 1.};

    Intersection intersectPrimitive(const glm::vec3 &o, const glm::vec3 &d) override;
};

struct Box : Primitive {
    glm::vec3 size{1., 1., 1.};

    Intersection intersectPrimitive(const glm::vec3 &o, const glm::vec3 &d) override;
};

struct Light {
    glm::vec3 intensity {0., 0., 0.};
    glm::vec3 direction{0., 0., 0.};
    glm::vec3 position{0., 0., 0.};
    glm::vec3 attenuation{0., 0., 0.};
    bool isDirected = false;
};

struct InputData {
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t rayDepth = 0;
    glm::vec3 backgroundColor{0., 0., 0.};
    glm::vec3 ambientLight{0., 0., 0.};

    glm::vec3 cameraPosition{0., 0., 0.};
    glm::vec3 cameraRight{1., 0., 0.};
    glm::vec3 cameraUp{0., 1., 0.};
    glm::vec3 cameraForward{0., 0., 1.};
    glm::vec2 cameraFovTan{0., 0.};

    vector<shared_ptr<Primitive>> primitives;
    vector<shared_ptr<Light>> lights;
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

glm::vec3 colorIntToFloat(ColorInt &color) {
    return glm::vec3 {
        float(color.red) / 255,
        float(color.green) / 255,
        float(color.blue) / 255,
    };
}

SceneInt sceneFloatToInt(SceneFloat &scene) {
    SceneInt sceneInt(scene.width, scene.height);
    for (auto &pixel: scene.data) {
        sceneInt.data.push_back(colorFloatToInt(pixel));
    }
    return sceneInt;
}

SceneFloat sceneIntToFloat(SceneInt &scene) {
    SceneFloat sceneFloat(scene.width, scene.height);
    for (auto &pixel: scene.data) {
        sceneFloat.data.push_back(colorIntToFloat(pixel));
    }
    return sceneFloat;
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
        } else if (command == "BG_COLOR") {
            inputFile >> inputData.backgroundColor.x >> inputData.backgroundColor.y >> inputData.backgroundColor.z;
        } else if (command == "AMBIENT_LIGHT") {
            inputFile >> inputData.ambientLight.x >> inputData.ambientLight.y >> inputData.ambientLight.z;
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
            inputData.cameraFovTan.x = tan(fovX / 2);
        } else if (command == "NEW_PRIMITIVE" && lastPrimitive != nullptr) {
            inputData.primitives.push_back(lastPrimitive);
            lastPrimitive = nullptr;
        } else if (command == "PLANE") {
            std::shared_ptr<Plane> newPlane(new Plane());
            inputFile >> newPlane->normal.x >> newPlane->normal.y >> newPlane->normal.z;
            lastPrimitive = newPlane;
        } else if (command == "ELLIPSOID") {
            std::shared_ptr<Ellipsoid> newEllipsoid(new Ellipsoid());
            inputFile >> newEllipsoid->radius.x >> newEllipsoid->radius.y >> newEllipsoid->radius.z;
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
        } else if (command == "COLOR") {
            inputFile >> lastPrimitive->color.x >> lastPrimitive->color.y >> lastPrimitive->color.z;
        } else if (command == "METALLIC") {
            lastPrimitive->material = Material::METALLIC;
        } else if (command == "DIELECTRIC") {
            lastPrimitive->material = Material::DIELECTRIC;
        } else if (command == "IOR") {
            inputFile >> lastPrimitive->ior;
        }else if (command == "NEW_LIGHT") {
                inputData.lights.emplace_back(new Light());
        } else if (command == "LIGHT_INTENSITY") {
            Light &light = *inputData.lights.back();
            inputFile >> light.intensity.x >> light.intensity.y >> light.intensity.z;
        } else if (command == "LIGHT_DIRECTION") {
            Light &light = *inputData.lights.back();
            light.isDirected = true;
            inputFile >> light.direction.x >> light.direction.y >> light.direction.z;
        } else if (command == "LIGHT_POSITION") {
            Light &light = *inputData.lights.back();
            inputFile >> light.position.x >> light.position.y >> light.position.z;
        } else if (command == "LIGHT_ATTENUATION") {
            Light &light = *inputData.lights.back();
            inputFile >> light.attenuation.x >> light.attenuation.y >> light.attenuation.z;
        }
    }
    inputFile.close();
    if (lastPrimitive != nullptr) {
        inputData.primitives.push_back(lastPrimitive);
    }
    inputData.cameraFovTan.y = inputData.cameraFovTan.x * float(inputData.height) / float(inputData.width);

    return inputData;
}

Intersection Plane::intersectPrimitive(const glm::vec3 &o, const glm::vec3 &d) {
    glm::quat conj = glm::conjugate(this->rotation);
    glm::vec3 no = conj * (o - this->position);
    glm::vec3 nd = conj * d;

    Intersection intersection;
    intersection.normal = glm::normalize(this->rotation * this->normal);
    intersection.color = this->color;
    intersection.material = this->material;
    intersection.ior = this->ior;
    intersection.t = -glm::dot(no, this->normal) / glm::dot(nd, this->normal);
    intersection.p = o + intersection.t * d;
    intersection.isIntersected = intersection.t > 0;
    return intersection;
}

Intersection Ellipsoid::intersectPrimitive(const glm::vec3 &o, const glm::vec3 &d) {
    glm::quat conj = glm::conjugate(this->rotation);
    glm::vec3 no = conj * (o - this->position);
    glm::vec3 nd = conj * d;

    glm::vec3 ndr = nd / this->radius;
    glm::vec3 nor = no / this->radius;

    float a = glm::dot(ndr, ndr);
    float b = 2 * glm::dot(nor, ndr);
    float c = glm::dot(nor, nor) - 1;

    if (a < 0) {
        a = -a;
        b = -b;
        c = -c;
    }

    Intersection intersection;

    float discr = b * b - 4 * a * c;
    if (discr < 0) {
        return intersection;
    }
    float sqrtDiscr = sqrt(discr);

    float mind = (-b - sqrtDiscr) / (2 * a);
    float maxd = (-b + sqrtDiscr) / (2 * a);

    intersection.color = this->color;
    intersection.material = this->material;
    intersection.ior = this->ior;
    if (mind < 0 && maxd > 0) {
        intersection.t = maxd;
        intersection.normal = -glm::normalize(this->rotation * ((no + maxd * nd) / this->radius));
        intersection.isInside = true;
        intersection.isIntersected = true;
    } else if (mind > 0) {
        intersection.t = mind;
        intersection.normal = glm::normalize(this->rotation * ((no + mind * nd) / this->radius));
        intersection.isIntersected = true;
    }
    intersection.p = o + intersection.t * d;
    return intersection;
}

Intersection Box::intersectPrimitive(const glm::vec3 &o, const glm::vec3 &d) {
    glm::quat conj = glm::conjugate(this->rotation);
    glm::vec3 no = conj * (o - this->position);
    glm::vec3 nd = conj * d;

    glm::vec3 t1 = (this->size - no) / nd;
    glm::vec3 t2 = (-this->size - no) / nd;

    float d1 = max(max(min(t1.x, t2.x), min(t1.y, t2.y)), min(t1.z, t2.z));
    float d2 = min(min(max(t1.x, t2.x), max(t1.y, t2.y)), max(t1.z, t2.z));

    Intersection intersection;

    if (d1 > d2 || d2 < 0) {
        return intersection;
    }

    intersection.isIntersected = true;
    intersection.color = this->color;
    intersection.material = this->material;
    intersection.ior = this->ior;
    if (d1 < 0) {
        intersection.t = d2;
        intersection.isInside = true;
    } else {
        intersection.t = d1;
    }
    intersection.p = o + intersection.t * d;

    glm::vec3 ps = (no + intersection.t * nd) / this->size;
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
    intersection.normal = this->rotation * intersection.normal;
    if (intersection.isInside) {
        intersection.normal = -intersection.normal;
    }

    return intersection;
}

Intersection intersectScene(const glm::vec3 &o, const glm::vec3 &d, const InputData &inputData) {
    Intersection closestIntersection;
    for (auto &primitive: inputData.primitives) {
        Intersection newIntersection = primitive->intersectPrimitive(o, d);
        if (newIntersection.isIntersected &&
                (!closestIntersection.isIntersected || newIntersection.t < closestIntersection.t)) {
            closestIntersection = newIntersection;
        }
    }
    if (!closestIntersection.isIntersected) {
        closestIntersection.color = inputData.backgroundColor;
    }
    closestIntersection.d = glm::normalize(d);
    return closestIntersection;
}

glm::vec3 applyLight(const Intersection &intersection, const InputData &inputData, const uint32_t &rayDepth);

glm::vec3 getReflectedLight(const Intersection &intersection, const InputData &inputData, const uint32_t &rayDepth) {
    glm::vec3 rd = intersection.d - 2.f * intersection.normal * glm::dot(intersection.normal, intersection.d);
    Intersection reflectionIntersection = intersectScene(intersection.p + EPS * rd, rd, inputData);
    return applyLight(reflectionIntersection, inputData, rayDepth - 1);
}

glm::vec3 getLight(const Intersection &intersection, const Light &light, const InputData &inputData) {
    glm::vec3 lightDirection;
    float r = 0.;
    if (light.isDirected) {
        lightDirection = light.direction;
    } else {
        lightDirection = light.position - intersection.p;
        r = glm::length(lightDirection);
    }
    lightDirection = glm::normalize(lightDirection);

    Intersection shadowIntersection = intersectScene(
            intersection.p + EPS * lightDirection, lightDirection, inputData);
    if (shadowIntersection.isIntersected &&
            (light.isDirected || glm::length(shadowIntersection.p - intersection.p) < r)) {
        return {0., 0., 0.};
    }

    glm::vec3 color = intersection.color;

    color *= glm::dot(lightDirection, intersection.normal);
    color = light.intensity * color;
    if (!light.isDirected) {
        color /= glm::dot(light.attenuation, glm::vec3 {1.f, r, r * r});
    }

    return color;
}

glm::vec3 applyLightDiffuser(const Intersection &intersection, const InputData &inputData, const uint32_t &rayDepth) {
    glm::vec3 color = intersection.color;
    color *= inputData.ambientLight;
    for (auto &light: inputData.lights) {
        color += getLight(intersection, *light, inputData);
    }
    return color;
}

glm::vec3 applyLightMetallic(const Intersection &intersection, const InputData &inputData, const uint32_t &rayDepth) {
    glm::vec3 reflectedColor = getReflectedLight(intersection, inputData, rayDepth);
    return intersection.color * reflectedColor;
}

glm::vec3 applyLightDielectric(const Intersection &intersection, const InputData &inputData, const uint32_t &rayDepth) {
    glm::vec3 reflectedColor = getReflectedLight(intersection, inputData, rayDepth);

    float n1 = 1., n2 = intersection.ior;
    if (intersection.isInside) {
        n1 = n2;
        n2 = 1.;
    }
    float nl = -glm::dot(intersection.normal, intersection.d);
    float n12 = n1 / n2;
    float s = n12 * sqrt(1.f - nl * nl);

    glm::vec3 refractedColor{0., 0., 0.};
    float r1 = 1.;
    if (s <= 1.f) {
        float r0 = (n1 - n2) / (n1 + n2);
        r0 *= r0;
        r1 = r0 + (1.f - r0) * powf((1.f - nl), 5);

        glm::vec3 rd = n12 * intersection.d + (n12 * nl - sqrt(1 - s * s)) * intersection.normal;
        Intersection refractedIntersection = intersectScene(intersection.p + EPS * rd, rd, inputData);
        refractedColor = applyLight(refractedIntersection, inputData, rayDepth - 1);
    }

    return (1.f - r1) * intersection.color * refractedColor + r1 * reflectedColor;
}

glm::vec3 applyLight(const Intersection &intersection, const InputData &inputData, const uint32_t &rayDepth) {
    if (!intersection.isIntersected) {
        return inputData.backgroundColor;
    }
    glm::vec3 color{0., 0., 0.};
    if (intersection.material == Material::DIFFUSER) {
        color = applyLightDiffuser(intersection, inputData, rayDepth);
    } else if (intersection.material == Material::METALLIC && rayDepth > 0) {
        color = applyLightMetallic(intersection, inputData, rayDepth);
    } else if (intersection.material == Material::DIELECTRIC && rayDepth > 0) {
        color = applyLightDielectric(intersection, inputData, rayDepth);
    }
    return color;
}

glm::vec3 generatePixel(uint32_t px, uint32_t py, const InputData &inputData) {
    float x = (2 * (float(px) + 0.5f) / float(inputData.width) - 1) * inputData.cameraFovTan.x;
    float y = -(2 * (float(py) + 0.5f) / float(inputData.height) - 1) * inputData.cameraFovTan.y;
    glm::vec3 d = x * inputData.cameraRight + y * inputData.cameraUp + inputData.cameraForward;
    Intersection intersection = intersectScene(inputData.cameraPosition, d, inputData);
    glm::vec3 color = applyLight(intersection, inputData, inputData.rayDepth);
    color = aces_tonemap(color);
    float p = 1.f / 2.2f;
    color = {powf(color.x, p), powf(color.y, p), powf(color.z, p)};
    return color;
}

SceneFloat generateScene(const InputData &inputData) {
    SceneFloat scene(inputData.width, inputData.height);
    for (uint32_t i = 0; i < inputData.height; ++i) {
        for (uint32_t j = 0; j < inputData.width; ++j) {
            if (j % 4 != 0) {
                scene.data.push_back(scene.data.back());
            } else if (i % 4 != 0) {
                scene.data.push_back(scene.data[scene.data.size() - inputData.width]);
            } else {
                scene.data.push_back(generatePixel(j, i, inputData));
            }
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
