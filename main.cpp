#include <array>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

using namespace std;

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

struct Primitive {
    glm::vec3 position{0., 0., 0.};
    glm::quat rotation{1., 0., 0., 0.};
    glm::vec3 color{0., 0., 0.};

    virtual bool intersectPrimitive(const glm::vec3 &o, const glm::vec3 &d, float &distance) = 0;
};

struct Plane : Primitive {
    glm::vec3 normal{0., 0., 1.};

    bool intersectPrimitive(const glm::vec3 &o, const glm::vec3 &d, float &distance) override;
};

struct Ellipsoid : Primitive {
    glm::vec3 radius{1., 1., 1.};

    bool intersectPrimitive(const glm::vec3 &o, const glm::vec3 &d, float &distance) override;
};

struct Box : Primitive {
    glm::vec3 size{1., 1., 1.};

    bool intersectPrimitive(const glm::vec3 &o, const glm::vec3 &d, float &distance) override;
};

struct InputData {
    uint32_t width = 0;
    uint32_t height = 0;
    glm::vec3 backgroundColor{0., 0., 0.};
    glm::vec3 cameraPosition{0., 0., 0.};
    glm::vec3 cameraRight{1., 0., 0.};
    glm::vec3 cameraUp{0., 1., 0.};
    glm::vec3 cameraForward{0., 0., 1.};
    glm::vec2 cameraFovTan{0., 0.};
    vector<shared_ptr<Primitive>> primitives;
};

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
        }
    }
    inputFile.close();
    if (lastPrimitive != nullptr) {
        inputData.primitives.push_back(lastPrimitive);
    }
    inputData.cameraFovTan.y = inputData.cameraFovTan.x * float(inputData.height) / float(inputData.width);

    return inputData;
}

bool Plane::intersectPrimitive(const glm::vec3 &o, const glm::vec3 &d, float &distance) {
    glm::vec3 no = this->rotation * (o - this->position);
    glm::vec3 nd = this->rotation * d;
    distance = -glm::dot(no, this->normal) / glm::dot(nd, this->normal);
    return distance > 0;
}

bool Ellipsoid::intersectPrimitive(const glm::vec3 &o, const glm::vec3 &d, float &distance) {
    glm::vec3 no = this->rotation * (o - this->position);
    glm::vec3 nd = this->rotation * d;

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

    float discr = b * b - 4 * a * c;
    if (discr < 0) {
        return false;
    }
    float sqrtDiscr = sqrt(discr);

    float mind = (-b - sqrtDiscr) / (2 * a);
    float maxd = (-b + sqrtDiscr) / (2 * a);

    if (maxd < 0) {
        return false;
    } else if (mind < 0) {
        distance = maxd;
        return true;
    } else {
        distance = mind;
        return true;
    }
}

bool Box::intersectPrimitive(const glm::vec3 &o, const glm::vec3 &d, float &distance) {
    glm::vec3 no = this->rotation * (o - this->position);
    glm::vec3 nd = this->rotation * d;

    glm::vec3 t1 = (this->size - no) / nd;
    glm::vec3 t2 = (-this->size - no) / nd;

    float d1 = max(max(min(t1.x, t2.x), min(t1.y, t2.y)), min(t1.z, t2.z));
    float d2 = min(min(max(t1.x, t2.x), max(t1.y, t2.y)), max(t1.z, t2.z));

    if (d1 > d2) {
        return false;
    }

    if (d2 < 0) {
        return false;
    } else if (d1 < 0) {
        distance = d2;
        return true;
    } else {
        distance = d1;
        return true;
    }
}

glm::vec3 intersectScene(const glm::vec3 &o, const glm::vec3 &d,
                         const vector<shared_ptr<Primitive>> &primitives, const glm::vec3 &backgroundColor) {
    shared_ptr<Primitive> closestPrimitive = nullptr;
    float minDistance = -1.;
    for (auto &primitive: primitives) {
        float distance;
        if (primitive->intersectPrimitive(o, d, distance) && (minDistance < 0 || distance < minDistance)) {
            minDistance = distance;
            closestPrimitive = primitive;
        }
    }
    if (closestPrimitive != nullptr) {
        return closestPrimitive->color;
    }
    return backgroundColor;
}

glm::vec3 generateRay(uint32_t px, uint32_t py, const InputData &inputData) {
    float x = (2 * (float(px) + 0.5f) / float(inputData.width) - 1) * inputData.cameraFovTan.x;
    float y = -(2 * (float(py) + 0.5f) / float(inputData.height) - 1) * inputData.cameraFovTan.y;
    glm::vec3 d = x * inputData.cameraRight + y * inputData.cameraUp + inputData.cameraForward;
    return intersectScene(inputData.cameraPosition, d, inputData.primitives, inputData.backgroundColor);
}

SceneFloat generateScene(const InputData &inputData) {
    SceneFloat scene(inputData.width, inputData.height);
    for (uint32_t i = 0; i < inputData.height; ++i) {
        for (uint32_t j = 0; j < inputData.width; ++j) {
            scene.data.emplace_back(generateRay(j, i, inputData));
        }
    }
    return scene;
}

int main(int, char *argv[]) {
    string inputPath = argv[1];
    string outputPath = argv[2];

    InputData inputData = parseInput(inputPath);
    SceneFloat scene = generateScene(inputData);

    printImage(scene, outputPath);
    return 0;
}
