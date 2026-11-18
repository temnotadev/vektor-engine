#pragma once

#include <memory>
#include <string>
#include <vector>

struct PhysicsBodyDesc {
    std::string name;
    float position[3] = {};
    float rotation[3] = {};
    float size[3] = {1.0f, 1.0f, 1.0f};
    float mass = 1.0f;
    float friction = 0.2f;
    float restitution = 0.0f;
    bool dynamic = false;
};

struct PhysicsTransform {
    std::string name;
    float position[3] = {};
    float rotation[3] = {};
};

class PhysicsWorld {
public:
    PhysicsWorld();
    ~PhysicsWorld();

    PhysicsWorld(const PhysicsWorld&) = delete;
    PhysicsWorld& operator=(const PhysicsWorld&) = delete;

    void clear();
    void update(const std::vector<PhysicsBodyDesc>& bodies, float dt,
                std::vector<PhysicsTransform>& transforms);

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};
