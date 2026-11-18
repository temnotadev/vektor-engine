#include "physics.hpp"

#include <Jolt/Jolt.h>
#include <Jolt/Core/Factory.h>
#include <Jolt/Core/JobSystemThreadPool.h>
#include <Jolt/Core/TempAllocator.h>
#include <Jolt/Physics/Body/BodyCreationSettings.h>
#include <Jolt/Physics/Body/BodyInterface.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Physics/PhysicsSystem.h>
#include <Jolt/RegisterTypes.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <map>
#include <set>
#include <thread>

namespace {
constexpr JPH::ObjectLayer kStaticLayer = 0;
constexpr JPH::ObjectLayer kMovingLayer = 1;
constexpr JPH::uint kMaxBodies = 4096;
constexpr float kFixedStep = 1.0f / 60.0f;

int physicsThreadCount() {
    const unsigned int cores = std::thread::hardware_concurrency();
    return static_cast<int>(std::min(8u, std::max(1u, cores > 1 ? cores - 1 : 1u)));
}

class BroadPhaseLayers final : public JPH::BroadPhaseLayerInterface {
public:
    JPH::uint GetNumBroadPhaseLayers() const override { return 2; }

    JPH::BroadPhaseLayer GetBroadPhaseLayer(JPH::ObjectLayer layer) const override {
        return JPH::BroadPhaseLayer(layer);
    }

#if defined(JPH_EXTERNAL_PROFILE) || defined(JPH_PROFILE_ENABLED)
    const char* GetBroadPhaseLayerName(JPH::BroadPhaseLayer layer) const override {
        return layer.GetValue() == 0 ? "Static" : "Moving";
    }
#endif
};

class ObjectVsBroadPhase final : public JPH::ObjectVsBroadPhaseLayerFilter {
public:
    bool ShouldCollide(JPH::ObjectLayer, JPH::BroadPhaseLayer) const override {
        return true;
    }
};

class ObjectLayerPairs final : public JPH::ObjectLayerPairFilter {
public:
    bool ShouldCollide(JPH::ObjectLayer, JPH::ObjectLayer) const override {
        return true;
    }
};

JPH::Quat toJoltRotation(const float degrees[3]) {
    constexpr float toRadians = JPH::DegreesToRadians(1.0f);
    const JPH::Quat x = JPH::Quat::sRotation(JPH::Vec3::sAxisX(), degrees[0] * toRadians);
    const JPH::Quat y = JPH::Quat::sRotation(JPH::Vec3::sAxisY(), degrees[1] * toRadians);
    const JPH::Quat z = JPH::Quat::sRotation(JPH::Vec3::sAxisZ(), degrees[2] * toRadians);
    return z * y * x;
}

bool near(float a, float b) {
    return std::fabs(a - b) < 1.0e-4f;
}

struct JoltRuntime {
    JoltRuntime() {
        JPH::RegisterDefaultAllocator();
        JPH::Factory::sInstance = new JPH::Factory();
        JPH::RegisterTypes();
    }

    ~JoltRuntime() {
        JPH::UnregisterTypes();
        delete JPH::Factory::sInstance;
        JPH::Factory::sInstance = nullptr;
    }
};

JoltRuntime& getJoltRuntime() {
    static JoltRuntime runtime;
    return runtime;
}
}

struct PhysicsWorld::Impl {
    struct Body {
        JPH::BodyID id;
        std::array<float, 3> size{};
        float mass = 1.0f;
        float friction = 0.2f;
        float restitution = 0.0f;
        bool dynamic = false;
        std::array<float, 3> position{};
        std::array<float, 3> rotation{};
    };

    JoltRuntime& runtime;
    BroadPhaseLayers broadPhaseLayers;
    ObjectVsBroadPhase objectVsBroadPhase;
    ObjectLayerPairs objectLayerPairs;
    JPH::PhysicsSystem system;
    JPH::TempAllocatorImpl tempAllocator{10 * 1024 * 1024};
    JPH::JobSystemThreadPool jobSystem{
        JPH::cMaxPhysicsJobs, JPH::cMaxPhysicsBarriers, physicsThreadCount()};
    std::map<std::string, Body> bodies;
    float accumulator = 0.0f;

    Impl() : runtime(getJoltRuntime()) {
        system.Init(kMaxBodies, 0, 16384, 10240, broadPhaseLayers,
                    objectVsBroadPhase, objectLayerPairs);
        system.SetGravity(JPH::Vec3(0.0f, -9.81f, 0.0f));
    }

    ~Impl() {
        clearBodies();
    }

    void clearBodies() {
        while (!bodies.empty()) remove(bodies.begin()->first);
        accumulator = 0.0f;
    }

    void remove(const std::string& name) {
        auto it = bodies.find(name);
        if (it == bodies.end()) return;
        system.GetBodyInterface().RemoveBody(it->second.id);
        system.GetBodyInterface().DestroyBody(it->second.id);
        bodies.erase(it);
    }

    bool matches(const Body& body, const PhysicsBodyDesc& desc) const {
        if (body.dynamic != desc.dynamic || !near(body.mass, desc.mass) ||
            !near(body.friction, desc.friction) || !near(body.restitution, desc.restitution))
            return false;
        for (int i = 0; i < 3; ++i)
            if (!near(body.size[i], std::fabs(desc.size[i]))) return false;
        return true;
    }

    void create(const PhysicsBodyDesc& desc) {
        JPH::Vec3 halfSize(
            std::max(std::fabs(desc.size[0]) * 0.5f, 0.001f),
            std::max(std::fabs(desc.size[1]) * 0.5f, 0.001f),
            std::max(std::fabs(desc.size[2]) * 0.5f, 0.001f));
        JPH::BoxShapeSettings shapeSettings(halfSize);
        JPH::ShapeSettings::ShapeResult shapeResult = shapeSettings.Create();
        if (shapeResult.HasError()) {
            std::fprintf(stderr, "[physics] cannot create collider for '%s': %s\n",
                         desc.name.c_str(), shapeResult.GetError().c_str());
            return;
        }

        const JPH::EMotionType motion = desc.dynamic ? JPH::EMotionType::Dynamic : JPH::EMotionType::Static;
        const JPH::ObjectLayer layer = desc.dynamic ? kMovingLayer : kStaticLayer;
        JPH::BodyCreationSettings settings(shapeResult.Get(),
            JPH::RVec3(desc.position[0], desc.position[1], desc.position[2]),
            toJoltRotation(desc.rotation), motion, layer);
        settings.mFriction = std::max(desc.friction, 0.0f);
        settings.mRestitution = std::clamp(desc.restitution, 0.0f, 1.0f);
        if (desc.dynamic) {
            settings.mOverrideMassProperties = JPH::EOverrideMassProperties::CalculateInertia;
            settings.mMassPropertiesOverride.mMass = std::max(desc.mass, 0.001f);
        }

        JPH::BodyInterface& bodyInterface = system.GetBodyInterface();
        JPH::Body* body = bodyInterface.CreateBody(settings);
        if (!body) {
            std::fprintf(stderr, "[physics] cannot create body '%s'\n", desc.name.c_str());
            return;
        }
        const JPH::BodyID id = body->GetID();
        bodyInterface.AddBody(id, desc.dynamic ? JPH::EActivation::Activate : JPH::EActivation::DontActivate);

        Body record;
        record.id = id;
        for (int i = 0; i < 3; ++i) {
            record.size[i] = std::fabs(desc.size[i]);
            record.position[i] = desc.position[i];
            record.rotation[i] = desc.rotation[i];
        }
        record.mass = desc.mass;
        record.friction = desc.friction;
        record.restitution = desc.restitution;
        record.dynamic = desc.dynamic;
        bodies.emplace(desc.name, record);
    }

    void setTransform(Body& body, const PhysicsBodyDesc& desc) {
        body.position = {desc.position[0], desc.position[1], desc.position[2]};
        body.rotation = {desc.rotation[0], desc.rotation[1], desc.rotation[2]};
        system.GetBodyInterface().SetPositionAndRotation(
            body.id, JPH::RVec3(desc.position[0], desc.position[1], desc.position[2]),
            toJoltRotation(desc.rotation),
            body.dynamic ? JPH::EActivation::Activate : JPH::EActivation::DontActivate);
    }

    void sync(const std::vector<PhysicsBodyDesc>& descs) {
        std::set<std::string> active;
        for (const PhysicsBodyDesc& desc : descs) {
            active.insert(desc.name);
            auto it = bodies.find(desc.name);
            if (it != bodies.end() && !matches(it->second, desc)) {
                remove(desc.name);
                it = bodies.end();
            }
            if (it == bodies.end()) {
                create(desc);
                it = bodies.find(desc.name);
            } else {
                Body& body = it->second;
                bool moved = false;
                for (int i = 0; i < 3; ++i)
                    moved |= !near(body.position[i], desc.position[i]) ||
                             !near(body.rotation[i], desc.rotation[i]);
                if (moved) setTransform(body, desc);
            }
        }

        for (auto it = bodies.begin(); it != bodies.end();) {
            if (!active.count(it->first)) {
                const std::string name = it->first;
                ++it;
                remove(name);
            } else {
                ++it;
            }
        }
    }

    void writeTransforms(std::vector<PhysicsTransform>& transforms) {
        transforms.clear();
        JPH::BodyInterface& bodyInterface = system.GetBodyInterface();
        for (auto& [name, body] : bodies) {
            if (!body.dynamic) continue;
            JPH::RVec3 position;
            JPH::Quat rotation;
            bodyInterface.GetPositionAndRotation(body.id, position, rotation);
            const JPH::Vec3 euler = rotation.GetEulerAngles() / JPH::DegreesToRadians(1.0f);
            body.position = {float(position.GetX()), float(position.GetY()), float(position.GetZ())};
            body.rotation = {euler.GetX(), euler.GetY(), euler.GetZ()};
            PhysicsTransform transform;
            transform.name = name;
            for (int i = 0; i < 3; ++i) {
                transform.position[i] = body.position[i];
                transform.rotation[i] = body.rotation[i];
            }
            transforms.push_back(transform);
        }
    }
};

PhysicsWorld::PhysicsWorld() : impl(std::make_unique<Impl>()) {}
PhysicsWorld::~PhysicsWorld() = default;

void PhysicsWorld::clear() {
    if (!impl) return;
    impl->clearBodies();
}

void PhysicsWorld::update(const std::vector<PhysicsBodyDesc>& bodies, float dt,
                          std::vector<PhysicsTransform>& transforms) {
    impl->sync(bodies);
    const bool hasDynamicBody = std::any_of(impl->bodies.begin(), impl->bodies.end(),
        [](const auto& entry) { return entry.second.dynamic; });
    if (!hasDynamicBody) {
        transforms.clear();
        impl->accumulator = 0.0f;
        return;
    }
    impl->accumulator = std::min(impl->accumulator + std::clamp(dt, 0.0f, 0.1f), 0.1f);
    while (impl->accumulator >= kFixedStep) {
        impl->system.Update(kFixedStep, 1, &impl->tempAllocator, &impl->jobSystem);
        impl->accumulator -= kFixedStep;
    }
    impl->writeTransforms(transforms);
}
