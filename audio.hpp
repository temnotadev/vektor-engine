// Simple game audio system built on miniaudio.
//
// Supported formats: MP3, WAV, OGG, FLAC (whatever miniaudio is configured
// with; MP3 is always available in the vendored header).
//
// The public interface intentionally stays tiny and pimpl-based so including
// this header from world.hpp / scene.hpp does not pull in miniaudio itself.
#pragma once

#include <memory>
#include <string>

class AudioSystem {
public:
    AudioSystem();
    ~AudioSystem();

    AudioSystem(const AudioSystem&) = delete;
    AudioSystem& operator=(const AudioSystem&) = delete;

    // Creates the playback device/engine. Safe to call more than once.
    bool init();
    void shutdown();
    bool ready() const;

    // Loads (or reloads) a clip under an id. Re-loading the same id with a
    // different path unloads the previous clip first. Returns false when the
    // file cannot be decoded or audio is unavailable.
    bool load(const std::string& id, const std::string& path);
    void unload(const std::string& id);
    void unloadAll();
    bool exists(const std::string& id) const;

    // Starts the clip from the beginning. If it is already playing it is
    // restarted. `loop` controls looping, `volume` is 0..1 and `pan` is -1..1.
    bool play(const std::string& id, bool loop = false, float volume = 1.0f, float pan = 0.0f);
    bool pause(const std::string& id);
    bool resume(const std::string& id);
    bool stop(const std::string& id);
    bool isPlaying(const std::string& id) const;
    bool isPaused(const std::string& id) const;

    void setVolume(const std::string& id, float volume);
    void setPan(const std::string& id, float pan);
    void setLooping(const std::string& id, bool loop);

    void setMasterVolume(float volume);
    float masterVolume() const;

    void stopAll();
    void pauseAll();
    void resumeAll();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
