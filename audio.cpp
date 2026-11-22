#include "audio.hpp"

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif

#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <unordered_map>

struct AudioSystem::Impl {
    struct Voice {
        ma_sound sound{};
        bool loaded = false;
        bool paused = false;
        std::string path;
    };

    ma_engine engine{};
    bool engineReady = false;
    float master = 1.0f;
    std::unordered_map<std::string, std::unique_ptr<Voice>> voices;

    Voice* find(const std::string& id) {
        auto it = voices.find(id);
        return it == voices.end() ? nullptr : it->second.get();
    }

    const Voice* find(const std::string& id) const {
        auto it = voices.find(id);
        return it == voices.end() ? nullptr : it->second.get();
    }

    void destroyVoice(const std::string& id) {
        auto it = voices.find(id);
        if (it == voices.end()) return;
        if (it->second->loaded) {
            ma_sound_stop(&it->second->sound);
            ma_sound_uninit(&it->second->sound);
            it->second->loaded = false;
        }
        voices.erase(it);
    }
};

AudioSystem::AudioSystem() : impl_(new Impl()) {}

AudioSystem::~AudioSystem() {
    shutdown();
}

bool AudioSystem::init() {
    if (impl_->engineReady) return true;

    ma_engine_config config = ma_engine_config_init();
    ma_result result = ma_engine_init(&config, &impl_->engine);
    if (result != MA_SUCCESS) {
        std::fprintf(stderr, "audio: failed to initialise playback device (%d)\n", (int)result);
        return false;
    }

    impl_->engineReady = true;
    ma_engine_set_volume(&impl_->engine, impl_->master);
    return true;
}

void AudioSystem::shutdown() {
    if (!impl_) return;
    unloadAll();
    if (impl_->engineReady) {
        ma_engine_uninit(&impl_->engine);
        impl_->engineReady = false;
    }
}

bool AudioSystem::ready() const {
    return impl_ && impl_->engineReady;
}

bool AudioSystem::load(const std::string& id, const std::string& path) {
    if (!impl_ || !impl_->engineReady || id.empty() || path.empty()) return false;

    Impl::Voice* existing = impl_->find(id);
    if (existing && existing->loaded && existing->path == path) return true;

    impl_->destroyVoice(id);

    auto voice = std::make_unique<Impl::Voice>();
    voice->path = path;

    const ma_uint32 flags = MA_SOUND_FLAG_DECODE | MA_SOUND_FLAG_NO_SPATIALIZATION;
    ma_result result = ma_sound_init_from_file(&impl_->engine, path.c_str(), flags,
                                               nullptr, nullptr, &voice->sound);
    if (result != MA_SUCCESS) {
        std::fprintf(stderr, "audio: cannot load '%s' (%d)\n", path.c_str(), (int)result);
        return false;
    }

    voice->loaded = true;
    impl_->voices[id] = std::move(voice);
    return true;
}

void AudioSystem::unload(const std::string& id) {
    if (!impl_) return;
    impl_->destroyVoice(id);
}

void AudioSystem::unloadAll() {
    if (!impl_) return;
    for (auto& entry : impl_->voices) {
        Impl::Voice& voice = *entry.second;
        if (voice.loaded) {
            ma_sound_stop(&voice.sound);
            ma_sound_uninit(&voice.sound);
            voice.loaded = false;
        }
    }
    impl_->voices.clear();
}

bool AudioSystem::exists(const std::string& id) const {
    const Impl::Voice* voice = impl_ ? impl_->find(id) : nullptr;
    return voice && voice->loaded;
}

bool AudioSystem::play(const std::string& id, bool loop, float volume, float pan) {
    Impl::Voice* voice = impl_ ? impl_->find(id) : nullptr;
    if (!voice || !voice->loaded) return false;

    ma_sound_stop(&voice->sound);
    ma_sound_seek_to_pcm_frame(&voice->sound, 0);
    ma_sound_set_looping(&voice->sound, loop ? MA_TRUE : MA_FALSE);
    ma_sound_set_volume(&voice->sound, std::clamp(volume, 0.0f, 1.0f));
    ma_sound_set_pan(&voice->sound, std::clamp(pan, -1.0f, 1.0f));

    voice->paused = false;
    if (ma_sound_start(&voice->sound) != MA_SUCCESS) {
        std::fprintf(stderr, "audio: cannot play '%s'\n", id.c_str());
        return false;
    }
    return true;
}

bool AudioSystem::pause(const std::string& id) {
    Impl::Voice* voice = impl_ ? impl_->find(id) : nullptr;
    if (!voice || !voice->loaded) return false;
    if (ma_sound_is_playing(&voice->sound)) {
        ma_sound_stop(&voice->sound);
        voice->paused = true;
    }
    return true;
}

bool AudioSystem::resume(const std::string& id) {
    Impl::Voice* voice = impl_ ? impl_->find(id) : nullptr;
    if (!voice || !voice->loaded) return false;
    if (ma_sound_at_end(&voice->sound)) {
        ma_sound_seek_to_pcm_frame(&voice->sound, 0);
    }
    if (ma_sound_start(&voice->sound) != MA_SUCCESS) return false;
    voice->paused = false;
    return true;
}

bool AudioSystem::stop(const std::string& id) {
    Impl::Voice* voice = impl_ ? impl_->find(id) : nullptr;
    if (!voice || !voice->loaded) return false;
    ma_sound_stop(&voice->sound);
    ma_sound_seek_to_pcm_frame(&voice->sound, 0);
    voice->paused = false;
    return true;
}

bool AudioSystem::isPlaying(const std::string& id) const {
    const Impl::Voice* voice = impl_ ? impl_->find(id) : nullptr;
    if (!voice || !voice->loaded) return false;
    return ma_sound_is_playing(&voice->sound) && !ma_sound_at_end(&voice->sound);
}

bool AudioSystem::isPaused(const std::string& id) const {
    const Impl::Voice* voice = impl_ ? impl_->find(id) : nullptr;
    return voice && voice->loaded && voice->paused && !ma_sound_is_playing(&voice->sound);
}

void AudioSystem::setVolume(const std::string& id, float volume) {
    Impl::Voice* voice = impl_ ? impl_->find(id) : nullptr;
    if (!voice || !voice->loaded) return;
    ma_sound_set_volume(&voice->sound, std::clamp(volume, 0.0f, 1.0f));
}

void AudioSystem::setPan(const std::string& id, float pan) {
    Impl::Voice* voice = impl_ ? impl_->find(id) : nullptr;
    if (!voice || !voice->loaded) return;
    ma_sound_set_pan(&voice->sound, std::clamp(pan, -1.0f, 1.0f));
}

void AudioSystem::setLooping(const std::string& id, bool loop) {
    Impl::Voice* voice = impl_ ? impl_->find(id) : nullptr;
    if (!voice || !voice->loaded) return;
    ma_sound_set_looping(&voice->sound, loop ? MA_TRUE : MA_FALSE);
}

void AudioSystem::setMasterVolume(float volume) {
    if (!impl_) return;
    impl_->master = std::clamp(volume, 0.0f, 1.0f);
    if (impl_->engineReady) ma_engine_set_volume(&impl_->engine, impl_->master);
}

float AudioSystem::masterVolume() const {
    return impl_ ? impl_->master : 0.0f;
}

void AudioSystem::stopAll() {
    if (!impl_) return;
    for (auto& entry : impl_->voices) {
        Impl::Voice& voice = *entry.second;
        if (!voice.loaded) continue;
        ma_sound_stop(&voice.sound);
        ma_sound_seek_to_pcm_frame(&voice.sound, 0);
        voice.paused = false;
    }
}

void AudioSystem::pauseAll() {
    if (!impl_) return;
    for (auto& entry : impl_->voices) {
        Impl::Voice& voice = *entry.second;
        if (!voice.loaded) continue;
        if (ma_sound_is_playing(&voice.sound)) {
            ma_sound_stop(&voice.sound);
            voice.paused = true;
        }
    }
}

void AudioSystem::resumeAll() {
    if (!impl_) return;
    for (auto& entry : impl_->voices) {
        Impl::Voice& voice = *entry.second;
        if (!voice.loaded) continue;
        if (ma_sound_at_end(&voice.sound)) ma_sound_seek_to_pcm_frame(&voice.sound, 0);
        ma_sound_start(&voice.sound);
        voice.paused = false;
    }
}
