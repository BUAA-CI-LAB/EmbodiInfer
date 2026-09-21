// In-process bridge for the pinned Embodied.cpp public C++ API.
// It forwards complete predictions without changing model execution.
#include "runtime/model.h"

#include <algorithm>
#include <cstdint>
#include <exception>
#include <string>
#include <vector>

namespace {
thread_local std::string error_message;
}

extern "C" {
const char * benchmark_error() { return error_message.c_str(); }

void * benchmark_load(const char * checkpoint, const char * mmproj, const char * backbone) {
    try {
        auto * model = vla::model_load(mmproj ? mmproj : "", checkpoint, backbone ? backbone : "");
        if (!model) error_message = "Embodied.cpp model_load failed";
        return model;
    } catch (const std::exception & error) {
        error_message = error.what();
        return nullptr;
    }
}

void benchmark_free(void * handle) {
    if (handle) vla::model_free(static_cast<vla::Model *>(handle));
}

int benchmark_predict(void * handle, const float * pixels, int views, int side,
                      const int32_t * tokens, int token_count, const char * instruction,
                      const float * state, const float * noise, float * output,
                      int64_t output_count, float * timing) {
    if (!handle || !pixels || views < 1 || side < 1 || !noise || !output || !timing) {
        error_message = "invalid benchmark input";
        return -1;
    }
    try {
        auto * model = static_cast<vla::Model *>(handle);
        const auto & config = vla::model_config(model);
        if (config.n_suffix * config.max_action_dim != output_count) {
            error_message = "checkpoint output shape differs from benchmark";
            return -2;
        }
        std::vector<vla::ImageView> images(views);
        for (int i = 0; i < views; ++i) {
            images[i].data = pixels + static_cast<int64_t>(i) * side * side * 3;
            images[i].w = side;
            images[i].h = side;
            images[i].format = vla::PixelFormat::F32_RGB_01;
        }
        vla::Inputs input{};
        input.images = images.data();
        input.n_images = views;
        input.lang_tokens = tokens;
        input.n_lang = token_count;
        input.language_text = instruction;
        input.state = state;
        input.noise = noise;
        input.timing_detail = vla::TimingDetail::PHASE;
        const auto actions = vla::predict(model, input);
        if (static_cast<int64_t>(actions.size()) != output_count) {
            error_message = "Embodied.cpp returned an unexpected action size";
            return -3;
        }
        std::copy(actions.begin(), actions.end(), output);
        const auto & stats = vla::last_stats(model);
        timing[0] = stats.ms_total;
        timing[1] = stats.ms_vision;
        timing[2] = stats.ms_inference;
        timing[3] = stats.ms_prefill;
        timing[4] = stats.ms_denoise;
        return 0;
    } catch (const std::exception & error) {
        error_message = error.what();
        return -4;
    }
}
}
