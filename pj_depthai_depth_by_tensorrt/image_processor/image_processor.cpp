/*** Include ***/
/* for general */
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>
#include <array>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <memory>

/* for OpenCV */
#include <opencv2/opencv.hpp>

/* for My modules */
#include "common_helper.h"
#include "common_helper_cv.h"
#include "depth_stereo_engine.h"
#include "depth_midasv2_engine.h"
#include "image_processor.h"

/*** Macro ***/
#define TAG "ImageProcessor"
#define PRINT(...)   COMMON_HELPER_PRINT(TAG, __VA_ARGS__)
#define PRINT_E(...) COMMON_HELPER_PRINT_E(TAG, __VA_ARGS__)

/*** Global variable ***/
std::unique_ptr<DepthStereoEngine> s_depth_stereo_engine;
std::unique_ptr<DepthMidasv2Engine> s_depth_midasv2_engine;

/*** Function ***/
static void DrawFps(cv::Mat& mat, double time_inference, cv::Point pos, double font_scale, int32_t thickness, cv::Scalar color_front, cv::Scalar color_back, bool is_text_on_rect = true)
{
    char text[64];
    static auto time_previous = std::chrono::steady_clock::now();
    auto time_now = std::chrono::steady_clock::now();
    double fps = 1e9 / (time_now - time_previous).count();
    fps = 0;
    time_previous = time_now;
    snprintf(text, sizeof(text), "FPS: %.1f, Inference: %.1f [ms]", fps, time_inference);
    CommonHelper::DrawText(mat, text, cv::Point(0, 0), 0.5, 2, CommonHelper::CreateCvColor(0, 0, 0), CommonHelper::CreateCvColor(180, 180, 180), true);
}

int32_t ImageProcessor::Initialize(const InputParam& input_param)
{
    if (s_depth_stereo_engine) {
        PRINT_E("Already initialized\n");
        return -1;
    }

    s_depth_midasv2_engine.reset(new DepthMidasv2Engine());
    if (s_depth_midasv2_engine->Initialize(input_param.work_dir, input_param.num_threads) != DepthMidasv2Engine::kRetOk) {
        s_depth_midasv2_engine->Finalize();
        s_depth_midasv2_engine.reset();
        return -1;
    }
    s_depth_stereo_engine.reset(new DepthStereoEngine());
    if (s_depth_stereo_engine->Initialize(input_param.work_dir, input_param.num_threads) != DepthStereoEngine::kRetOk) {
        s_depth_stereo_engine->Finalize();
        s_depth_stereo_engine.reset();
        return -1;
    }

    return 0;
}

int32_t ImageProcessor::Finalize(void)
{
    if (!s_depth_stereo_engine) {
        PRINT_E("Not initialized\n");
        return -1;
    }

    if (s_depth_midasv2_engine->Finalize() != DepthMidasv2Engine::kRetOk) {
        return -1;
    }
    if (s_depth_stereo_engine->Finalize() != DepthStereoEngine::kRetOk) {
        return -1;
    }

    return 0;
}


int32_t ImageProcessor::Command(int32_t cmd)
{
    if (!s_depth_stereo_engine) {
        PRINT_E("Not initialized\n");
        return -1;
    }

    switch (cmd) {
    case 0:
    default:
        PRINT_E("command(%d) is not supported\n", cmd);
        return -1;
    }
}

static cv::Mat ConvertDisparity2Depth(const cv::Mat& mat_disparity, float fov, float baseline, float mag = 1.0f)
{
    cv::Mat mat_depth(mat_disparity.size(), CV_8UC1);
    const float scale = mag * fov * baseline;
#pragma omp parallel for
    for (int32_t i = 0; i < mat_disparity.total(); i++) {
        if (mat_disparity.at<float>(i) > 0) {
            float Z = scale / mat_disparity.at<float>(i);   // [meter]
            if (Z <= 255.0f) {
                mat_depth.at<uint8_t>(i) = static_cast<uint8_t>(Z);
            } else {
                mat_depth.at<uint8_t>(i) = 255;
            }
        } else {
            mat_depth.at<uint8_t>(i) = 255;
        }
    }
    return mat_depth;
}

static cv::Mat NormalizeDisparity(const cv::Mat& mat_disparity, float max_disparity, float mag = 1.0f)
{
    cv::Mat mat_depth(mat_disparity.size(), CV_8UC1);
    const float scale = mag * 255.0f / max_disparity;
#pragma omp parallel for
    for (int32_t i = 0; i < mat_disparity.total(); i++) {
        mat_depth.at<uint8_t>(i) = static_cast<uint8_t>(mat_disparity.at<float>(i) * scale);
    }
    return mat_depth;
}

cv::Mat NormalizeMinMax(cv::Mat& mat_depth)
{
    /* (255 * (prediction - depth_min) / (depth_max - depth_min)) */
    cv::Mat mat_out;
    double depth_min, depth_max;
    cv::minMaxLoc(mat_depth, &depth_min, &depth_max);
    mat_depth.convertTo(mat_out, CV_8UC1, 255. / (depth_max - depth_min), (-255. * depth_min) / (depth_max - depth_min));
    return mat_out;
}

int32_t ImageProcessor::Process(cv::Mat& mat_color, cv::Mat& mat_left, cv::Mat& mat_right, cv::Mat& mat_result_0, cv::Mat& mat_result_1, Result& result)
{
    if (!s_depth_stereo_engine) {
        PRINT_E("Not initialized\n");
        return -1;
    }

    /* Mono depth by Midas V2 */
    DepthMidasv2Engine::Result result_depth_midasv2_engine;
    if (s_depth_midasv2_engine->Process(mat_color, result_depth_midasv2_engine) != DepthStereoEngine::kRetOk) {
        return -1;
    }
    
    cv::Mat mat_depth_midasv2 = NormalizeMinMax(result_depth_midasv2_engine.mat_out);
    cv::applyColorMap(mat_depth_midasv2, mat_depth_midasv2, cv::COLORMAP_MAGMA);
    cv::resize(mat_depth_midasv2, mat_depth_midasv2, mat_color.size());
    DrawFps(mat_depth_midasv2, result_depth_midasv2_engine.time_inference, cv::Point(0, 0), 0.5, 2, CommonHelper::CreateCvColor(0, 0, 0), CommonHelper::CreateCvColor(180, 180, 180), true);
  

    /* Stereo depth by HITNET */
    DepthStereoEngine::Result result_depth_stereo_engine;
    if (s_depth_stereo_engine->Process(mat_left, mat_right, result_depth_stereo_engine) != DepthStereoEngine::kRetOk) {
        return -1;
    }
    //cv::Mat mat_depth = ConvertDisparity2Depth(result_depth_stereo_engine.image, 500.0f, 0.2f, 50);
    //cv::Mat mat_depth_stereo = NormalizeDisparity(result_depth_stereo_engine.image, s_depth_stereo_engine->GetMaxDisparity(), 1.0f);
    cv::Mat mat_depth_stereo = NormalizeMinMax(result_depth_stereo_engine.image);
    cv::applyColorMap(mat_depth_stereo, mat_depth_stereo, cv::COLORMAP_MAGMA);
    cv::resize(mat_depth_stereo, mat_depth_stereo, mat_left.size());
    DrawFps(mat_depth_stereo, result_depth_stereo_engine.time_inference, cv::Point(0, 0), 0.5, 2, CommonHelper::CreateCvColor(0, 0, 0), CommonHelper::CreateCvColor(180, 180, 180), true);

    /* Return the results */
    mat_result_0 = mat_depth_midasv2;
    mat_result_1 = mat_depth_stereo;
    result.time_pre_process = result_depth_midasv2_engine.time_pre_process + result_depth_stereo_engine.time_pre_process;
    result.time_inference = result_depth_midasv2_engine.time_inference + result_depth_stereo_engine.time_inference;
    result.time_post_process = result_depth_midasv2_engine.time_post_process + result_depth_stereo_engine.time_post_process;

    return 0;
}

