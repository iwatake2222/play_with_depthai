/* Copyright 2022 iwatake2222

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
/*** Include ***/
/* for general */
#include <cstdint>
#include <cstdlib>
#include <string>
#include <algorithm>
#include <chrono>

/* for OpenCV */
//#include <opencv2/opencv.hpp>
#include "depthai/depthai.hpp"

/* for My modules */


/*** Macro ***/

/*** Function ***/
class DepthAiWrapper
{
public:
    DepthAiWrapper()
    {
        /*** Define source ***/
        /* Color Camera */
        auto color_camera = pipeline.create<dai::node::ColorCamera>();
        /* Stereo Camera */
        auto mono_camera_right = pipeline.create<dai::node::MonoCamera>();
        auto mono_camera_left = pipeline.create<dai::node::MonoCamera>();
        auto stereo = pipeline.create<dai::node::StereoDepth>();

        /*** Define output ***/
        /* Color Camera */
        auto xout_color_camera_video = pipeline.create<dai::node::XLinkOut>();
        xout_color_camera_video->setStreamName("color_camera_video");
        auto xout_color_camera_preview = pipeline.create<dai::node::XLinkOut>();
        xout_color_camera_preview->setStreamName("color_camera_preview");
        /* Stereo Camera */
        auto xout_mono_camera_rectified_right = pipeline.create<dai::node::XLinkOut>();
        xout_mono_camera_rectified_right->setStreamName("mono_camera_rectified_right");
        auto xout_mono_camera_rectified_left = pipeline.create<dai::node::XLinkOut>();
        xout_mono_camera_rectified_left->setStreamName("mono_camera_rectified_left");
        auto xout_disparity = pipeline.create<dai::node::XLinkOut>();
        xout_disparity->setStreamName("disparity");

        /*** Properties ***/
        /* Color Camera */
        color_camera->setBoardSocket(dai::CameraBoardSocket::RGB);
        color_camera->setResolution(dai::ColorCameraProperties::SensorResolution::THE_1080_P);
        color_camera->setInterleaved(false);
        color_camera->setColorOrder(dai::ColorCameraProperties::ColorOrder::RGB);
        color_camera->setVideoSize(1920, 1080);
        color_camera->setPreviewSize(480 * 1920 / 1080, 480);
        /* Stereo Camera */
        mono_camera_right->setBoardSocket(dai::CameraBoardSocket::RIGHT);
        mono_camera_right->setResolution(dai::MonoCameraProperties::SensorResolution::THE_400_P);
        mono_camera_left->setBoardSocket(dai::CameraBoardSocket::LEFT);
        mono_camera_left->setResolution(dai::MonoCameraProperties::SensorResolution::THE_400_P);
        stereo->setDefaultProfilePreset(dai::node::StereoDepth::PresetMode::HIGH_DENSITY);
        stereo->setRectifyEdgeFillColor(0);
        stereo->initialConfig.setMedianFilter(dai::MedianFilter::KERNEL_7x7);
        stereo->setLeftRightCheck(true);
        stereo->setExtendedDisparity(false);
        stereo->setSubpixel(false);

        /*** Linking ***/
        /* Color Camera */
        color_camera->video.link(xout_color_camera_video->input);
        color_camera->preview.link(xout_color_camera_preview->input);
        /* Stereo Camera */
        mono_camera_right->out.link(stereo->right);
        mono_camera_left->out.link(stereo->left);
        stereo->disparity.link(xout_disparity->input);
        stereo->rectifiedRight.link(xout_mono_camera_rectified_right->input);
        stereo->rectifiedLeft.link(xout_mono_camera_rectified_left->input);

        /*** Connect to deviceand start pipeline ***/
        device = std::make_unique<dai::Device>(pipeline, dai::UsbSpeed::SUPER);

        /*** Get Output Queue ***/
        /* Color Camera */
        queue_color_camera_video = device->getOutputQueue("color_camera_video", 4, false);
        queue_color_camera_preview = device->getOutputQueue("color_camera_preview", 4, false);
        /* Stereo Camera */
        queue_mono_camera_rectified_right = device->getOutputQueue("mono_camera_rectified_right", 4, false);
        queue_mono_camera_rectified_left = device->getOutputQueue("mono_camera_rectified_left", 4, false);
        queue_disparity = device->getOutputQueue("disparity", 4, false);
        disparity_multiplier = 255 / stereo->initialConfig.getMaxDisparity();
    }
    ~DepthAiWrapper() {}

    cv::Mat GetColorCameraVideo()
    {
        return queue_color_camera_video->get<dai::ImgFrame>()->getCvFrame();
    }

    cv::Mat GetColorCameraPreview()
    {
        return queue_color_camera_preview->get<dai::ImgFrame>()->getCvFrame();
    }

    cv::Mat GetMonoCameraRectifiedRight()
    {
        return queue_mono_camera_rectified_right->get<dai::ImgFrame>()->getCvFrame();
    }

    cv::Mat GetMonoCameraRectifiedLeft()
    {
        return queue_mono_camera_rectified_left->get<dai::ImgFrame>()->getCvFrame();
    }

    cv::Mat GetDisparity()
    {
        return queue_disparity->get<dai::ImgFrame>()->getCvFrame();
    }

    float GetDisparityMultiplier()
    {
        return disparity_multiplier;
    }

private:
    dai::Pipeline pipeline;
    std::unique_ptr<dai::Device> device;

    /* Color Camera */
    std::shared_ptr<dai::DataOutputQueue> queue_color_camera_video;
    std::shared_ptr<dai::DataOutputQueue> queue_color_camera_preview;
    /* Stereo Camera */
    std::shared_ptr<dai::DataOutputQueue> queue_mono_camera_rectified_right;
    std::shared_ptr<dai::DataOutputQueue> queue_mono_camera_rectified_left;
    std::shared_ptr<dai::DataOutputQueue> queue_disparity;
    float disparity_multiplier;
};

int32_t main(int argc, char* argv[])
{
    /*** Initialize ***/
    /* variables for processing time measurement */
    double total_time_all = 0;
    double total_time_cap = 0;
    double total_time_image_process = 0;

    DepthAiWrapper depth_ai;

    /*** Process for each frame ***/
    int32_t frame_cnt = 0;
    for (frame_cnt = 0; ; frame_cnt++) {
        const auto& time_all0 = std::chrono::steady_clock::now();
        /* Read image */
        const auto& time_cap0 = std::chrono::steady_clock::now();
        cv::Mat image_color_camera_video = depth_ai.GetColorCameraVideo();
        cv::Mat image_color_camera_preview = depth_ai.GetColorCameraPreview();
        cv::Mat image_mono_camera_rectified_right = depth_ai.GetMonoCameraRectifiedRight();
        cv::Mat image_mono_camera_rectified_left = depth_ai.GetMonoCameraRectifiedLeft();
        cv::Mat image_disparity = depth_ai.GetDisparity();
        const auto& time_cap1 = std::chrono::steady_clock::now();
        
        /* Call image processor library */
        const auto& time_image_process0 = std::chrono::steady_clock::now();
        const auto& time_image_process1 = std::chrono::steady_clock::now();

        /* Extend disparity range */
        cv::Mat image_disparity_colored;
        image_disparity.convertTo(image_disparity_colored, CV_8UC1, depth_ai.GetDisparityMultiplier());
        cv::applyColorMap(image_disparity_colored, image_disparity_colored, cv::COLORMAP_JET);

        /* Display result */
        cv::imshow("image_color_camera_video", image_color_camera_video);
        cv::imshow("image_color_camera_preview", image_color_camera_preview);
        cv::imshow("image_mono_camera_rectified_right", image_mono_camera_rectified_right);
        cv::imshow("image_mono_camera_rectified_left", image_mono_camera_rectified_left);
        cv::imshow("image_disparity", image_disparity);
        cv::imshow("image_disparity_colored", image_disparity_colored);

        /* Input key command */
        int key = cv::waitKey(1);
        if (key == 'q' || key == 'Q' || key == 27) {
            break;
        }

        /* Print processing time */
        const auto& time_all1 = std::chrono::steady_clock::now();
        double time_all = (time_all1 - time_all0).count() / 1000000.0;
        double time_cap = (time_cap1 - time_cap0).count() / 1000000.0;
        double time_image_process = (time_image_process1 - time_image_process0).count() / 1000000.0;
        printf("Total:               %9.3lf [msec]\n", time_all);
        printf("  Capture:           %9.3lf [msec]\n", time_cap);
        printf("  Image processing:  %9.3lf [msec]\n", time_image_process);
        printf("=== Finished %d frame ===\n\n", frame_cnt);

        if (frame_cnt > 0) {    /* do not count the first process because it may include initialize process */
            total_time_all += time_all;
            total_time_cap += time_cap;
            total_time_image_process += time_image_process;
        }
    }
    
    /*** Finalize ***/
    /* Print average processing time */
    if (frame_cnt > 1) {
        frame_cnt--;    /* because the first process was not counted */
        printf("=== Average processing time ===\n");
        printf("Total:               %9.3lf [msec]\n", total_time_all / frame_cnt);
        printf("  Capture:           %9.3lf [msec]\n", total_time_cap / frame_cnt);
        printf("  Image processing:  %9.3lf [msec]\n", total_time_image_process / frame_cnt);
    }

    cv::waitKey(-1);

    return 0;
}
