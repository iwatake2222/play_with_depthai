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
#define MODEL_FILENAME RESOURCE_DIR"/model/mobilenet-ssd_openvino_2021.2_6shave.blob"

/*** Function ***/
class DepthAiWrapper
{
public:
    DepthAiWrapper()
    {
        /*** Define source ***/
        /* Color Camera */
        auto color_camera = pipeline.create<dai::node::ColorCamera>();
        auto manip = pipeline.create<dai::node::ImageManip>();
        /* MobileNet */
        auto nn = pipeline.create<dai::node::MobileNetDetectionNetwork>();

        /*** Define output ***/
        /* Color Camera */
        auto xout_color_camera_preview = pipeline.create<dai::node::XLinkOut>();
        xout_color_camera_preview->setStreamName("color_camera_preview");
        /* MobileNet */
        auto nnOut = pipeline.create<dai::node::XLinkOut>();
        nnOut->setStreamName("nn");

        /*** Properties ***/
        /* Color Camera */
        color_camera->setBoardSocket(dai::CameraBoardSocket::RGB);
        color_camera->setResolution(dai::ColorCameraProperties::SensorResolution::THE_1080_P);
        color_camera->setInterleaved(false);
        color_camera->setColorOrder(dai::ColorCameraProperties::ColorOrder::RGB);
        color_camera->setVideoSize(1920, 1080);
        color_camera->setPreviewSize(300 * 1920 / 1080, 300);
        /* manip */
        manip->initialConfig.setResize(300, 300);
        manip->initialConfig.setFrameType(dai::ImgFrame::Type::BGR888p);
        /* MobileNet */
        nn->setConfidenceThreshold(0.5);
        nn->setBlobPath(MODEL_FILENAME);
        nn->setNumInferenceThreads(2);
        nn->input.setBlocking(false);

        /*** Linking ***/
        /* Color Camera */
        //color_camera->preview.link(xout_color_camera_preview->input);
        manip->out.link(xout_color_camera_preview->input);
        /* MobileNet */
        //color_camera->preview.link(nn->input);
        color_camera->preview.link(manip->inputImage);
        manip->out.link(nn->input);
        nn->out.link(nnOut->input);

        /*** Connect to deviceand start pipeline ***/
        device = std::make_unique<dai::Device>(pipeline, dai::UsbSpeed::SUPER);

        /*** Get Output Queue ***/
        /* Color Camera */
        queue_color_camera_preview = device->getOutputQueue("color_camera_preview", 4, false);
        /* MobileNet */
        queue_mobilenet = device->getOutputQueue("nn", 4, false);
    }
    ~DepthAiWrapper() {}

    cv::Mat GetColorCameraPreview()
    {
        return queue_color_camera_preview->get<dai::ImgFrame>()->getCvFrame();
    }

    std::shared_ptr<dai::ImgDetections> GetDetection()
    {
        return queue_mobilenet->tryGet<dai::ImgDetections>();
    }

private:
    dai::Pipeline pipeline;
    std::unique_ptr<dai::Device> device;

    /* Color Camera */
    std::shared_ptr<dai::DataOutputQueue> queue_color_camera_preview;
    /* MobileNet */
    std::shared_ptr<dai::DataOutputQueue> queue_mobilenet;
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
        cv::Mat image_color_camera_preview = depth_ai.GetColorCameraPreview();
        const auto& time_cap1 = std::chrono::steady_clock::now();

        /* Call image processor library */
        const auto& time_image_process0 = std::chrono::steady_clock::now();
        const auto& time_image_process1 = std::chrono::steady_clock::now();

        /* Decode detections */
        auto detections = depth_ai.GetDetection();
        if (detections) {
            for (auto& detection : detections->detections) {
                int x1 = detection.xmin * image_color_camera_preview.cols;
                int y1 = detection.ymin * image_color_camera_preview.rows;
                int x2 = detection.xmax * image_color_camera_preview.cols;
                int y2 = detection.ymax * image_color_camera_preview.rows;
                cv::rectangle(image_color_camera_preview, cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2)), cv::Scalar(0, 255, 0), cv::FONT_HERSHEY_SIMPLEX);
            }
        }

        /* Display result */
        cv::imshow("image_color_camera_preview", image_color_camera_preview);

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
