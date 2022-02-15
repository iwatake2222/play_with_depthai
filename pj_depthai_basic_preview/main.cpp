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
int32_t main(int argc, char* argv[])
{
    /*** Initialize ***/
    /* variables for processing time measurement */
    double total_time_all = 0;
    double total_time_cap = 0;
    double total_time_image_process = 0;

    /*** Process for each frame ***/
    int32_t frame_cnt = 0;
    for (frame_cnt = 0; ; frame_cnt++) {
        const auto& time_all0 = std::chrono::steady_clock::now();
        /* Read image */
        const auto& time_cap0 = std::chrono::steady_clock::now();
        cv::Mat image = cv::Mat::zeros(4, 4, CV_8UC3);
        if (image.empty()) break;
        const auto& time_cap1 = std::chrono::steady_clock::now();

        /* Call image processor library */
        const auto& time_image_process0 = std::chrono::steady_clock::now();

        const auto& time_image_process1 = std::chrono::steady_clock::now();

        /* Display result */
        cv::imshow("test", image);

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
