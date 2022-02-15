https://user-images.githubusercontent.com/11009876/154006646-262cb61b-6a58-4559-a08e-c3ba5d9f6ddc.mp4

# Play with DepthAI
- Sample projects to use DepthAI + OpenCV with OAK-D cameras

# Target
- Platform
    - ~~Linux (x64)~~
    - ~~Linux (aarch64)~~
    - Windows (x64). Visual Studio 2019
- Note: It's not tested in Linux yet, but it probably works

# How to build a project
## 0. Requirements
- Windows
    - OpenCV 4.5.5
        - https://github.com/opencv/opencv/releases/download/4.5.5/opencv-4.5.5-vc14_vc15.exe
        - Extact to `third_party/opencv`
    - depthai-core v2.14.1
        - https://github.com/luxonis/depthai-core/releases/download/v2.14.1/depthai-core-v2.14.1-win64.zip
        - Extact + rename to `third_party/depthai-core`
- Linux
    - Follow the instructions
        - https://github.com/luxonis/depthai-core
    - Set `OpenCV_DIR` and `depthai_DIR` variables if needed

## 1. Download 
- Download source code and pre-built libraries
    ```sh
    git clone https://github.com/iwatake2222/play_with_depthai.git
    cd play_with_depthai
    git submodule update --init
    ```
- Download models
    ```sh
    sh ./download_resource.sh
    ```

## 2-a. Build in Linux
todo

## 2-b. Build in Windows (Visual Studio)
- Configure and Generate a new project using cmake-gui for Visual Studio 2019 64-bit
    - `Where is the source code` : path-to-play_with_depthai/pj_depthai_basic_camera	(for example)
    - `Where to build the binaries` : path-to-build	(any)
- Open `main.sln`
- Run

# License
- Copyright 2022 iwatake2222
- Licensed under the Apache License, Version 2.0
    - [LICENSE](LICENSE)
