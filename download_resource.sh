move_dir_to_shell_file() {
    dir_shell_file=`dirname "$0"`
    cd ${dir_shell_file}
}

download() {
    local url=$1
    local path=$2
    echo "Downloading ${url}"
    curl -Lo ${2} ${url}
}

download_and_extract() {
    local url=$1
    echo "Downloading ${url}"
    local ext=${url##*.}
    if [ `echo ${ext} | grep zip` ]; then
        curl -Lo temp.zip ${url}
        unzip -o temp.zip
        rm temp.zip
    else
        curl -Lo temp.tgz ${url}
        tar xzvf temp.tgz
        rm temp.tgz
    fi
}
########################################################################

move_dir_to_shell_file

# pj_depthai_basic_mobilenet
download "https://artifacts.luxonis.com/artifactory/luxonis-depthai-data-local/network/mobilenet-ssd_openvino_2021.2_6shave.blob" "resource/model/mobilenet-ssd_openvino_2021.2_6shave.blob"

