#!/bin/bash

SCRIPT_DIR=$(realpath "$(dirname "$0")")

# Default values
PROJECT_NAME="dx-modelzoo"
PROJECT_ROOT=$(realpath -s "${SCRIPT_DIR}")
RUNTIME_PATH=$(realpath -s "${PROJECT_ROOT}/../dx-runtime")
DXRT_SRC_PATH="${RUNTIME_PATH}/dx_rt"
DX_AS_PATH=$(realpath -s "${RUNTIME_PATH}/..")
DOCKER_VOLUME_PATH=${DOCKER_VOLUME_PATH}
FORCE_ARGS=""

# color env settings
source ${PROJECT_ROOT}/scripts/color_env.sh
source ${PROJECT_ROOT}/scripts/common_util.sh

pushd $PROJECT_ROOT >&2

# Function to display help message
show_help() {
    print_colored_v2 "YELLO" "Usage: $(basename "$0") [OPTIONS]"
    print_colored_v2 "YELLO" "Options:"
    print_colored_v2 "YELLO" "  --dxrt_src_path=<path>         Set DXRT source path (default: ../dx-runtime/dx_rt)"
    print_colored_v2 "YELLO" "  --docker_volume_path=<path>    Set Docker volume path (required in container mode)"
    print_colored_v2 "YELLO" "  [--force]                      Force overwrite if the file already exists"
    print_colored_v2 "YELLO" "  [--help]                       Show this help message"

    if [ "$1" == "error" ] && [[ ! -n "$2" ]]; then
        print_colored_v2 "ERROR" "Invalid or missing arguments."
        exit 1
    elif [ "$1" == "error" ] && [[ -n "$2" ]]; then
        print_colored_v2 "ERROR" "$2"
        exit 1
    elif [[ "$1" == "warn" ]] && [[ -n "$2" ]]; then
        print_colored_v2 "WARNING" "$2"
        return 0
    fi
    exit 0
}

validate_path() {
    # check DX-AS mode
    if [ ! -d "$DX_AS_PATH" ]; then
        print_colored_v2 "INFO" "[Normal mode]"
        print_colored_v2 "INFO" "[DXRT_SRC_PATH=${DXRT_SRC_PATH}]"
    else
        print_colored_v2 "INFO" "[DX-AS mode] (DX_AS_PATH: ${DX_AS_PATH} is detected)"
        print_colored_v2 "INFO" "[DXRT_SRC_PATH=${DXRT_SRC_PATH}]"
    fi

    
    # Check if DXRT_SRC_PATH exists
    if [ ! -d $DXRT_SRC_PATH ]; then
        local err_msg="DXRT_SRC_PATH ($DXRT_SRC_PATH) does not exist.\nUse the '--dxrt_src_path=<dir>' option to specify the directory where the 'dx_rt' source code is located."
        show_help "error" "${err_msg}"
    fi
}

# NOTE: This function requires sudo privileges to install Python and its dependencies
# via apt, and is intended for Debian-based systems like Ubuntu.
install_python() {
    REQUIRED_MAJOR=3
    MIN_REQUIRED_MINOR=11
    MAX_POSSIBLE_MINOR_TO_CHECK=20
    TARGET_INSTALL_VERSION="3.11"

    print_colored_v2 "INFO" "Checking if Python $REQUIRED_MAJOR.$MIN_REQUIRED_MINOR or newer is installed..."

    found_suitable_python=false
    installed_version_message=""

    if command -v python3 &> /dev/null; then
        print_colored_v2 "INFO" "Checking version of 'python3' command..."
        CURRENT_PYTHON_VERSION_FULL=$(python3 --version 2>&1)
        CURRENT_PYTHON_VERSION_STRING=$(print_colored_v2 "INFO" "$CURRENT_PYTHON_VERSION_FULL" | awk '{print $2}' | sed 's/[^0-9.]*//g')

        if [[ -n "$CURRENT_PYTHON_VERSION_STRING" && "$CURRENT_PYTHON_VERSION_STRING" =~ ^[0-9]+\.[0-9]+ ]]; then
            IFS='.' read -r -a VERSION_PARTS <<< "$CURRENT_PYTHON_VERSION_STRING"
            CURRENT_MAJOR=${VERSION_PARTS[0]}
            CURRENT_MINOR=${VERSION_PARTS[1]:-0}

            print_colored_v2 "INFO" "Detected 'python3' points to version: $CURRENT_MAJOR.$CURRENT_MINOR"

            if [ "$CURRENT_MAJOR" -eq "$REQUIRED_MAJOR" ] && [ "$CURRENT_MINOR" -ge "$MIN_REQUIRED_MINOR" ]; then
                installed_version_message="'python3' command refers to $CURRENT_MAJOR.$CURRENT_MINOR"
                found_suitable_python=true
            else
                print_colored_v2 "INFO" "'python3' is version $CURRENT_MAJOR.$CURRENT_MINOR, which is older than required."
            fi
        else
            print_colored_v2 "INFO" "Could not parse version from 'python3 --version' output: $CURRENT_PYTHON_VERSION_FULL"
        fi
    else
        print_colored_v2 "INFO" "'python3' command not found."
    fi

    if ! $found_suitable_python; then
        print_colored_v2 "INFO" "Checking for specific Python versions (python$REQUIRED_MAJOR.$MIN_REQUIRED_MINOR onwards)..."
        for (( minor_v=$MIN_REQUIRED_MINOR; minor_v<=$MAX_POSSIBLE_MINOR_TO_CHECK; minor_v++ )); do
            python_executable="python$REQUIRED_MAJOR.$minor_v"
            if command -v "$python_executable" &> /dev/null; then
                version_output=$("$python_executable" --version 2>&1)
                installed_version_message="Found '$python_executable' (version: $version_output)"
                found_suitable_python=true
                break
            fi
        done
    fi

    if $found_suitable_python; then
        print_colored_v2 "INFO" "$installed_version_message, which satisfies the requirement of Python $REQUIRED_MAJOR.$MIN_REQUIRED_MINOR or newer."
        print_colored_v2 "INFO" "Skipping Python $TARGET_INSTALL_VERSION installation."
        return 0
    else
        print_colored_v2 "INFO" "No suitable Python version ($REQUIRED_MAJOR.$MIN_REQUIRED_MINOR or newer) found through 'python3' or specific 'python3.X' commands."
    fi

    print_colored_v2 "INFO" "Installing Python $TARGET_INSTALL_VERSION..."
    if sudo add-apt-repository -y ppa:deadsnakes/ppa; then
        sudo apt-get update
        if sudo apt-get install -y "python${TARGET_INSTALL_VERSION}" "python${TARGET_INSTALL_VERSION}-dev" "python${TARGET_INSTALL_VERSION}-venv"; then
            print_colored_v2 "INFO" "Python $TARGET_INSTALL_VERSION installation process completed successfully."
        else
            print_colored_v2 "ERROR" "Failed to install Python $TARGET_INSTALL_VERSION packages."
            return 1
        fi
    else
        print_colored_v2 "ERROR" "Failed to add PPA ppa:deadsnakes/ppa."
        return 1
    fi
}

setup_venv() {
    VENV_PATH="./venv-${PROJECT_NAME}"
    CONTAINER_MODE=false

    # Check if running in a container
    if grep -qE "/docker|/lxc|/containerd" /proc/1/cgroup || [ -f /.dockerenv ]; then
        CONTAINER_MODE=true
        print_colored_v2 "INFO" "(container mode detected)"

        if [ -z "$DOCKER_VOLUME_PATH" ]; then
            show_help "error" "--docker_volume_path must be provided in container mode."
            exit 1
        fi

        VENV_SYMLINK_TARGET_PATH="${DOCKER_VOLUME_PATH}/venv/${PROJECT_NAME}"
        VENV_SYMLINK_TARGET_PATH_ARGS="--venv_symlink_target_path=${VENV_SYMLINK_TARGET_PATH}"
    else
        print_colored_v2 "INFO" "(host mode detected)"
        VENV_PATH="${VENV_PATH}-local"
        VENV_SYMLINK_TARGET_PATH_ARGS=""
    fi

    print_colored_v2 "INFO" "VENV_PATH: ${VENV_PATH}"

    RUN_SETUP_CMD="./scripts/setup_venv.sh --project_name=${PROJECT_NAME} --venv_path=${VENV_PATH} --dxrt_src_path=${DXRT_SRC_PATH} ${VENV_SYMLINK_TARGET_PATH_ARGS} ${FORCE_ARGS}"
    print_colored_v2 "INFO" "CMD : $RUN_SETUP_CMD"
    $RUN_SETUP_CMD >&2 || { print_colored_v2 "ERROR" "Setup script failed."; rm -rf $VENV_PATH; exit 1; }

    echo "$VENV_PATH"
}

main() {
    validate_path
    install_python
    setup_venv
}


# Parse arguments
while [ "$#" -gt 0 ]; do
    case "$1" in
        --dxrt_src_path=*)
            DXRT_SRC_PATH="${1#*=}"
            ;;
        --docker_volume_path=*)
            DOCKER_VOLUME_PATH="${1#*=}"
            ;;
        --force)
            FORCE_ARGS="--force"
            ;;
        --help)
            show_help
            ;;
        *)
            show_help "error" "Unknown option: $1"
            ;;
    esac
    shift
done

main

popd >&2
exit 0
