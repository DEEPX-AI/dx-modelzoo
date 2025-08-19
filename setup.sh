#!/bin/bash

SCRIPT_DIR=$(realpath "$(dirname "$0")")
PROJECT_ROOT=$(realpath -s "${SCRIPT_DIR}")

# color env settings
source ${PROJECT_ROOT}/scripts/color_env.sh
source ${PROJECT_ROOT}/scripts/common_util.sh

# Default values
PROJECT_NAME="dx-modelzoo"
RUNTIME_PATH=$(realpath -s "${PROJECT_ROOT}/../dx-runtime")
DXRT_SRC_PATH="${RUNTIME_PATH}/dx_rt"
DX_AS_PATH=$(realpath -s "${RUNTIME_PATH}/..")
ENABLE_DEBUG_LOGS=0   # New flag for debug logging
DOCKER_VOLUME_PATH=${DOCKER_VOLUME_PATH}
USE_FORCE=0
REUSE_VENV=1
FORCE_REMOVE_VENV=0
VENV_SYSTEM_SITE_PACKAGES_ARGS=""

# Global variables for script configuration
PYTHON_VERSION=""
MIN_PY_VERSION="3.11.0"
# VENV_PATH and VENV_SYMLINK_TARGET_PATH will be set dynamically in install_python_and_venv()
VENV_PATH=""
VENV_SYMLINK_TARGET_PATH=""
# User override options
VENV_PATH_OVERRIDE=""
VENV_SYMLINK_TARGET_PATH_OVERRIDE=""

pushd $PROJECT_ROOT >&2

# Function to display help message
show_help() {
    echo -e "Usage: ${COLOR_CYAN}$(basename "$0") [OPTIONS]${COLOR_RESET}"
    echo -e ""
    echo -e "Options:"
    echo -e "  ${COLOR_GREEN}[--dxrt_src_path=<path>]${COLOR_RESET}              Set DXRT source path (default: ${DXRT_SRC_PATH})"
    echo -e "  ${COLOR_GREEN}[--docker_volume_path=<path>]${COLOR_RESET}         Set Docker volume path (required in container mode)"
    echo -e ""
    echo -e "  ${COLOR_GREEN}[--venv_path=<path>]${COLOR_RESET}                  Set virtual environment path (default: PROJECT_ROOT/venv-${PROJECT_NAME})"
    echo -e "  ${COLOR_GREEN}[--venv_symlink_target_path=<dir>]${COLOR_RESET}    Set symlink target path for venv (ex: PROJECT_ROOT/../workspace/venv/${PROJECT_NAME})"
    echo -e ""
    echo -e "  ${COLOR_GREEN}[--system-site-packages]${COLOR_RESET}              Set venv '--system-site-packages' option."    
    echo -e "                                          - This option is applied only when venv is created. If you use '-venv-reuse', it is ignored. "
    echo -e ""
    echo -e "  ${COLOR_GREEN}[-f | --venv-force-remove]${COLOR_RESET}            If specified, force remove existing virtual environment at --venv_path before creation."
    echo -e "  ${COLOR_GREEN}[-r | --venv-reuse]${COLOR_RESET}                   If specified, reuse existing virtual environment at --venv_path if it's valid, skipping creation."
    echo -e ""
    echo -e "  ${COLOR_GREEN}[--force]${COLOR_RESET}                             Force overwrite if the file already exists"
    echo -e "                                          - This option is applied. --venv-force-remove option is enabled automatically."
    echo -e ""
    echo -e "  ${COLOR_GREEN}[--help]${COLOR_RESET}                              Display this help message and exit."
    echo -e ""
    echo -e "${COLOR_BOLD}Examples:${COLOR_RESET}"
    echo -e "  ${COLOR_YELLOW}$0"
    echo -e "  ${COLOR_YELLOW}$0 --dxrt_src_path=/path/to/dx_rt${COLOR_RESET}"
    echo -e "  ${COLOR_YELLOW}$0 --docker_volume_path=/path/to/docker/volume${COLOR_RESET}"
    echo -e ""
    echo -e "  ${COLOR_YELLOW}$0 --venv_path=./my_venv # Installs default Python, creates venv${COLOR_RESET}"
    echo -e "  ${COLOR_YELLOW}$0 --venv_path=./existing_venv --venv-reuse # Reuse existing venv${COLOR_RESET}"
    echo -e "  ${COLOR_YELLOW}$0 --venv_path=./old_venv --venv-force-remove # Force remove and recreate venv${COLOR_RESET}"
    echo -e "  ${COLOR_YELLOW}$0 --venv_path=./my_venv --venv_symlink_target_path=/tmp/actual_venv # Create venv at /tmp with symlink${COLOR_RESET}"
    echo -e ""

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

validate_environment() {
    echo -e "=== validate_environment() ${TAG_START} ==="

    # Handle --venv-force-remove and --venv-reuse conflicts
    if [ ${FORCE_REMOVE_VENV} -eq 1 ] && [ ${REUSE_VENV} -eq 1 ]; then
        print_colored "Cannot use both --venv-force-remove and --venv-reuse simultaneously. Please choose one." "ERROR" >&2
        exit 1
    fi

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

    echo -e "=== validate_environment() ${TAG_DONE} ==="
}

install_python_and_venv() {
    print_colored "--- Install Python and Create Virtual environment..... ---" "INFO"

    # Check if running in a container and set appropriate paths
    local CONTAINER_MODE=false
    
    # Check if running in a container
    if check_container_mode; then
        CONTAINER_MODE=true
        print_colored_v2 "INFO" "(container mode detected)"

        if [ -z "$DOCKER_VOLUME_PATH" ]; then
            show_help "error" "--docker_volume_path must be provided in container mode."
            exit 1
        fi

        # In container mode, use symlink to docker volume
        VENV_SYMLINK_TARGET_PATH="${DOCKER_VOLUME_PATH}/venv/${PROJECT_NAME}"
        VENV_PATH="${PROJECT_ROOT}/venv-${PROJECT_NAME}"
    else
        print_colored_v2 "INFO" "(host mode detected)"
        # In host mode, use local venv without symlink
        VENV_PATH="${PROJECT_ROOT}/venv-${PROJECT_NAME}-local"
        VENV_SYMLINK_TARGET_PATH=""
    fi

    # Override with user-specified options if provided
    if [ -n "${VENV_PATH_OVERRIDE}" ]; then
        VENV_PATH="${VENV_PATH_OVERRIDE}"
        print_colored_v2 "INFO" "Using user-specified VENV_PATH: ${VENV_PATH}"
    else
        print_colored_v2 "INFO" "Auto-detected VENV_PATH: ${VENV_PATH}"
    fi
    
    if [ -n "${VENV_SYMLINK_TARGET_PATH_OVERRIDE}" ]; then
        VENV_SYMLINK_TARGET_PATH="${VENV_SYMLINK_TARGET_PATH_OVERRIDE}"
        print_colored_v2 "INFO" "Using user-specified VENV_SYMLINK_TARGET_PATH: ${VENV_SYMLINK_TARGET_PATH}"
    elif [ -n "${VENV_SYMLINK_TARGET_PATH}" ]; then
        print_colored_v2 "INFO" "Auto-detected VENV_SYMLINK_TARGET_PATH: ${VENV_SYMLINK_TARGET_PATH}"
    fi

    local install_py_cmd_args=""

    if [ -n "${PYTHON_VERSION}" ]; then
        install_py_cmd_args+=" --python_version=$PYTHON_VERSION"
    fi

    if [ -n "${MIN_PY_VERSION}" ]; then
        install_py_cmd_args+=" --min_py_version=$MIN_PY_VERSION"
    fi

    if [ -n "${VENV_PATH}" ]; then
        install_py_cmd_args+=" --venv_path=$VENV_PATH"
    fi

    if [ -n "${VENV_SYMLINK_TARGET_PATH}" ]; then
        install_py_cmd_args+=" --symlink_target_path=$VENV_SYMLINK_TARGET_PATH"
    fi

    if [ ${USE_FORCE} -eq 1 ] || [ ${FORCE_REMOVE_VENV} -eq 1 ]; then
        install_py_cmd_args+=" --venv-force-remove"
    fi

    if [ ${REUSE_VENV} -eq 1 ]; then
        install_py_cmd_args+=" --venv-reuse"
    fi

    if [ -n "${VENV_SYSTEM_SITE_PACKAGES_ARGS}" ]; then
        install_py_cmd_args+=" ${VENV_SYSTEM_SITE_PACKAGES_ARGS}"
    fi

    # Pass the determined VENV_PATH and new options to install_python_and_venv.sh
    local install_py_cmd="${PROJECT_ROOT}/scripts/install_python_and_venv.sh ${install_py_cmd_args}"
    echo "CMD: ${install_py_cmd}"
    ${install_py_cmd} || {
        print_colored "Failed to Install Python and Create Virtual environment. Exiting." "ERROR"
        exit 1
    }

    print_colored "[OK] Completed to Install Python and Create Virtual environment." "INFO"
}

activate_venv() {
    echo -e "=== activate_venv() ${TAG_START} ==="

    # activate venv
    source ${VENV_PATH}/bin/activate
    if [ $? -ne 0 ]; then
        print_colored_v2 "ERROR" "Activate Virtual environment(${VENV_PATH}) failed! Please try installing again with the '--force' option. "
        print_colored_v2 "HINT" "Please run 'setup.sh --force' to set up and activate the environment first."
        exit 1
    fi

    echo -e "=== activate_venv() ${TAG_DONE} ==="
}

setup_dx_engine(){
    echo -e "=== setup_dx_engine() ${TAG_STRT} ==="
    ### Setup dx_rt python package
    #### 2. Install dx_engine (dx_rt Python package)
    pushd ${DXRT_SRC_PATH}
    ./build.sh --clean
    pip install ./python_package/.
    popd
    echo -e "=== setup_dx_engine() ${TAG_DONE} ==="
}

install_pip_packages(){
    echo -e "=== install_pip_packages() ${TAG_START:-[START]} ==="

    #### Install pip packages
    pip install -e ${PROJECT_ROOT}/.
    # failed check
    if [ $? -ne 0 ]; then
        echo -e "${TAG_ERROR:-[ERROR]} Install pip packages failed!"
        exit 1
    fi

    echo -e "=== install_pip_packages() ${TAG_DONE:-[DONE]} ==="
}

setup_project() {
    echo -e "=== setup_${PROJECT_NAME}() ${TAG_START} ==="

    if check_virtualenv; then
        setup_dx_engine
        install_pip_packages
    else
        if [ -d "$VENV_PATH" ]; then
            activate_venv
            setup_dx_engine
            install_pip_packages
        else
            print_colored_v2 "ERROR" "Virtual environment '${VENV_PATH}' is not exist."
            print_colored_v2 "HINT" "Please run 'setup.sh' to set up and activate the environment first."
        fi
    fi

    echo -e "=== setup_${PROJECT_NAME}() ${TAG_DONE} ==="
}

create_activation_script(){
    local ACTIVATE_SCRIPT="${PROJECT_ROOT}/activate_venv.sh"
    cat > "${ACTIVATE_SCRIPT}" << EOF
#!/bin/bash
# Auto-generated activation script for ${PROJECT_NAME}
# Generated by setup.sh on $(date)

VENV_PATH="${VENV_PATH}"

if [ -f "\${VENV_PATH}/bin/activate" ]; then
    echo "Activating virtual environment at: \${VENV_PATH}"
    source "\${VENV_PATH}/bin/activate"
    echo "Virtual environment activated!"
    echo "Current Python: \$(which python)"
    echo "To deactivate, run: deactivate"
else
    echo "Error: Virtual environment not found at \${VENV_PATH}"
    echo "Please run setup.sh first to create the virtual environment."
    exit 1
fi
EOF
    chmod +x "${ACTIVATE_SCRIPT}"
    echo -e "${TAG_INFO} Created activation script: ${ACTIVATE_SCRIPT}"
}

show_information_message(){
    # Create convenient activation script
    create_activation_script
    
    # Check if script was sourced (running in current shell)
    if [ "${BASH_SOURCE[0]}" != "${0}" ] || [ -n "${ZSH_VERSION}" ]; then
        echo -e "${TAG_INFO} Script was sourced. Activating virtual environment automatically..."
        source "${VENV_PATH}/bin/activate"
        echo -e "${TAG_SUCC} Virtual environment activated!"
        echo -e "${TAG_INFO} Current Python: $(which python)"
    else
        echo -e "${TAG_INFO} To activate the virtual environment, choose one of:"
        echo -e "${COLOR_BRIGHT_YELLOW_ON_BLACK}  source ${VENV_PATH}/bin/activate ${COLOR_RESET}  # Direct activation"
        echo -e "${COLOR_BRIGHT_YELLOW_ON_BLACK}  source ./activate_venv.sh ${COLOR_RESET}                # Using generated script"
        echo -e "${COLOR_BRIGHT_YELLOW_ON_BLACK}  source $(basename "$0") [OPTIONS] ${COLOR_RESET}        # Re-run with auto-activation"
    fi
}


setup_venv() {
    setup_project
    show_information_message
}


main() {
    validate_environment
    install_python_and_venv
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
        --venv_path=*)
            VENV_PATH_OVERRIDE="${1#*=}"
            ;;
        --venv_symlink_target_path=*)
            VENV_SYMLINK_TARGET_PATH_OVERRIDE="${1#*=}"
            ;;
        -f|--venv-force-remove)
            FORCE_REMOVE_VENV=1
            REUSE_VENV=0
            ;;
        -r|--venv-reuse)
            REUSE_VENV=1
            ;;
        --system-site-packages)
            VENV_SYSTEM_SITE_PACKAGES_ARGS="--system-site-packages"
            ;;
        --force)
            USE_FORCE=1
            FORCE_REMOVE_VENV=1
            REUSE_VENV=0
            ;;
        --verbose)
            ENABLE_DEBUG_LOGS=1
            VERBOSE_ARGS="--verbose"
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
