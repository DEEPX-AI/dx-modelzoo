#!/bin/bash
PROJECT_NAME="PROJECT_NAME"
SCRIPT_DIR=$(realpath "$(dirname "$0")")
PROJECT_ROOT=$(realpath -s "${SCRIPT_DIR}/..")
RUNTIME_PATH=$(realpath -s "${PROJECT_ROOT}/../dx-runtime")
DXRT_SRC_PATH="${RUNTIME_PATH}/dx_rt"
VENV_PATH="${PROJECT_ROOT}/venv-${PROJECT_NAME}"
VENV_SYMLINK_TARGET_PATH=""
VENV_MAKE_ARGS=""
USE_FORCE=0
SKIP_INSTALL_PIP_PACKAGE=0
PYTHON_EXE=""

# color env settings
source ${SCRIPT_DIR}/color_env.sh

# Function to display help message
show_help()
{
    echo "Usage: $(basename "$0") [OPTIONS]"
    echo "Example: $0 --project_name="my_project" --dxrt_src_path=/deepx/dx_rt --venv_path=./venv-${PROJECT_NAME} --venv_symlink_target_path=../workspace/venv/${PROJECT_NAME}"
    echo "Options:"
    echo "  --project_name=<str>                  Set Project name"
    echo "  --dxrt_src_path=<dir>                 Set DXRT source path (default: /deepx/dx_rt/)"
    echo "  --venv_path=<dir>                     Set virtual environment path (default: PROJECT_ROOT/venv-${PROJECT_NAME})"
    echo "  [--venv_symlink_target_path=<dir>]    Set symlink target path for venv (ex: PROJECT_ROOT/../workspace/venv/${PROJECT_NAME})"
    echo "  [--system-site-packages]              Set venv '--system-site-packages' option"
    echo "  [--force]                             Force overwrite if the file already exists"
    echo "  [--help]                              Show this help message"

    if [ "$1" == "error" ] && [[ ! -n "$2" ]]; then
        echo -e "${TAG_ERROR} Invalid or missing arguments."
        exit 1
    elif [ "$1" == "error" ] && [[ -n "$2" ]]; then
        echo -e "${TAG_ERROR} $2"
        exit 1
    elif [[ "$1" == "warn" ]] && [[ -n "$2" ]]; then
        echo -e "${TAG_WARN} $2"
        return 0
    fi
    exit 0
}

make_venv(){
    echo -e "=== make_venv() ${TAG_START} ==="
    #### 1. Set up Virtual Environment

    # Print VENV_PATH for verification
    echo "Using virtual environment at: $VENV_PATH"

    # setting venv location
    VENV_ORIGIN_DIR="$VENV_PATH"

    if [ -n "$VENV_SYMLINK_TARGET_PATH" ]; then
        # if '--venv_symlink_target_path' option is exist.
        VENV_ORIGIN_DIR="$VENV_SYMLINK_TARGET_PATH"
        echo "creating python venv to --venv_symlink_target_path: $VENV_ORIGIN_DIR"
    else
        echo "creating python venv to this path: $VENV_ORIGIN_DIR"
    fi

    if [ -e "$VENV_ORIGIN_DIR" ] && [ "$USE_FORCE" -eq 0 ]; then
        echo -e "${TAG_INFO} === MAKE VENV SKIP ==="
        echo -e "${TAG_INFO} ** python venv path($VENV_ORIGIN_DIR) is already exist. so, skip to make venv **"
        SKIP_INSTALL_PIP_PACKAGE=1
    else
        if [ -e "$VENV_ORIGIN_DIR" ] && [ "$USE_FORCE" -eq 1 ]; then
            echo -e "${TAG_INFO} The '--force' option is enabled. so, remove previous python venv (${VENV_ORIGIN_DIR})"
            rm -rf "$VENV_ORIGIN_DIR"
        fi

        for ver in 13 12 11; do
            py_cmd="python3.$ver"
            if command -v "$py_cmd" &>/dev/null; then
                PYTHON_EXE="$py_cmd"
                break
            fi
        done

        if [ -z "$PYTHON_EXE" ]; then
            echo "ERROR: Could not determine a suitable Python executable." >&2
            exit 1
        fi

        # create venv
        "$PYTHON_EXE" -m venv ${VENV_ORIGIN_DIR} ${VENV_MAKE_ARGS}

        # create venv failed check
        if [ $? -ne 0 ]; then
            echo -e "${TAG_ERROR} Creation venv failed! Please try installing again with the '--force' option."
            rm -rf "$VENV_ORIGIN_DIR"
            echo -e "${TAG_ERROR} === MAKE VENV FAIL ==="
            exit 1
        fi

        echo "Creation venv complete."
    fi
}

make_symlink() {
    # If '--symlink_target_path' option is set, create a symbolic link
    if [ -n "$VENV_SYMLINK_TARGET_PATH" ]; then
        echo -e "=== make_symlink() ${TAG_START} ==="

        if [ "$USE_FORCE" -eq 1 ]; then
            echo -e "${TAG_INFO} The '--force' option is enabled. so, remove previous python venv symlink (${VENV_PATH})"
            rm -rf "$VENV_PATH"
        fi

        local prev_symlink_target_path="$(readlink -e "$VENV_PATH")"
        local new_symlink_target_path="$(readlink -e "$VENV_SYMLINK_TARGET_PATH")"

        # If the symlink already exists
        if [ -L "$VENV_PATH" ] && [ -e "${prev_symlink_target_path}" ]; then
            # If the current symlink target matches the new target, no further action is needed
            if [ "${prev_symlink_target_path}" == "${new_symlink_target_path}" ]; then
                echo -e "${TAG_INFO} ${VENV_PATH} is already properly configured. To force reset, use the '--force' option."
                show_information_message
                exit 0
            # If the current symlink target is different, delete the existing symlink and recreate it
            else
                echo -e "${TAG_INFO} Existing symlink ${VENV_PATH} found but target mismatch. Deleting symlink and recreating."
                echo -e "    - Previous target path: ${prev_symlink_target_path}"
                echo -e "    - New target path: ${new_symlink_target_path}"
                rm -rf "$VENV_PATH"
            fi
        else
            echo -e "${TAG_INFO} Creating symlink to: ${VENV_PATH}"
            echo -e "   - VENV_PATH: ${VENV_PATH}"
            echo -e "   - VENV_SYMLINK_TARGET_PATH: ${VENV_SYMLINK_TARGET_PATH}"
            echo -e "   - prev_symlink_target_path: ${prev_symlink_target_path}"
            echo -e "   - new_symlink_target_path: ${new_symlink_target_path}"
        fi

        # Ensure the parent directory exists
        mkdir -p "$(dirname "$VENV_PATH")"

        # Create the new symbolic link
        VENV_SYMLINK_TARGET_REAL_PATH=$(readlink -f $VENV_SYMLINK_TARGET_PATH)
        ln -s "$VENV_SYMLINK_TARGET_REAL_PATH" "$VENV_PATH"

        # Check if symlink creation failed
        if [ $? -ne 0 ]; then
            echo -e "${TAG_ERROR} Failed to create symlink! Please try again using the '--force' option."
            rm -rf "$VENV_PATH"
            echo -e "${TAG_ERROR} === MAKE SYMLINK FAIL ==="
            exit 1
        fi
        echo "Created symbolic link: $VENV_PATH -> $VENV_SYMLINK_TARGET_REAL_PATH"
        echo -e "=== make_symlink() ${TAG_DONE} ==="
    else
        echo -e "${TAG_INFO} '--symlink_target_path' option not set. Skipping symlink creation."
    fi
}

activate_venv() {
    echo -e "=== activate_venv() ${TAG_START} ==="

    # activate venv
    source ${VENV_PATH}/bin/activate
    if [ $? -ne 0 ]; then
        echo -e "${TAG_ERROR} Activate venv failed! Please try installing again with the '--force' option."
        rm -rf "$VENV_PATH"
        echo -e "${TAG_ERROR} === ACTIVATE VENV FAIL ==="
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
    pushd ${DXRT_SRC_PATH}/python_package
    pip install .
    popd
    popd
    echo -e "=== setup_dx_engine() ${TAG_DONE} ==="
}

setup_project(){
    echo -e "=== setup_${PROJECT_NAME}() ${TAG_STRT} ==="
    pushd ${PROJECT_ROOT}

    #### Install packages
    eval ${PROJECT_ROOT}/install.sh

    popd
    echo -e "=== setup_${PROJECT_NAME}() ${TAG_DONE} ==="
}

function show_information_message(){
    echo -e "${TAG_INFO} To activate the virtual environment, run:"
    echo -e "${COLOR_BRIGHT_YELLOW_ON_BLACK}  source ${VENV_PATH}/bin/activate ${COLOR_RESET}"
}

main(){
    if [ "${PROJECT_NAME}" == "PROJECT_NAME" ]; then
        show_help "error" "'--project_name' option must be set."
    fi

    # Check if DXRT_SRC_PATH exists
    if [ ! -d $DXRT_SRC_PATH ]; then
        show_help "error" "DXRT_SRC_PATH ($DXRT_SRC_PATH) does not exist."
    fi

    make_venv
    make_symlink
    if [ ${SKIP_INSTALL_PIP_PACKAGE} -eq 0 ]; then
        activate_venv
        setup_dx_engine
        setup_project
    fi
    show_information_message
}

# Parse arguments
while [ "$#" -gt 0 ]; do
    case "$1" in
        --project_name=*)
            PROJECT_NAME="${1#*=}"
            VENV_PATH="${PROJECT_ROOT}/venv-${PROJECT_NAME}"
            ;;
        --dxrt_src_path=*)
            DXRT_SRC_PATH="${1#*=}"
            ;;
        --venv_path=*)
            VENV_PATH="${1#*=}"

            # Symbolic link cannot be created when output_dir is the current directory.
            VENV_REAL_DIR=$(readlink -f "$VENV_PATH")
            CURRENT_REAL_DIR=$(readlink -f "./")
            if [ "$VENV_REAL_DIR" == "$CURRENT_REAL_DIR" ]; then
                echo -e "${TAG_ERROR} '--venv_path' is the same as the current directory. Please specify a different directory."
                exit 1
            fi
            ;;
        --venv_symlink_target_path=*)
            VENV_SYMLINK_TARGET_PATH="${1#*=}"
            ;;
        --system-site-packages)
            VENV_MAKE_ARGS="--system-site-packages"
            ;;
        --force)
            USE_FORCE=1
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

exit 0
