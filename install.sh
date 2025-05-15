#!/bin/bash
PROJECT_NAME="dx-modelzoo"
SCRIPT_DIR=$(realpath "$(dirname "$0")")
PROJECT_ROOT="${SCRIPT_DIR}"
VENV_PATH="${PROJECT_ROOT}/venv-${PROJECT_NAME}"

source ${SCRIPT_DIR}/scripts/color_env.sh

check_virtualenv() {
    if [ -n "$VIRTUAL_ENV" ]; then
        venv_name=$(basename "$VIRTUAL_ENV")
        if [ "$venv_name" = "dx-modelzoo" ]; then
            echo "✅ Virtual environment 'dx-modelzoo' is currently active."
            return 0
        else
            echo "⚠️ A different virtual environment '$venv_name' is currently active."
            return 1
        fi
    else
        echo "❌ No virtual environment is currently active."
        return 1
    fi
}

install_pip_packages(){
    echo -e "=== install_pip_packages() ${TAG_START:-[START]} ==="
    pushd ${SCRIPT_DIR}

    #### Install pip packages
    pip install -e .
    # failed check
    if [ $? -ne 0 ]; then
        echo -e "${TAG_ERROR:-[ERROR]} Install pip packages failed!"
        exit 1
    fi

    popd
    echo -e "=== install_pip_packages() ${TAG_DONE:-[DONE]} ==="
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

main() {
    if check_virtualenv; then
        install_pip_packages
    else
        if [ -d "$VENV_PATH" ]; then
            activate_venv
            install_pip_packages
        else
            echo -e "${TAG_ERROR:-[ERROR]}${COLOR_BRIGHT_RED_ON_BLACK} Virtual environment 'venv-dx-modelzoo' is not exist.\nPlease run 'setup.sh' to set up and activate the environment first.${COLOR_RESET}"
        fi
    fi
}

main

exit 0

