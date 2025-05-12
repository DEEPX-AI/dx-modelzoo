#!/bin/bash
SCRIPT_DIR=$(realpath "$(dirname "$0")")


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

install_pip_packages

exit 0

