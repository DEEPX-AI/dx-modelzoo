ARG PYTORCH_TAG=24.04-py3
FROM nvcr.io/nvidia/pytorch:${PYTORCH_TAG}

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    python3.10-venv \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /app

RUN echo "Attempting to clone repository into /app/ultralytics..."
RUN GIT_CURL_VERBOSE=1 GIT_TRACE=1 git clone https://github.com/ultralytics/ultralytics.git /app/ultralytics


WORKDIR /app/ultralytics

RUN python3 -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

RUN /app/venv/bin/pip install --no-cache-dir --upgrade pip
RUN /app/venv/bin/pip install --no-cache-dir ultralytics

ARG USERNAME=deepx
ARG USER_UID=1000
ARG USER_GID=1000

RUN groupadd --gid $USER_GID $USERNAME && \
    useradd --uid $USER_UID --gid $USER_GID -m -s /bin/bash $USERNAME

RUN chown -R $USERNAME:$USERNAME /app
USER $USER_UID:$USER_GID