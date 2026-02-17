# FROM ubuntu:24.04
FROM nvidia/cuda:12.9.1-base-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

# Base + build deps (pkg-config needed for PyAV; wget/xz-utils/nasm for building FFmpeg; cmake required for some Python deps)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-dev \
        git \
        curl \
        build-essential \
        pkg-config \
        wget \
        xz-utils \
        nasm \
        cmake \
        libnuma-dev \
        rdma-core \
        ibverbs-providers \
        libibverbs1 && \
    rm -rf /var/lib/apt/lists/*


RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libegl1 \
        libgl1 \
        libopengl0 \
        libglvnd0 \
        libgles2 \
        mesa-utils \
        libx11-6 \
        libxext6 \
        libxrender1 \
        libsm6 \
        libice6 && \
    rm -rf /var/lib/apt/lists/*

# Build and install FFmpeg 7 from source (PyAV 14.4.0 requires FFmpeg 7)
ARG FFMPEG_VERSION=7.0.2
RUN wget -q https://ffmpeg.org/releases/ffmpeg-${FFMPEG_VERSION}.tar.xz -O /tmp/ffmpeg.tar.xz && \
    tar xf /tmp/ffmpeg.tar.xz -C /tmp && \
    cd /tmp/ffmpeg-${FFMPEG_VERSION} && \
    ./configure --prefix=/usr/local --enable-shared --disable-static --enable-pic && \
    make -j$(nproc) && \
    make install && \
    cd / && \
    rm -rf /tmp/ffmpeg.tar.xz /tmp/ffmpeg-${FFMPEG_VERSION} && \
    echo "/usr/local/lib" > /etc/ld.so.conf.d/ffmpeg.conf && \
    ldconfig

# So PyAV build finds FFmpeg 7 (not system FFmpeg 6)
ENV PKG_CONFIG_PATH=/usr/local/lib/pkgconfig
ENV LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH}

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Set working directory
WORKDIR /workspace

# Copy project files
COPY . /workspace/

WORKDIR /workspace
CMD ["/bin/bash"]
