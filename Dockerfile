FROM ubuntu:24.04
RUN apt-get update && apt-get install -y --no-install-recommends \
    g++-14 gcc-14 cmake make \
    clang-tidy clang-format cppcheck iwyu \
    libfmt-dev \
    python3 python3-pip python3-dev python3-venv \
    git ca-certificates \
  && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-14 100 \
  && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-14 100 \
  && git clone --depth 1 https://github.com/kthohr/gcem.git /tmp/gcem \
  && cmake -S /tmp/gcem -B /tmp/gcem/build -DCMAKE_INSTALL_PREFIX=/usr/local \
  && cmake --install /tmp/gcem/build \
  && rm -rf /tmp/gcem /var/lib/apt/lists/*
