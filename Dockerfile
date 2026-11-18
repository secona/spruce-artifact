FROM nvidia/cuda:13.2.0-devel-ubuntu24.04@sha256:f9492f2eea77fbc3d0c14fa8738f35946b42da72917bf5959d284ca39b4f209a

ENV CUDA_PATH=/usr/local/cuda

WORKDIR /work

# Installed required deps from apt.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential ca-certificates cmake git ninja-build python3-venv wget
ENV PATH="/opt/venv/bin:${PATH}"

# Install required Python dependencies.
COPY requirements.txt /tmp/requirements.txt
RUN python3 -m venv /opt/venv \
    && pip install --no-cache-dir -r /tmp/requirements.txt

# Build LLVM.
ARG BUILD_JOBS=4
RUN git clone --depth 1 --branch llvmorg-23.1.6 https://github.com/llvm/llvm-project.git /src/llvm-project \
    && cmake -S /src/llvm-project/llvm -B /tmp/build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/llvm \
      -DLLVM_ENABLE_PROJECTS=mlir '-DLLVM_TARGETS_TO_BUILD=X86;NVPTX' \
      -DMLIR_ENABLE_CUDA_RUNNER=ON -DCMAKE_CUDA_ARCHITECTURES=86 \
      -DCMAKE_LIBRARY_PATH=/usr/local/cuda/lib64/stubs \
    && cmake --build /tmp/build --parallel "${BUILD_JOBS}" --target \
      install-mlir-opt install-mlir-translate install-llc install-mlir-headers \
      install-mlir_cuda_runtime install-mlir_c_runner_utils \
      install-mlir_float16_utils install-mlir_apfloat_wrappers \
    && rm -rf /tmp/build /src/llvm-project
ENV LLVM_DIR=/opt/llvm

# Get the dataset.
COPY dataset/*.py dataset/*.sh dataset/
RUN bash dataset/download.sh \
    && bash dataset/extract.sh \
    && python -m dataset.generate_inputs \
    && chmod -R a+rX dataset \
    && rm dataset/*.gz

# Copy remaining directories.
COPY benchmarks benchmarks
COPY systems systems

CMD ["bash"]
