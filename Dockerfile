FROM nvcr.io/nvidia/deepstream:9.0-triton-multiarch

# Install multimedia codecs needed by nvurisrcbin (gst-plugins-bad etc.)
RUN /opt/nvidia/deepstream/deepstream/user_additional_install.sh

# Install libmosquitto1 (required by NvDCF tracker at runtime)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libmosquitto1 \
    && rm -rf /var/lib/apt/lists/*

ENV NVIDIA_DRIVER_CAPABILITIES=video,compute,utility,graphics

WORKDIR /app
COPY requirements.txt requirements.txt

# Install pyservicemaker from the bundled wheel + Python deps.
RUN PSMAKER_WHL="$(find /opt/nvidia/deepstream -path '*/service-maker/python/pyservicemaker*.whl' | head -n1)" \
    && pip3 install --no-cache-dir "$PSMAKER_WHL" -r requirements.txt

# Copy project source. Docker Compose mounts ./models over /app/models so
# TensorRT engines built in the container persist next to their source model.
COPY configs/   configs/
COPY src/       src/
COPY milestones/ milestones/
COPY models/    models/
COPY scripts/   scripts/

# Main demo entrypoint. Docker Compose overrides only the source list.
CMD ["python3", "-m", "src.main"]
