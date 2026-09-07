# Based on https://github.com/napari/napari/blob/main/dockerfile
FROM python:3.12-slim-bookworm AS muvis-align

ARG DEBIAN_FRONTEND=noninteractive

# install python resources + graphical libraries used by qt and vispy
RUN apt-get update && \
    apt-get install -qqy  \
        build-essential \
        git \
        libglib2.0-0 \
        mesa-utils \
        libglx-mesa0 \
        # tlambert03/setup-qt-libs
        libegl1 \
        libdbus-1-3 \
        libxkbcommon-x11-0 \
        libxcb-icccm4 \
        libxcb-image0 \
        libxcb-keysyms1 \
        libxcb-randr0 \
        libxcb-render-util0 \
        libxcb-xinerama0 \
        libxcb-xinput0 \
        libxcb-xfixes0 \
        x11-utils \
        libxcb-cursor0 \
        libopengl0 \
        # other/remaining
        libfontconfig1 \
        libxrender1 \
        libxi6 \
        libxcb-shape0 \
        && apt-get clean

# Set working directory
WORKDIR /app

# Copy dependency declarations first so source changes do not invalidate
# the dependency installation layers.
COPY requirements.txt .

# Install dependencies
RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install napari[all]

# Copy project files after dependencies have been installed.
COPY run.py .
COPY pyproject.toml .
COPY src/ src/

RUN --mount=type=bind,source=.git,target=/app/.git \
    pip install .

ENTRYPOINT ["python3", "-m", "napari", "--plugin", "muvis-align"]


FROM muvis-align AS muvis-align-xpra

ARG DEBIAN_FRONTEND=noninteractive

# Install Xpra and dependencies
# Remember to update the xpra.sources link for any change in distro version
RUN apt-get update && apt-get install -y wget gnupg2 apt-transport-https \
    software-properties-common ca-certificates && \
    wget -O "/usr/share/keyrings/xpra.asc" https://xpra.org/xpra.asc && \
    wget -O "/etc/apt/sources.list.d/xpra.sources" https://raw.githubusercontent.com/Xpra-org/xpra/master/packaging/repos/bookworm/xpra.sources


RUN apt-get update && \
    apt-get install -yqq \
        xpra \
        xvfb \
        menu-xdg \
        xdg-utils \
        xterm \
        sshfs \
        x11-xkb-utils \
        xkb-data && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV DISPLAY=:100
ENV XPRA_PORT=9876
ENV XPRA_START="python3 -m napari --with muvis-align"
ENV XPRA_EXIT_WITH_CHILDREN="yes"
ENV XPRA_EXIT_WITH_CLIENT="no"
ENV XPRA_XVFB_SCREEN="1920x1080x24+32"
ENV XDG_RUNTIME_DIR=/tmp/runtime-muvis
# Debian's Xpra module is installed for /usr/bin/python3. Xpra uses this
# interpreter for helper processes such as the IBus daemonizer.
ENV XPRA_PYTHON_EXECFILE_COMMAND=/usr/bin/python3
EXPOSE 9876

CMD echo "Launching napari on Xpra. Connect via http://localhost:$XPRA_PORT or $(hostname -i):$XPRA_PORT"; \
    PATH=/usr/bin:/usr/local/bin:$PATH xpra start \
    --bind-tcp=0.0.0.0:$XPRA_PORT \
    --html=on \
    --socket-dir="$XDG_RUNTIME_DIR/xpra" \
    --mdns=no \
    --dbus=no \
    --printing=no \
    --webcam=no \
    --speaker=disabled \
    --microphone=disabled \
    --desktop-scaling=auto \
    --resize-display=yes \
    --dpi=96 \
    --start-child="$XPRA_START" \
    --exit-with-children="$XPRA_EXIT_WITH_CHILDREN" \
    --exit-with-client="$XPRA_EXIT_WITH_CLIENT" \
    --daemon=no \
    --xvfb="/usr/bin/Xvfb +extension Composite -screen 0 $XPRA_XVFB_SCREEN -dpi 96 -nolisten tcp -noreset" \
    --pulseaudio=no \
    --notifications=no \
    --bell=no \
    $DISPLAY

ENTRYPOINT []

# Build:
# docker build -t muvis-align-xpra .

# Run:
# docker run -v "D:\slides:/data" -p 9876:9876 muvis-align-xpra

# Build & push (tagged with the current GitHub release version, and "latest",
# for xpra-pull.sh) - see docker-build-push.sh:
# docker login quay.io
# ./docker-build-push.sh

# apptainer remote login --username [username] docker://quay.io
# apptainer pull docker://quay.io/ccp-volume-em/muvis-align-xpra:v1.0.0
# apptainer run --mount type=bind,src=/camp/project/proj-ccp-vem/datasets,dst=/data --net --network-args "portmap=9876:9876" muvis-align-xpra_v1.0.0.sif
