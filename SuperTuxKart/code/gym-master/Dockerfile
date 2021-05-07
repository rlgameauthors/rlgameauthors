ARG PYTHON_VER="3.8.6"
FROM python:$PYTHON_VER
RUN apt-get -y update && apt-get install -y unzip libglu1-mesa-dev libgl1-mesa-dev libosmesa6-dev xvfb patchelf ffmpeg scrot
# --- Install external games ---
RUN apt-get install -y bastet
# ------------------------------

COPY . /usr/local/gym/
WORKDIR /usr/local/gym/
RUN [ "$PYTHON_VER" != "3.8.1" ] && pip install .[all] || pip install .

ENTRYPOINT ["/usr/local/gym/bin/docker_entrypoint"]
CMD ["python", "examples/agents/random_agent.py", "bastet"]
