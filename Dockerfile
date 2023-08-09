FROM python:3.10.6-slim-bullseye

ENV DEBIAN_FRONTEND=noninteractive

COPY requirements.txt /tmp/requirements.txt 

RUN apt-get update && apt-get install -y git wget software-properties-common gnupg && \
    cd /tmp && \
    wget https://developer.download.nvidia.com/compute/cuda/11.7.1/local_installers/cuda-repo-debian11-11-7-local_11.7.1-515.65.01-1_amd64.deb && \
    dpkg -i /tmp/cuda-repo-debian11-11-7-local_11.7.1-515.65.01-1_amd64.deb && \
    rm /tmp/cuda-repo-debian11-11-7-local_11.7.1-515.65.01-1_amd64.deb && \
    cp /var/cuda-repo-debian11-11-7-local/cuda-*-keyring.gpg /usr/share/keyrings/ && \
    add-apt-repository contrib && \
    apt-get update && \
    apt-get -y install cuda && \
    apt -y remove nvidia-* && \
    rm -rf /var/cuda-repo-debian11-11-6-local && \
    pip install --no-cache-dir -r requirements.txt

COPY stylingapp.py .

# Set the command to run the Python script when the container starts
CMD ["streamlit", "run", "stylingapp.py"]
