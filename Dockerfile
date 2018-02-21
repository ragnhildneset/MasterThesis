FROM gcr.io/tensorflow/tensorflow:latest-gpu

# install dependencies from debian packages
RUN apt-get update -qq \
 && apt-get install --no-install-recommends -y \
    libsm6 \
    libxext6 \
    libfontconfig1 \
    libxrender1 \
    python-tk \
    python-pillow \
    vim

# install dependencies from python packages
RUN pip --no-cache-dir install \
    opencv-python \
    seaborn \
    scikit-learn \
    keras \
    keras-vis

# install your app
RUN mkdir -p /ai
COPY . /ai
RUN chmod +x /ai/model.py

WORKDIR /ai
CMD ["/bin/bash"]
