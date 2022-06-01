FROM tensorflow/tensorflow:2.7.0-gpu-jupyter
WORKDIR /GANime
ENV PROJECT_DIR=/GANime
COPY requirements.txt /GANime/requirements.txt
RUN pip install -r requirements.txt
COPY . .
RUN pip install -e .
EXPOSE 8888