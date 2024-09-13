# Use a Python base image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

RUN  apt-get update && apt-get install -y wget 

# Copy the requirements file and install dependencies
COPY receiver/requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Copy the application code into the container
COPY receiver/receiver.py receiver.py
COPY receiver/processor.py processor.py
COPY receiver/transforms.py transforms.py
COPY receiver/models models

# Copy the utils directory into the container
COPY utils utils

# Copy the uploads directory into the container
COPY receiver/uploads uploads

# Copy config.json file into the container
COPY receiver/config.json config.json

# Copy config.json file into the container
WORKDIR /app/receiver/weights
RUN wget --no-check-certificate "https://drive.google.com/uc?export=download&id=1NSnpFnXlNunrFN98mQd-ZIHIqE3y8RZU" -O model.pth.tar

# Predownload Torchvision Models
RUN ["python", "-c", "from torchvision import models as m; m.resnet152(weights=m.ResNet152_Weights.IMAGENET1K_V1)"]

# Command to run the application
WORKDIR /app
CMD ["python", "receiver.py"]
