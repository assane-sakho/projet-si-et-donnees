# Specify your base image
FROM tensorflow/tensorflow

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

# create a work directory
RUN mkdir /app
# navigate to this work directory
WORKDIR /app
#Copy all files
COPY . /app
RUN python -m pip install --upgrade pip
# Install dependencies
RUN pip install -r requirements.txt
# Run
CMD ["python","waitress_server.py"]
