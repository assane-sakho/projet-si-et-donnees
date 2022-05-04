# Specify your base image
FROM cudachen/mask-rcnn-docker

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

# create a work directory
RUN mkdir /app
# navigate to this work directory
WORKDIR /app
#Copy all files
COPY . /app
# RUN python -m pip install --upgrade pip
# Install dependencies
RUN pip3 install -r requirements.txt
# Run
# CMD ["ls", "/data"]
CMD ["python3","test_predict.py"]
