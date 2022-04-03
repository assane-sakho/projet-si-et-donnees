# Example of Dockerfile

FROM python:3.8.5-alpine3.12

WORKDIR /app

EXPOSE $PORT
ENV FLASK_APP=waitress_server.py

COPY . /app
RUN pip install -r requirements.txt

ENTRYPOINT [ "flask"]
CMD [ "run", "--host", "0.0.0.0" ]