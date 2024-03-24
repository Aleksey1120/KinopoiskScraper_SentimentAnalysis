FROM python:3.11.8

WORKDIR /app

COPY . .

EXPOSE 8000

RUN ["pip", "install", "-r", "requirements.txt"]

ENTRYPOINT [ "python", "api.py"]