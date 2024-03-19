FROM python:3.11.8

WORKDIR /app

COPY . .

EXPOSE 8080

RUN ["pip", "install", "-r", "requirements-torch.txt"]

RUN ["pip", "install", "-r", "requirements.txt"]

ENTRYPOINT [ "uvicorn", "api:app" , "--host", "0.0.0.0", "--port", "8080"]