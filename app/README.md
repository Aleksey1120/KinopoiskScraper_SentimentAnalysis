# App

This module contains applications that interact with the sentiment analysis models trained in the nn_classifier module.

## Applications

classifier_app: run a web server that provides an API for accessing the sentiment analysis models. You can use this API to integrate sentiment analysis into other applications or services.

bot_app: run a Telegram bot that allows users to request sentiment analysis of movie reviews directly through the Telegram interface.

## Usage

### classifier_app

To run locally, you can use commands:
```
cd app/classifier_app
uvicorn api:app --host 0.0.0.0 --port 8080
```
Running in docker container:
```
docker build -t classifier_app ./app/classifier_app
docker run -p 8080:8080 classifier_app
```
### bot_app

To run locally, you can use commands:
```
cd app/bot_app
python api.py
```
The application uses the TOKEN and CLASSIFIER_URL environment variables.

Running in docker container:
```
docker build -t bot_app ./app/bot_app
docker run -p 8080:8080 -e TOKEN=token -e CLASSIFIER_URL=classifier_url bot_app
```

### Docker compose

You can run both applications using Docker Compose.

1. Create .env file in the root directory of the project with two variables:
```
TOKEN=your_bot_token
CLASSIFIER_URL=http://classifier_app:8080/
```
2. Run command
```
docker compose up
```
