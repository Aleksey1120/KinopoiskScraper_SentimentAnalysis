# Apps

This package consists of two microservices designed to work together for the classification of reviews.
The system is built using Python, with communication between services handled via Kafka and Redis.

## Services

review_service:

- This is the FastAPI-based web server that handles incoming classification requests.
- It sends review texts to Kafka for processing and retrieves the classification results from Redis.

review_classifier:
- This service is responsible for performing the actual classification using a neural network.
- It subscribes to Kafka topics to receive review texts, processes them, and stores the results in Redis.

## Setup

From the root directory, run the following command to build and start the services using Docker Compose:

```
docker-compose build
docker-compose up -d
```

## Usage

Send a POST request to the Review Service's /predict endpoint with the review text. Example using requests lib:

```
import requests
response = requests.post('http://localhost:8080/predict', json={'text': 'Your review text here'})
print(response.json())
```
Response example:
```
{
    'sentiment': 'positive',
    'confidence': 0.517476499080658,
    'probability': 
    {
        'negative': 0.00668255053460598,
        'neutral': 0.4758409261703491,
        'positive': 0.517476499080658
    }
}
```

## Testing

The project includes tests to ensure that the services are functioning as expected. To run the tests:
```
python -m  unittest discover tests 
```