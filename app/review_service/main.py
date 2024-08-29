import json
import uuid
from typing import Dict

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from aiokafka import AIOKafkaProducer
from redis import asyncio as aioredis

classification_request_topic = 'classification_request'
kafka_server = 'localhost:9092'
redis_url = 'redis://localhost:6379'

app = FastAPI()
kafka_producer = None
redis_client = None


class SentimentRequest(BaseModel):
    text: str


class SentimentResponse(BaseModel):
    sentiment: str
    confidence: float
    probability: Dict[str, float]


@app.on_event('startup')
async def startup_event():
    global kafka_producer, redis_client

    kafka_producer = AIOKafkaProducer(
        bootstrap_servers=kafka_server,
        key_serializer=lambda k: k.encode(),
        value_serializer=lambda v: v.encode(),
    )
    await kafka_producer.start()

    redis_client = await aioredis.from_url(redis_url)


@app.on_event('shutdown')
async def shutdown_event():
    global kafka_producer, redis_client

    if kafka_producer:
        await kafka_producer.stop()

    if redis_client:
        await redis_client.aclose()


async def publish_classification_request(text, key):
    await kafka_producer.send_and_wait(
        classification_request_topic,
        key=key,
        value=text
    )


async def receive_classification_result(key):
    message = await redis_client.xread({key: '0'}, block=15_000, count=1)
    if message:
        (_, ((message_id, result),)) = message[0]
        result = json.loads(result[b'result'].decode())
    else:
        result = None
    return result


@app.post('/predict', response_model=SentimentResponse)
async def predict(request: SentimentRequest):
    key = uuid.uuid4().hex
    await publish_classification_request(request.text, key)
    classification_result = await receive_classification_result(key)
    if classification_result is not None:
        return SentimentResponse(
            sentiment=classification_result['sentiment'],
            confidence=classification_result['confidence'],
            probability=classification_result['probability']
        )
    else:
        return JSONResponse(
            content={'error': 'Classification result is taking too long to process'},
            status_code=504
        )
