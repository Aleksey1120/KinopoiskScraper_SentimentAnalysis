import asyncio
import json
import os

from kafka import KafkaConsumer
import redis.asyncio as aioredis

from apps.review_classifier.model import Model
from apps.review_classifier.config import TTL, MAX_POLL_RECORDS, CLIENT_ID, GROUP_ID

topic_name = os.getenv('REQUEST_TOPIC')
kafka_url = f'{os.getenv("KAFKA_HOST")}:{os.getenv("KAFKA_PORT")}'
redis_url = f'redis://{os.getenv("REDIS_HOST")}:{os.getenv("REDIS_PORT")}'


async def publish_result(key, result, redis_client):
    await redis_client.xadd(key, {'result': json.dumps(result)})
    await redis_client.expire(key, TTL)


async def publish_results(keys, results, redis_client):
    tasks = []
    for key, result in zip(keys, results):
        task = asyncio.create_task(publish_result(key, result, redis_client))
        tasks.append(task)
    await asyncio.gather(*tasks)


async def main():
    consumer = KafkaConsumer(
        client_id=CLIENT_ID,
        group_id=GROUP_ID,
        bootstrap_servers=kafka_url,
        key_deserializer=lambda k: k.decode(),
        value_deserializer=lambda v: v.decode(),
        max_poll_records=MAX_POLL_RECORDS,
        auto_offset_reset='latest'
    )
    consumer.subscribe([topic_name])
    redis_client = await aioredis.from_url(redis_url)

    classification_model = Model()
    try:
        while True:
            messages = consumer.poll(1000, max_records=MAX_POLL_RECORDS)
            if messages:
                texts = []
                keys = []
                for tp, msgs in messages.items():
                    for message in msgs:
                        keys.append(message.key)
                        texts.append(message.value)
                results = classification_model.predict(texts)
                await publish_results(keys, results, redis_client)
            await asyncio.sleep(0.001)
    except asyncio.exceptions.CancelledError:
        print('End event loop.')
    finally:
        await redis_client.aclose()
        consumer.close()


if __name__ == '__main__':
    asyncio.run(main())
