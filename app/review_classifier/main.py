import asyncio
import json

from kafka import KafkaConsumer
import redis.asyncio as aioredis

from model import Model

topic_name = 'classification_request'
client_id = 'client_1'
group_id = 'group_1'
kafka_server = 'localhost:9092'
redis_url = 'redis://localhost:6379'
max_poll_records = 64
ttl = 60 * 15


async def publish_result(key, result, redis_client):
    await redis_client.xadd(key, {'result': json.dumps(result)})
    await redis_client.expire(key, ttl)


async def publish_results(keys, results, redis_client):
    tasks = []
    for key, result in zip(keys, results):
        task = asyncio.create_task(publish_result(key, result, redis_client))
        tasks.append(task)
    await asyncio.gather(*tasks)


async def main():
    consumer = KafkaConsumer(
        client_id=client_id,
        group_id=group_id,
        bootstrap_servers=kafka_server,
        key_deserializer=lambda k: k.decode(),
        value_deserializer=lambda v: v.decode(),
        max_poll_records=max_poll_records,
        auto_offset_reset='latest'
    )
    consumer.subscribe([topic_name])
    redis_client = await aioredis.from_url(redis_url)

    classification_model = Model()
    try:
        while True:
            messages = consumer.poll(1000, max_records=max_poll_records)
            if messages:
                print(len(list(*messages.values())))
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
