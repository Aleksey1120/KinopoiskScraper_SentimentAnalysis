import asyncio
import time
import sys

sys.path.append('apps/review_classifier')

import aiohttp
import numpy as np
import pandas as pd

from apps.review_classifier.model import config

num_concurrent_requests = 10
n_iter = 10
url = 'http://localhost:8080/predict'

data_path = './data/interim/test_reviews.csv'
dataset = pd.read_csv(data_path).values


async def send_requests(n_iter, session, data):
    successful_requests = 0
    correct_predictions = 0
    for i in range(n_iter):
        try:
            timeout = aiohttp.ClientTimeout(total=15)
            async with session.post(url, json={'text': data[i, 0]},
                                    timeout=timeout) as response:
                if response.status == 200:
                    successful_requests += 1
                if config['CLASS_NAMES'][data[i, 1]] == (await response.json())['sentiment']:
                    correct_predictions += 1
        except aiohttp.client_exceptions.ClientConnectorError:
            print(data[i])
        except asyncio.TimeoutError:
            print(data[i])
    return successful_requests, correct_predictions


async def main():
    t_start = time.time()
    async with aiohttp.ClientSession() as session:
        tasks = []
        for _ in range(num_concurrent_requests):
            indices = np.random.choice(np.arange(0, dataset.shape[0]), size=n_iter)
            task = asyncio.create_task(send_requests(n_iter, session, dataset[indices, :]))
            tasks.append(task)
        results = await asyncio.gather(*tasks)
        successful_requests, correct_predictions = np.sum(results, axis=0)
    t_end = time.time()
    print(f'Total requests: {num_concurrent_requests * n_iter}.\n'
          f'Total time: {t_end - t_start:.2f}.\n'
          f'RPS {num_concurrent_requests * n_iter / (t_end - t_start):.2f}.\n'
          f'Successful: {successful_requests}({successful_requests / (num_concurrent_requests * n_iter):.1%}).\n'
          f'Correct: {correct_predictions}({correct_predictions / (num_concurrent_requests * n_iter):.1%}).')


if __name__ == '__main__':
    asyncio.run(main())
