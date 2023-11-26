import json
import os.path
import logging

logger = logging.getLogger('logger')


class Film:

    def __init__(self, film_id, reviews):
        self.film_id = film_id
        self.reviews = reviews

    def save_reviews(self, path):
        json.dump(
            [review.to_dict() for review in self.reviews],
            open(os.path.join(path, f'{self.film_id}.json'), 'w', encoding='utf-8'),
            ensure_ascii=False
        )
        logger.info(f'Completed saving film {self.film_id}.')
