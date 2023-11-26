from selenium.webdriver.remote.webdriver import WebDriver

from selenium.webdriver.common.by import By
from kinopoisk_scraper import constants as const
from kinopoisk_scraper.review import Review
import time
import re
import random
import logging

logger = logging.getLogger('logger')


class KinopoiskScraper:
    def __init__(self, driver: WebDriver):
        self.driver = driver

    def get_film_ids(self):
        for i in range(const.FIRST_PAGE, const.LAST_PAGE + 1):
            page_film_ids = self._get_film_ids_from_page(i)
            logger.info(f'Got film ids from page {i}: {len(page_film_ids)}.')
            time.sleep(const.WAIT_TIME + const.RANDOM_WAIT_TIME * random.random())
            yield page_film_ids

    def _get_film_ids_from_page(self, page_idx):
        url = f'https://www.kinopoisk.ru/lists/movies/popular-films/?sort=votes&b=released&page={page_idx}'
        self.driver.get(url)
        film_ids = []
        for film_element in self.driver.find_elements(By.CSS_SELECTOR, 'a[href^="/film/"]'):
            film_href = film_element.get_attribute('href')
            film_id = re.search(r'/film/(\d*)/', film_href).group(1)
            if film_id not in film_ids:
                film_ids.append(film_id)
        return film_ids

    def get_reviews(self, film_id):
        reviews = []
        page_idx = 1
        while True:
            url = f'https://www.kinopoisk.ru/film/{film_id}/reviews/ord/date/status/all/perpage/{const.PERPAGE}/page/{page_idx}/'
            self.driver.get(url)
            for review_element in self.driver.find_elements(By.CSS_SELECTOR, 'div[class^="response"]'):
                review_type = review_element.get_attribute('class')
                review_title = review_element.find_element(
                    By.CSS_SELECTOR, 'meta[itemprop="headline"]'
                ).get_attribute('content').strip()
                review_text = review_element.find_element(
                    By.CSS_SELECTOR, '[itemprop="reviewBody"]'
                ).text.strip()
                reviews.append(
                    Review(review_type, review_title, review_text)
                )
            reviews_from_page = len(reviews) - (page_idx - 1) * const.PERPAGE
            if reviews_from_page == 0:
                logger.warning(
                    f'Got film {film_id} reviews from page {page_idx}: {reviews_from_page}.')
            else:
                logger.info(
                    f'Got film {film_id} reviews from page {page_idx}: {reviews_from_page}.')
            time.sleep(const.WAIT_TIME + const.RANDOM_WAIT_TIME * random.random())
            if reviews_from_page < const.PERPAGE:
                logger.info(f'Completed scraping reviews for film {film_id}.')
                break
            page_idx += 1
        return reviews
