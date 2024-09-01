import glob
import os.path
import pickle
import time
import logging

from selenium import webdriver
from fake_useragent import UserAgent

from kinopoisk_scraper.kinopoisk_scraper import KinopoiskScraper
from kinopoisk_scraper.film import Film
from kinopoisk_scraper.processor import ReviewProcessor
from kinopoisk_scraper import constants as const


logger = logging.getLogger('logger')
logger.setLevel(logging.INFO)

current_dir = os.path.dirname(os.path.realpath(__file__))

handler = logging.FileHandler(os.path.join(current_dir, 'logs.log'), mode='a')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

logger.addHandler(handler)
handler.setFormatter(formatter)


def get_driver():
    ua = UserAgent()
    user_agent = ua.random
    logger.info(f'Set user agent = {user_agent}.')
    opts = webdriver.ChromeOptions()
    opts.add_experimental_option("detach", True)
    opts.add_argument(f'user-agent={user_agent}')
    prefs = {"profile.default_content_setting_values.geolocation": 2}
    opts.add_experimental_option("prefs", prefs)
    driver = webdriver.Chrome(options=opts)
    driver.implicitly_wait(15)

    driver.get(f'https://www.kinopoisk.ru')
    for cookie in pickle.load(open(os.path.join(current_dir, 'cookies'), 'rb')):
        driver.add_cookie(cookie)
    time.sleep(5)
    return driver


def main():
    logger.info('Starting the scraper script.')
    driver = get_driver()
    scraper = KinopoiskScraper(driver)

    if not os.path.exists(const.DATASET_PATH):
        os.mkdir(const.DATASET_PATH)

    if const.IGNORE_ALREADY_SCRAPED:
        scraped_film_ids = glob.glob('*.json', root_dir=const.DATASET_PATH)
        scraped_film_ids = [scraped_film_id.split('.')[0] for scraped_film_id in scraped_film_ids]
    else:
        scraped_film_ids = []

    try:
        for film_ids in scraper.get_film_ids():
            for film_id in film_ids:
                if film_id not in scraped_film_ids:
                    reviews = scraper.get_reviews(film_id)
                    if reviews:
                        reviews = ReviewProcessor.process_reviews(reviews)
                        logger.info(f'Complete processing film {film_id} reviews.')
                        film = Film(film_id, reviews)
                        film.save_reviews(const.DATASET_PATH)
        logger.info('Scraper script completed successfully.')
    except Exception as e:
        logger.error(str(e))
        raise
    finally:
        driver.quit()


if __name__ == '__main__':
    main()

# driver = get_driver()
# time.sleep(120)
# pickle.dump(driver.get_cookies(), open('cookies', 'wb'))
