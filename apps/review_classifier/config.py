MODEL_NAME = 'cointegrated/rubert-tiny2'
MODEL_DIR = 'apps/review_classifier/models'
PRE_TRAINED_MODEL = 'rubert-tiny2-finetuned-kinopoisk-sentiment-analysis.bin'
GDRIVE_URL = 'https://drive.google.com/file/d/1wIb_-Ib7Ry7B0ZZ8Ep4JrFBA6zbSzwVY/view?usp=sharing'
CLASS_NAMES = [
    'negative',
    'neutral',
    'positive'
]
MAX_LENGTH = 128
DEVICE = 'cpu'
BATCH_SIZE = 16
CLIENT_ID = 'client_1'
GROUP_ID = 'group_1'
MAX_POLL_RECORDS = 64
TTL = 60 * 15
