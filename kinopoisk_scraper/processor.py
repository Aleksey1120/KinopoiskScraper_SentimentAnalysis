from kinopoisk_scraper.review import Review

_review_text_to_num = {
    'response good': 1,
    'response neutral': 0,
    'response bad': -1,
}


class ReviewProcessor:

    @staticmethod
    def process_review(review: Review) -> Review:
        new_review = Review(
            _review_text_to_num[review.review_type],
            review.review_title.replace('\xa0', ' '),
            review.review_text.replace('\xa0', ' ')
        )
        return new_review

    @staticmethod
    def process_reviews(reviews: list[Review]) -> list[Review]:
        new_reviews = []
        for review in reviews:
            new_reviews.append(ReviewProcessor.process_review(review))
        return new_reviews
