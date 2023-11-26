class Review:

    def __init__(self, review_type, review_title, review_text):
        self.review_type = review_type
        self.review_title = review_title
        self.review_text = review_text

    def to_dict(self):
        return {
            'review_type': self.review_type,
            'review_title': self.review_title,
            'review_text': self.review_text,
        }
