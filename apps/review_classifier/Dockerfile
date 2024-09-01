FROM python:3.11.8

EXPOSE 8080

COPY ./apps/review_classifier/requirements-torch.txt ./apps/review_classifier/requirements-torch.txt

RUN ["pip", "install", "-r", "apps/review_classifier/requirements-torch.txt"]

COPY ./apps/review_classifier/requirements.txt ./apps/review_classifier/requirements.txt

RUN ["pip", "install", "-r", "apps/review_classifier/requirements.txt"]

COPY ./apps/review_classifier ./apps/review_classifier

ENTRYPOINT ["python", "-m", "apps.review_classifier.main"]