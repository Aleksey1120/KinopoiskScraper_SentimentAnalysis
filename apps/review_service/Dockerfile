FROM python:3.11.8

EXPOSE 8080

COPY ./apps/review_service/requirements.txt ./apps/review_service/requirements.txt

RUN ["pip", "install", "-r", "apps/review_service/requirements.txt"]

COPY ./apps/review_service ./apps/review_service

ENTRYPOINT ["uvicorn", "apps.review_service.main:app", "--host", "0.0.0.0", "--port", "8080"]