import telebot
import requests
import os
import json

bot = telebot.TeleBot(os.environ['TOKEN'])

CLASSIFIER_URL = os.environ['CLASSIFIER_URL']


def get_movie_sentiment(text):
    response = requests.post(CLASSIFIER_URL, json.dumps({'texts': [text]})).json()
    predicted_sentiment = response['sentiments'][0]
    confidence = response['confidences'][0]
    return predicted_sentiment, confidence


@bot.message_handler(content_types=['text'])
def get_text_messages(message):
    if message.text == '/help' or message.text == '/start':
        bot.send_message(message.from_user.id, 'Write me a movie review. I\'ll try to determine its sentiment.')
    else:
        predicted_sentiment, confidence = get_movie_sentiment(message.text)
        bot.send_message(message.from_user.id,
                         f"I think the review is {predicted_sentiment} with {confidence:.2%} confidence.")


bot.polling(none_stop=True)
