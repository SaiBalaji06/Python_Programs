from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer
import yaml
chatbot=ChatBot
trainer = ChatterBotCorpusTrainer(chatbot)
trainer.train("chatterbot.corpus.english.greetings")
