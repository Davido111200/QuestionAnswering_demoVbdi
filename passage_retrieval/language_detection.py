from langdetect import detect

class Detect:
  def __init__(self):
    pass

  def det(self, text):
    return 0 if detect(text) != 'en' else 1

