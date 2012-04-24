import sys
import json
from nltk import SnowballStemmer

if __name__=='__main__':
  for line in sys.stdin:
    try:
      tweet=json.loads(line,strict=False)
    except:
      continue  
    lang=tweet['lang_det']
    tweet['stemmed']=[]
    options={'de': "german", 'en': "english", 'ro': "romanian", "da": "danish", "nl": "dutch", "fi": "finnish", "fr": "french", "hu": "hungarian", "it": "italian", "no": "norwegian", "pt": "portuguese", "ru": "russian", "es": "spanish", "sv": "swedish"}
    try:
      stemmer=SnowballStemmer(options[lang])
    except: 
      print json.dumps(tweet)
      continue
    tokens=tweet['tokens']
    for token in tokens:
      tweet['stemmed'].append(stemmer.stem(token))
    print json.dumps(tweet)

