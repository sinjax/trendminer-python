import json
import sys
import re

regnrt=re.compile(r"\(*RT[\s!.-:]*@\w+([\)\s:]|$)")
regrt=re.compile(r"^RT[\s!.-:]+")
reguser=re.compile(r"@\w+")
regbr=re.compile(r"\[.*\]")
regv1=re.compile(r"\(via @\w+\)")
regv2=re.compile(r" via @\w+")

def rttext(text):
  rt=''
  com=''
  c=regnrt.search(text)
  if c:
    rt=text[c.span()[1]:].strip().strip(':').strip()
    com=text[:c.span()[0]].strip().strip(':').strip()
    if c.span()[1]==len(text):
      aux=com
      com=rt
      rt=aux
  else:
    d=regrt.search(text)
    e=reguser.search(text)
    if d and e:
      com=text[d.span()[1]:e.span()[0]]
      rt=text[e.span()[1]:]
  a=regv1.search(text)
  if not a:
    a=regv2.search(text)
  if a:
    if a.span()[0]==0:
      b=regbr.search(text)
      rt=re.sub('^:','',text[a.span()[1]:b.span()[0]].strip()).strip()
      com=b.group()[1:len(b.group())-1]
    else:
      rt=re.sub('[|,.//]$','',text[:a.span()[0]].strip()).strip()
      com=re.sub('^:','',text[a.span()[1]:].strip()).strip()
  return rt,com

bf=[]
for line in sys.stdin:
  try:
    tweet=json.loads(line)
    text=tweet['text']
    texto=text
    textrt=''
    try:
      textrt=tweet['retweeted_status']['text']
    except:
      textrt=rttext(text)[0]
    if not textrt=='':
      text=textrt
      continue
    utok=tweet['analysis']['tokens']['unprotected']
    text=' '.join(utok).lower()
    if len(utok)>=6:
      if 'YouTube' in utok:
        text=' '.join(utok[0:5])
      else:
        text=' '.join(utok[0:6])
    if text in bf:
      pass
    else:
      print json.dumps(tweet)
      bf.append(text)
  except:
    continue
