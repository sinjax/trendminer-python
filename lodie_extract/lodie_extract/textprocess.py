# coding=utf-8
import re
import codecs
doublequote = re.compile(ur"(``|''|‘‘|’’|“|”)")
singlequote = re.compile(ur"(`|‘|’)")
number = re.compile(ur"[0-9]+")
onlypunct = re.compile(ur"^\W+$")
stopwordlist = set()
with codecs.open("data/stop-en", "r", encoding='utf-8') as stopfile:
    for word in stopfile:
        stopwordlist.add(word.strip())


## tokeniser changes punctuation in unfortunate ways..
stopwordlist.add(u"``")
stopwordlist.add(u"''")
stopwordlist.add(u"link")
def normalise(text):
    outtxt = []
    for txt in text:
        txt = txt.lower()
        txt = singlequote.sub(u"'", doublequote.sub(u'"', txt))
        txt = txt.replace(u'…', u'...')
        txt = number.sub(u'<num>', txt)
        outtxt.append(txt)
    return outtxt

def process(text):
#    print
#    print text.encode('utf-8')
    tmp = normalise(text)
    # print
    # print text.encode('utf-8')

    bigrams = [a + "__" + b for (a,b) in zip(tmp[:-1], tmp[1:])]
    unigrams = [x for x in tmp if len(x) > 1 if x not in stopwordlist if not onlypunct.match(x)]

    # print
    # print tmp
    # print
    return unigrams + bigrams