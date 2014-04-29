#!/usr/bin/env python
# coding=utf-8

import bs4
from bs4 import BeautifulSoup
from collections import defaultdict
import tldextract
from dateutil.parser import parse
from glob import glob
from os.path import *
import re
import codecs
from nltk.tokenize import sent_tokenize
from tokenizer import TreebankWordTokenizer
import sys
import os


extract = tldextract.TLDExtract(fetch=False)

if len(sys.argv) > 5 or len(sys.argv) < 2:
    print >>sys.stderr, "Usage: {} [InputFolder] <OutputPrefix> <NumFeats> <MinUserFreq>".format(sys.argv[0])
    sys.exit(1)

minUserFreq = 10
if len(sys.argv) > 4:
    minUserFreq = int(sys.argv[4])

N = 30000
if len(sys.argv) > 3:
    N = int(sys.argv[3])

prefix = ''
if len(sys.argv) > 2:
    prefix = sys.argv[2]

in_folder = sys.argv[1]


def ensure_dir(f):
    d = os.path.dirname(f)
    if d and not os.path.exists(d):
        os.makedirs(d)

if prefix:
	ensure_dir(prefix)

tokr = TreebankWordTokenizer()
word_tokenize = tokr.tokenize


stopwordlist = set()
with codecs.open("stop-en", "r", encoding='utf-8') as stopfile:
    for word in stopfile:
        stopwordlist.add(word.strip())


## tokeniser changes punctuation in unfortunate ways..
stopwordlist.add(u"``")
stopwordlist.add(u"''")
stopwordlist.add(u"link")

ipaddress = re.compile(r"[0-9]+(\.[0-9]+){3}$")

## extract valid data from some odd links in the text
def extract_valid_domain(link):
    if link.has_attr("href"):
        link = link["href"]
        ## deal with redirector
        lastredir = link.rfind('redir.asp?URL=')
        if lastredir > 0:
            link = link[lastredir + 14:]
        if link.startswith("http"):
            tldtuple = extract(link)
            ## don't want to treat ip-addresses as sources
            if not ipaddress.match(tldtuple.domain):
                ## could consider using only domain, without suffix
                return tldtuple.domain + "." + tldtuple.suffix
    return None


## source uses various utf-8 forms for quotes.
doublequote = re.compile(ur"(``|''|‘‘|’’|“|”)")
singlequote = re.compile(ur"(`|‘|’)")

number = re.compile(ur"[0-9]+")
onlypunct = re.compile(ur"^\W+$")


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


def update_counts(counts, words):
    for word in words:
        counts[word] += 1


class Record:

    def __init__(self, day):
        self.day = day
        self.items = []


class Item:

    def __init__(self, txt, users):
        self.txt = txt
        self.users = users


class Uids:

    def __init__(self):
        self.users = {"No Source":0}
        self.uid_counter = 1

    def getUid(self, username):
        if username in self.users:
            return self.users[username]
        else:
            self.users[username] = self.uid_counter
            self.uid_counter += 1
            return self.uid_counter - 1


## officeowl, officeprop, partyowl, partyprop, birthplaceowl, countryowl, countryprop, leadernameowl, birthcountry (owl)
## prop derived data is likely to be less useful than owl derived data, as it's not normalised across dbpedia
annots = ["officeowl", "partyowl", "birthplaceowl", "countryowl", "leadernameowl", "birthcountry", "officeprop", "partyprop", "countryprop"]


def find_one_ANNIE(par, annietype):
    annots = []
    for pers in par.find_all(annietype):
        #pers.stripped_strings = list(pers.stripped_strings)
        annots.append("annotated:" + "_".join(pers.stripped_strings))
    return annots

def find_ANNIE(par):
    annots = []
    annots.extend(find_one_ANNIE(par, "Person"))
    annots.extend(find_one_ANNIE(par, "Location"))
    annots.extend(find_one_ANNIE(par, "Organization"))
    return annots

def find_annotations(par):
    ## find only mentions for now, could potentially filter
    ## mentions or specialize them by types later
    annotations = []
    annotations.extend(find_ANNIE(par))
    for mention in par.find_all("Mention"):
        annotations.append("annotated:" + mention["inst"].replace("http://dbpedia.org/resource/", ""))
        for annot in annots:
            if annot in mention.attrs:
                props = mention[annot].replace("http://dbpedia.org/resource/", "").replace(" ", "_").split(",")
                props = ["annotated:" + p for p in props if p]
                annotations.extend(props)
    POStags=[]
    for tok in par.find_all("Token"):
        x = process([tok["string"]])
        if x:
           xx=x[0]+'/'+tok["category"]
           POStags.append(xx)
    annotations.extend(POStags)

    return annotations


def main():
    newsdata = []

    userfreqs = defaultdict(int)

    oldest_date = None
    ## load all relevant data in memory
    kk=0
    for fil in glob(in_folder + "/*.xml"):
        date = basename(fil)[:-8]
        print kk,fil
        kk=kk+1
        curdate = parse(date)
        if not oldest_date or oldest_date > curdate:
            oldest_date = curdate

        soup = BeautifulSoup(codecs.open(fil, "r", encoding='utf-8'), 'xml')

        ## one or two pages have embedded scripts that muck things up
        ## removing them doesn't remove any readable text on page.
        for script_block in soup.find_all("script"):
            script_block.clear()

        data = soup.find("div", "newsbody")

        curRecord = Record(curdate)
        curUsersCounted = set()

        curtext = []
        annotations=[]
        for par in data("p"):
         #   print par.prettify()
            ## paragraph contains links, end of current story
    #        print len(par)
            links = par.findAll("a")

            if links != []:
                users = set()
                for link in links:
                    domain = extract_valid_domain(link)
                    if domain:
                        users.add(domain)
                        link.clear()

                txt = list(par.stripped_strings)
                curtext.extend(txt)
                if users:
                    ## process text
                    tokens = []#process(curtext)
                    ## find mentions, properties etc
    #               # print "Paragraph ", ppar
    #               # print "P ",par
                    annotations.extend(find_annotations(par))
                    update_counts(userfreqs, [u for u in users if u not in curUsersCounted])
                    curUsersCounted.union(users)
                    curRecord.items.append(Item(tokens + annotations, list(users)))
    #                print "Tokens: ",len(tokens),tokens
    #                print "Annotations: ",len(annotations),annotations
                    #print "diff:", list(set(tokens)-set(annotations))
    #                print "Users: ",users
                    curtext = []
                    annotations=[]
                    #break
            else:
                txt = list(par.stripped_strings)
                if not txt:
                    continue
                ## paragraph ends with the text 'No Link', remove No Link text
                ## end current story
                if len(txt)==2 and txt[0]==u"No" and txt[1]==u"link":
                    curtext.extend(txt[:-1])
                    ## process text
                    tokens = []#process(curtext)
                    ## find mentions, properties etc
    #		print "Paragraph no link: ", par
                    annotations.extend(find_annotations(par))
                    update_counts(userfreqs, [u"No Source"] if u"No Source" not in curUsersCounted else [])
                    curUsersCounted.add(u"No Source")
                    curRecord.items.append(Item(tokens + annotations, [u"No Source"]))
    #                print "Tokens: ",len(tokens),tokens
    #                print "Annotations: ",len(annotations),annotations
    #		print "No user"
                    curtext = []
                    annotations = []
                    #break
                else:
                    curtext.extend(txt)
                    annotations.extend(find_annotations(par))
    #                print "Text: ",curtext
    #    print par
        newsdata.append(curRecord)

    users = set([user for (user, freq) in userfreqs.items() if freq >= minUserFreq])

    uids = Uids()

    tokenfreqs = defaultdict(int)

    ## filter out low-frequency news sources,
    ## remove news items that have no users
    filtered_newsdata = []
    for rec in newsdata:
        newItems = []
        for item in rec.items:
            newusers = []
            for user in item.users:
                if user in users:
                    newusers.append(uids.getUid(user))
            if newusers:
                newItems.append(Item(item.txt, newusers))
                update_counts(tokenfreqs, item.txt)
        newRec = Record(rec.day)
        newRec.items = newItems
        filtered_newsdata.append(newRec)

    newsdata = filtered_newsdata

    ## find top N tokens
    tokens = sorted(tokenfreqs.items(), reverse=True, key=lambda (a,b): b)
    tokens = [word for word, count in tokens if count>11]
    #tokens = sorted(tokenfreqs.items(), reverse=True, key=lambda (a,b): b)[:N]
    #tokens = [word for word, count in tokens]
    tokendict = dict(zip(tokens, range(len(tokens))))

    newsdata.sort(key=lambda a: a.day)

    def filtered_frequency(words):
        data = defaultdict(int)
        for word in words:
            if word in tokendict:
                data[tokendict[word]] += 1
        return data


    sora_vs = open(prefix + "sora_vs", "w")
    sora_vsd = open(prefix + "sora_vsd", "w")

    ## date index
    days = []

    for rec in newsdata:
        ## for date index
        daystring = rec.day.date().isoformat()
        rec.day = (rec.day - oldest_date).days

        days.append((rec.day, daystring))

        curdata = defaultdict(lambda: [])
        curcounts = defaultdict(int)
        for article in rec.items:
            for user in article.users:
                curdata[user] += article.txt
                curcounts[user] += 1
        for user in curdata:
            for word, freq in filtered_frequency(curdata[user]).items():
                print >>sora_vs, str(rec.day) + " " + str(user) + " " + str(word) + "\t" + str(freq)
            print >>sora_vsd, str(rec.day) + " " + str(user) + "\t" + str(curcounts[user])

    sora_vs.close()
    sora_vsd.close()

    with codecs.open(prefix + "dictionary", "w", encoding='utf-8') as dictfile:
        for word, index in sorted(tokendict.items(), key=lambda x: x[1]):
            print >>dictfile, index, word

    with codecs.open(prefix + "users", "w", encoding='utf-8') as userfile:
        for uname, index in sorted(uids.users.items(), key=lambda x: x[1]):
            print >>userfile, index, uname

    with codecs.open(prefix + "dates", "w", encoding='utf-8') as datefile:
        for index, date in sorted(days):
            print >>datefile, index, date


if __name__ == "__main__":
    main()
