# -*- coding: utf-8 -*-
""" tokenizer for tweets!  might be appropriate for other social media dialects too.
general philosophy is to throw as little out as possible.
development philosophy: every time you change a rule, do a diff of this
program's output on ~100k tweets.  if you iterate through many possible rules
and only accept the ones that seeem to result in good diffs, it's a sort of
statistical learning with in-the-loop human evaluation :)
"""

__author__="brendan o'connor (anyall.org)"

import re,sys
# import emoticons
mycompile = lambda pat:  re.compile(pat,  re.UNICODE)
def regex_or(*items):
  r = '|'.join(items)
  r = '(' + r + ')'
  return r
def pos_lookahead(r):
  return '(?=' + r + ')'
def neg_lookahead(r):
  return '(?!' + r + ')'
def optional(r):
  return '(%s)?' % r

# added by daniel 30.01.2012 taken from  http://en.wikipedia.org/wiki/List_of_emoticons plus some from yahoo messenger
EmoticonsDN= '(:\)|:\(|:-\)|>:]|:o\)|:3|:c\)|:>|=]|8\)|=\)|:}|:^\)|>:D\)|:-D|:D|8-D|8D|x-D|xD|X-D|XD|=-D|=D|=-3|=3\)|8-\)|:-\)\)|:\)\)|>-\[|:-\(|:\(|:-c|:c|:-<|:<|:-\[|:\[|:{|>.>|<.<|>.<|:-\|\||D:<|D:|D8|D;|D=|DX|v.v|D-\':|>;\]|;-\)|;\)|\*-\)|\*\)|;-\]|;\]|;D|;^\)|>:P|:-P|:P|X-P|x-p|xp|XP|:-p|:p|=p|:-b|:b|>:o|>:O|:-O|:O|:0|o_O|o_0|o.O|8-0|>:\\|>:/|:-/|:-.|:/|:\\|=/|=\\|:S|:\||:-\||>:X|:-X|:X|:-#|:#|:$|O:-\)|0:-3|0:3|O:-\)|O:\)|0;^\)|>:\)|>;\)|>:-\)|:\'-\(|:\'\(|:\'-\)|:\'\)|;\)\)|;;\)|<3|8-}|>:D<|=\)\)|=\(\(|x\(|X\(|:-\*|:\*|:\">|~X\(|:-?)'
PunctChars = r'''['“".?!,:;]'''
Punct = '%s+' % PunctChars
Entity = '&(amp|lt|gt|quot);'

# one-liner URL recognition:
#Url = r'''https?://\S+'''

# more complex version:
UrlStart1 = regex_or('https?://', r'www\.')
CommonTLDs = regex_or('com','co\\.uk','org','net','info','ca')
UrlStart2 = r'[a-z0-9\.-]+?' + r'\.' + CommonTLDs + pos_lookahead(r'[/ \W\b]')
UrlBody = r'[^ \t\r\n<>]*?'  # * not + for case of:  "go to bla.com." -- don't want period
UrlExtraCrapBeforeEnd = '%s+?' % regex_or(PunctChars, Entity)
UrlEnd = regex_or( r'\.\.+', r'[<>]', r'\s', '$')
Url = (r'\b' + 
    regex_or(UrlStart1, UrlStart2) + 
    UrlBody + 
    pos_lookahead( optional(UrlExtraCrapBeforeEnd) + UrlEnd))

Url_RE = re.compile("(%s)" % Url, re.U|re.I)

Timelike = r'\d+:\d+h{0,1}' # removes the h trailing the hour like in 18:00h
Number = r'^\d+'
NumNum = r'\d+\.\d+'
NumberWithCommas = r'(\d+,)+?\d{3}' + pos_lookahead(regex_or('[^,]','$'))

Abbrevs1 = ['am','pm','us','usa','ie','eg']
def regexify_abbrev(a):
  chars = list(a)
  icase = ["[%s%s]" % (c,c.upper()) for c in chars]
  dotted = [r'%s\.' % x for x in icase]
  return "".join(dotted)
Abbrevs = [regexify_abbrev(a) for a in Abbrevs1]

BoundaryNotDot = regex_or(r'\s', '[“"?!,:;]', Entity)
aa1 = r'''([A-Za-z]\.){2,}''' + pos_lookahead(BoundaryNotDot)
aa2 = r'''([A-Za-z]\.){1,}[A-Za-z]''' + pos_lookahead(BoundaryNotDot)
ArbitraryAbbrev = regex_or(aa1,aa2)

assert '-' != '―'
Separators = regex_or('--+', '―')
Decorations = r' [  ♫   ]+ '.replace(' ','')

EmbeddedApostrophe = r"\S+'\S+"



ProtectThese = [
#    emoticons.Emoticon,
    EmoticonsDN,
    Url,
    Entity,
    Timelike,
    NumNum,
    NumberWithCommas,
    Punct,
    ArbitraryAbbrev,
    Separators,
    Decorations,
    EmbeddedApostrophe,
]
Protect_RE = mycompile(regex_or(*ProtectThese))


class Tokenization(list):
  " list of tokens, plus extra info "
  def __init__(self):
    self.alignments = []
    self.text = ""
  def subset(self, tok_inds):
    new = Tokenization()
    new += [self[i] for i in tok_inds]
    new.alignments = [self.alignments[i] for i in tok_inds]
    new.text = self.text
    return new
  def assert_consistent(t):
    assert len(t) == len(t.alignments)
    assert [t.text[t.alignments[i] : (t.alignments[i]+len(t[i]))] for i in range(len(t))] == list(t)

def align(toks, orig):
  s_i = 0
  alignments = [None]*len(toks)
  for tok_i in range(len(toks)):
    while True:
      L = len(toks[tok_i])
      if orig[s_i:(s_i+L)] == toks[tok_i]:
        alignments[tok_i] = s_i
        s_i += L
        break
      s_i += 1
      if s_i >= len(orig): raise AlignmentFailed((orig,toks,alignments))
      #if orig[s_i] != ' ': raise AlignmentFailed("nonspace advance: %s" % ((s_i,orig),))
  if any(a is None for a in alignments): raise AlignmentFailed((orig,toks,alignments))

  return alignments

class AlignmentFailed(Exception): pass

def unicodify(s, encoding='utf8', *args):
  if isinstance(s,unicode): return s
  if isinstance(s,str): return s.decode(encoding, *args)
  return unicode(s)

def tokenize(tweet):
#  print("$"+repr(tweet))
  text = unicodify(tweet)
#  print("#"+repr(text))
  text = squeeze_whitespace(text)
  t = Tokenization()
  t += simple_tokenize(text)
  t.text = text
  t.alignments = align(t, text)
#  print("|".join(t))
  return t

def simple_tokenize(text):
  s = text
  s = edge_punct_munge(s)

  # strict alternating ordering through the string.  first and last are goods.
  # good bad good bad good bad good
  goods = []
  bads = []
  i = 0
  if Protect_RE.search(s):
    for m in Protect_RE.finditer(s):
      goods.append( (i,m.start()) )
      bads.append(m.span())
      i = m.end()
    goods.append( (m.end(), len(s)) )
  else:
    goods = [ (0, len(s)) ]
  assert len(bads)+1 == len(goods)

  goods = [s[i:j] for i,j in goods]
  bads  = [s[i:j] for i,j in bads]
  #print goods
  #print bads
  goods = [unprotected_tokenize(x) for x in goods]
  res = []
  for i in range(len(bads)):
    res += goods[i]
    res.append(bads[i])
  res += goods[-1]

  res = post_process(res)
  return res

AposS = mycompile(r"(\S+)('s)$")

def post_process(pre_toks):
  # hacky: further splitting of certain tokens
  post_toks = []
  for tok in pre_toks:
    m = AposS.search(tok)
    if m:
      post_toks += m.groups()
    else:
      post_toks.append( tok )
  return post_toks

WS_RE = mycompile(r'\s+')
def squeeze_whitespace(s):
  new_string = WS_RE.sub(" ",s)
  return new_string.strip()

# fun: copy and paste outta http://en.wikipedia.org/wiki/Smart_quotes
EdgePunct      = r"""[  ' " “ ” ‘ ’ < > « » { } ( \) [ \]  ]""".replace(' ','')
#NotEdgePunct = r"""[^'"([\)\]]"""  # alignment failures?
NotEdgePunct = r"""[a-zA-Z0-9]"""
EdgePunctLeft  = r"""(\s|^)(%s+)(%s)""" % (EdgePunct, NotEdgePunct)
EdgePunctRight =   r"""(%s)(%s+)(\s|$)""" % (NotEdgePunct, EdgePunct)
EdgePunctLeft_RE = mycompile(EdgePunctLeft)
EdgePunctRight_RE= mycompile(EdgePunctRight)

def edge_punct_munge(s):
  s = EdgePunctLeft_RE.sub( r"\1\2 \3", s)
  s = EdgePunctRight_RE.sub(r"\1 \2\3", s)
  return s


def unprotected_tokenize(s):
  return s.split()

OneCharTokens = r'^.{1}$' # remove the one character tokens (maybe too agressive)
ParNumber = r'[()][+-]*\d+[()]*' # remove stuff like (+1 (-2 that appear as tokens

ExcludeThese = [
    EmoticonsDN,
    Url,
    NumNum,
    NumberWithCommas,
    Punct,
    Separators,
    Timelike,
    Number,
    OneCharTokens,
    ParNumber
]
Exclude_RE = mycompile(regex_or(*ExcludeThese))

import json
fo=open("out","w")

if __name__=='__main__':
  for line in sys.stdin:
    try:
      tweet=json.loads(line,strict=False)
      l=tweet["text"]
      t=tokenize(l)
      tweet['tokens']=t
      s=""
      keywords = ["rt", "retweet", "via", "(", ")", "-", "/via","/","//","=","%","|"]
      lang=[]
      for token in t:
        if (token[0]!='#') and (token[0]!='@') and not Exclude_RE.search(token) and token.lower() not in keywords:
      	  lang.append(token.lower())
      tweet['tok_lang']=" ".join(lang)
      print(json.dumps(tweet))
    except:
      continue
