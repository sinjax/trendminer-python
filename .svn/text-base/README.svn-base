to run on a file:
cat input | python twokenize.py | python langid.py | python stemming.py > output

ad you will get the same tweets with some extra fields in the json:
tokens - list of tokens
tok_lang - string with proper words separated by whitespace
lang_det - the detected language of the tweet
stemming - list of stems

Works at a rate of about 1 million tweets/hour , although it's likely to be actually faster
