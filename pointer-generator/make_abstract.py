import struct
import sys
import run_summarization
import nltk.data
import json

import tensorflow as tf
from tensorflow.core.example import example_pb2

DECODE_PATH ="../news-summarizer-ai/pointer-generator/log/myexperiment/decode/000000_decoded.txt"


dm_single_close_quote = u'\u2019' # unicode
dm_double_close_quote = u'\u201d'
END_TOKENS = ['.', '!', '?', '...', "'", "`", '"', dm_single_close_quote, dm_double_close_quote, ")"] # acceptable ways to end a sentence

# We use these to separate the summary sentences in the .bin datafiles
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

def get_art_abs(text):
  text = text.strip()

  # Lowercase everything
  text = text.lower()


  text = fix_missing_period(text)

  # Separate out article and abstract sentences
  article_lines = []
  highlights = []
  next_is_highlight = False
  # for idx,line in enumerate(lines):
  #   if line == "":
  #     continue # empty line
  #   elif line.startswith("@highlight"):
  #     next_is_highlight = True
  #   elif next_is_highlight:
  #     highlights.append(line)
  #   else:
  #     article_lines.append(line)

  # Make article into a single string
  # article = ' '.join(article_lines)

  # Make abstract into a signle string, putting <s> and </s> tags around the sentences
  abstract = ' '.join(["%s %s %s" % (SENTENCE_START, sent, SENTENCE_END) for sent in highlights])

  return text, abstract

def fix_missing_period(line):
  """Adds a period to a line that is missing a period"""
  if "@highlight" in line: return line
  if line=="": return line
  if line[-1] in END_TOKENS: return line
  # print line[-1]
  return line + " ."

def read_text_file(text_file):
  lines = []
  with open(text_file, "r", encoding='utf-8') as f:
    for line in f:
      lines.append(line.strip())
  return lines

def _text_to_binary(text, out_file):
  with open(out_file, 'wb') as writer:
    article, abstract = get_art_abs(text)

    # Write to tf.Example
    tf_example = example_pb2.Example()
    tf_example.features.feature['article'].bytes_list.value.extend([article.encode()])
    tf_example.features.feature['abstract'].bytes_list.value.extend([abstract.encode()])
    tf_example_str = tf_example.SerializeToString()
    str_len = len(tf_example_str)
    writer.write(struct.pack('q', str_len))
    writer.write(struct.pack('%ds' % str_len, tf_example_str))


def main(text):
  _text_to_binary(text, '../news-summarizer-ai/pointer-generator/data/val')
  run_summarization.main('work')
  with open(DECODE_PATH, 'r', encoding="utf-8") as f:
    abstract = f.read()
    sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = sent_tokenizer.tokenize(abstract) # this text should be summary
    sentences = [sent.capitalize() for sent in sentences]
    result = ' '.join(sentences)
    print("DIMAAAAA")
    print(json.dumps(result))


if __name__ == '__main__':
  main(sys.argv[1])
