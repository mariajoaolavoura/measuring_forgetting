import pandas as pd
import gzip
import json

def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield json.loads(l)

def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')


def load_data(file_namepath:str, head:int = 500):
  '''
    loads *head* number of lines from the json.gz file 
    if head is None reads all
    returns: list of dictionaries
    code from https://github.com/MengtingWan/goodreads/blob/master/samples.ipynb
  '''
  count = 0
  data = []
  with gzip.open(file_namepath) as fin:
      for l in fin:
          d = json.loads(l)
          count += 1
          data.append(d)
          
          # break if reaches the 100th line
          if (head is not None) and (count > head):
              break
  return data


def getDF_n_lines(file_path:str, head):
  i = 0
  df = {}
  with gzip.open(file_path, 'rb') as fin:
      for l in fin:
          d = json.loads(l)
          df[i] = d
          i += 1
          # break if reaches the 100th line
          if (head is not None) and (i > head):
              break
  return pd.DataFrame.from_dict(df, orient='index')