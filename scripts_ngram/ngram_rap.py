from tqdm import tqdm
import numpy as np
from tqdm import tqdm
import os
import time
import pandas as pd
import nltk
from collections import Counter
import re
import sys
from collections import Counter
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String
import re
import sys
corpus = pd.read_csv("~/LRFAF/corpus.csv")
engines = []
meta = MetaData()
monogram = Table(
	'gram', meta,
	Column('n', Integer),
	Column('gram', String),
	Column('annee', Integer)
 )
for i in range(5):
	engines.append(create_engine(f'sqlite:///{i+1}gram_rap.db', echo = True))
	meta.create_all(engines[i])

tokenizer = nltk.RegexpTokenizer(r"[a-zà-ÿ0-9']+")

for year in np.arange(1989,2025):
	text = '\n'.join(corpus.lyrics.values[corpus.year==year])
	text = re.sub("(?<=[A-Z])\.","",text)  #Garder les Monsieur
	text_split = re.split('[!"#$%&\()*+,./:;<=>?@[\\]^_`{|}~\n]',text.lower().replace("’","'"))
	ngrams = []
	for length in range(5):
		ngrams.append([])
	for sentence in text_split:
		tokens = tokenizer.tokenize(sentence)
		for length in range(5):
			ngrams[length] += list(nltk.ngrams(tokens,length+1))
	for length in range(5):
		matrix = pd.DataFrame.from_dict(Counter(ngrams[length]),orient="index")
		if len(matrix.index)>1:
			matrix.columns = ["n"]
			matrix["gram"] = [' '.join(gram) for gram in matrix.index]
			matrix["annee"] = year
			matrix.to_sql("gram",engines[length],if_exists="append",index=False)
	print(year)


