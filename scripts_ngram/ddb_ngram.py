import json
import numpy as np
from urllib.request import urlopen
from bs4 import BeautifulSoup
import pandas as pd
import nltk
from collections import Counter
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String
import re
import sys


engines = []
meta = MetaData()
monogram = Table(
	'gram', meta,
	Column('n', Integer),
	Column('gram', String),
	Column('annee', Integer),
	Column('mois',Integer),
 )
for i in range(3):
	engines.append(create_engine(f'sqlite:///{i+1}gram_ddb.db', echo = True))
	meta.create_all(engines[i])

tokenizer = nltk.RegexpTokenizer(r"[0-9a-zà-ÿ'ß]+")
path="/shared/projects/project_gallica/presse/ddb/"

for year in np.arange(1928,1951):
	for month in np.arange(1,13):
		year2 = year
		month2 = month+1
		if month==12:
			year2 +=1
			month2=1
		if month<10:
			month = "0" + str(month)
		if month2<10:
			month2 =  "0" + str(month2)
		response = urlopen(f"https://api.deutsche-digitale-bibliothek.de/search/index/newspaper-issues/select?q=publication_date:[{year}-{month}-01T00:00:00.000Z%20TO%20{year2}-{month2}-01T00:00:00.000Z]%20AND%20language:ger&rows=1000000000")
		data = json.loads(response.read())
		docs = data["response"]["docs"]
		text = []
		for doc in docs:
			if "plainpagefulltext" in doc.keys():
				text.append(doc["plainpagefulltext"])
		text = '.'.join(text)
		#file = open(path + str(month) + "-" + str(year) + ".txt","a")
		#file.write(text)
		#file.close()
		text_split = re.split('[!"#$%&\()*+,./:;<=>?@[\\]^_`{|}~\n]',text.lower().replace("’","'"))
		ngrams = []
		for length in range(2):
			ngrams.append([])
		for sentence in text_split:
			tokens = tokenizer.tokenize(sentence)
			for length in range(2):
				ngrams[length] += list(nltk.ngrams(tokens,length+1))
		for length in range(2):
			matrix = pd.DataFrame.from_dict(Counter(ngrams[length]),orient="index")
			if len(matrix.index)>1:
				matrix.columns = ["n"]
				matrix = matrix.loc[matrix.n>(1+length)]
				matrix["gram"] = [' '.join(gram) for gram in matrix.index]
				matrix["annee"] = year
				matrix["mois"] = month
				matrix.to_sql("gram",engines[length],if_exists="append",index=False)
				print(str(month) + "-" + str(year))
