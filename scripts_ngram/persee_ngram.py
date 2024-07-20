import pandas as pd
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

data = pd.read_csv("~/files_persee.csv")
#data.loc[data.date==0,"date"] = [1986,1992,1992,1992,1992,1992,1993,1993,1993,1993,1993,1993,1997,1997]
data["identifier"] = [i.split("/")[1] for i in data["0"]]
data["revue"] = [i.split("_")[0] for i in data.identifier]

path="/opt/bazoulay/persee/"
path_cairn="/opt/bazoulay/"
engines = []
meta = MetaData()
monogram = Table(
	'gram', meta,
	Column('n', Integer),
	Column('gram', String),
	Column('annee', Integer),
	Column('revue',Integer)
 )
for i in range(2):
	engines.append(create_engine(f'sqlite:///{i+1}gram_persee.db', echo = True))
	meta.create_all(engines[i])


##Run the lines to get all the journal codes from cairn and persee

a = glob("cairn/*.txt")
years = [i.split("_")[-1] for i in a]
years = np.array([int(i.split(".")[0]) for i in years])
revues = [i.split("_")[0] for i in a]
revues = np.array([i.split("/")[-1] for i in revues])
collections_cairn_code_cairn.extend(['DRS1', 'RIED', 'VING','FLUX1'])
collections_cairn_code_persee.extend(['dreso','tiers','xxs','flux'])
for i in np.arange(0,len(collections_cairn_code_persee)):
	z = revues==collections_cairn_code_cairn[i]
	revues[z] = collections_cairn_code_persee[i]




tokenizer = nltk.RegexpTokenizer(r"[a-zà-ÿ']+|[0-9]{4}")
for revue in sorted(data.revue.unique()):
	print(revue)
	for year in sorted(data.date.unique()):
		z = np.logical_and(data.date==year,data.revue==revue)
		files = data.loc[z,"identifier"]
		z = np.logical_and(revues==revue,years==year)
		files_cairn = np.array(a)[z]
		text = ""
		for filename in files:
			file = open(path+filename+".txt","r",encoding="utf-8")
			text += file.read()
		for filename in files_cairn:
			file = open(path_cairn+filename,"r",encoding="utf-8")
			text += file.read()
		text = re.sub("(?<=[A-Z])\.","",text)  #Garder les Monsieur
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
				matrix["gram"] = [' '.join(gram) for gram in matrix.index]
				matrix["annee"] = year
				matrix["revue"] = revue
				matrix.to_sql("gram",engines[length],if_exists="append",index=False)
		print(year)


