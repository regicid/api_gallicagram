import nltk
from collections import Counter
import re
import sys
from datetime import datetime, timedelta
from collections import Counter
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String
import re
import sys
corpus = pd.read_csv("/data/corpus/liberation_metadata.csv")
corpus["year"] = pd.to_datetime(corpus.date_only).dt.year
corpus["month"] = pd.to_datetime(corpus.date_only).dt.month
corpus["day"] = pd.to_datetime(corpus.date_only).dt.day
engines = []
meta = MetaData()
monogram = Table(
	'gram', meta,
	Column('n', Integer),
	Column('gram', String),
	Column('annee', Integer),
	Column('mois', Integer),
	Column('jour', Integer)
 )
for i in range(5):
	engines.append(create_engine(f'sqlite:///{i+1}gram_libe.db', echo = True))
	meta.create_all(engines[i])

tokenizer = nltk.RegexpTokenizer(r"[a-zà-ÿ0-9']+")



# Iterate day by day from 1998-01-01 to 2026-12-31
start_date = datetime(1998, 1, 1)
end_date = datetime(2026, 12, 31)
current_date = start_date

while current_date <= end_date:
    year = current_date.year
    month = current_date.month
    day = current_date.day
    
    # Filter corpus for this specific day
    daily_corpus = corpus.loc[
        (corpus.year == year) & 
        (corpus.month == month) & 
        (corpus.day == day)
    ]
	text = []
	for url in daily_corpus.url:
		filename = /data/corpus/url.replace("/","_") + ".txt"
		filename = "/data/corpus/liberation/" + url.replace("/","_") + ".txt"
		f = open(filename,"r")
		text = text.append(f.read())
		f.close()
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
			matrix["mois"] = month
			matrix["jour"] = day
			matrix.to_sql("gram",engines[length],if_exists="append",index=False)
    # Move to next day
    current_date += timedelta(days=1)



