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
import requests
import urllib3.util
import requests.adapters
from bs4 import BeautifulSoup


s = requests.Session()
retries = urllib3.util.Retry(connect=10, read=10, redirect=10, status=10, other=10)
s.mount('https://', requests.adapters.HTTPAdapter(max_retries=retries))


#path="/opt/bazoulay/persee/"
#path_cairn="/opt/bazoulay/"
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
for i in range(2):
    engines.append(create_engine(f'sqlite:///{i+1}gram_libe.db', echo = True))
    meta.create_all(engines[i])


end_of_month = [31,28,31,30,31,30,31,31,30,31,30,31]
tokenizer = nltk.RegexpTokenizer(r"[a-zà-ÿ']+|[0-9]{4}")

for year in np.arange(2020,2024):
    for month in range(12):
        n_days = end_of_month[month]
        if year<2021 or (year==2021 and month < 6):continue
        month = month+1
        if month<10:month = "0" + str(month)
        for day in range(n_days):
            day = day+1
            if year==2021 and month=="07" and day<17:continue
            if day<10:day = "0" + str(day)
            page = s.get(f"https://www.liberation.fr/archives/{year}/{month}/{day}/")
            while page.status_code != 200:
                sleep(15)
                page = s.get(f"https://www.liberation.fr/archives/{year}/{month}/{day}/")
            print(page.url)
            soup = BeautifulSoup(page.content)
            articles = soup.find_all("article")
            articles = [article.find("a")["href"] for article in articles]
            text = ""
            for article in articles:
                page = s.get("https://www.liberation.fr" + article)
                while page.status_code != 200:
                    page = s.get("https://www.liberation.fr" + article)
                soup = BeautifulSoup(page.content)
                para = soup.find_all("p")
                text+='\n'+soup.find("h1").text
                text+='\n'.join(t.text for t in para)
                print(article)
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
                    matrix["mois"] = month
                    matrix["jour"] = day
                    matrix.to_sql("gram",engines[length],if_exists="append",index=False)


