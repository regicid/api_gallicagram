import io
from io import StringIO
import matplotlib.pyplot as plt
import base64
from flask import Flask,render_template,g,request,make_response, send_file
import sqlite3
import pandas as pd
import numpy as np
import re
from waitress import serve
from functools import lru_cache
app = Flask(__name__)

def get_db(corpus,n):
    if corpus=="livres":
        DATABASE = f"/opt/bazoulay/ngram/{n}gram.db"
    elif corpus=="presse":
        DATABASE = f"/opt/bazoulay/ngram/{n}gram_presse.db"
    elif corpus == "lemonde":
        DATABASE = f"/opt/bazoulay/ngram/{n}gram_lemonde.db"
    elif corpus == "huma":
        DATABASE = f"/opt/bazoulay/ngram/{n}gram_huma.db"
    elif corpus == "figaro":
        DATABASE = f"/opt/bazoulay/ngram/{n}gram_figaro.db"
    elif corpus == "moniteur":
        DATABASE = f"/opt/bazoulay/ngram/{n}gram_moniteur.db"
    elif corpus == "paris":
        DATABASE = f"/opt/bazoulay/ngram/{n}gram_paris.db"
    elif corpus == "temps":
        DATABASE = f"/opt/bazoulay/ngram/{n}gram_temps.db"
    elif corpus == "petit_journal":
        DATABASE = f"/opt/bazoulay/ngram/{n}gram_petit_journal.db"
    elif corpus == "journal_des_debats":
        DATABASE = f"/opt/bazoulay/ngram/{n}gram_journal_des_debats.db"
    elif corpus == "ddb":
        DATABASE = f"/opt/bazoulay/ngram/{n}gram_ddb.db"
    else:
        DATABASE = f"/opt/bazoulay/ngram/{n}gram_{corpus}.db"
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE,detect_types=sqlite3.PARSE_COLNAMES)
    db.row_factory = sqlite3.Row
    return db
def get_base(corpus,n):
    if corpus == "lemonde":
        base = pd.read_csv(f"/opt/bazoulay/docker_gallicagram/gallicagram/lemonde{n}.csv")
        base.columns = ['total','annee', 'mois', 'jour']
    if corpus == "ddb":
        base = pd.read_csv(f"/opt/bazoulay/docker_gallicagram/gallicagram/ddb{n}.csv")
        base.columns = ['total','annee', 'mois']
    elif corpus =="huma":
        base = pd.read_csv(f"/opt/bazoulay/docker_gallicagram/gallicagram/humanite{n}.csv")
        base.columns = ['total','annee', 'mois', 'jour']
    elif corpus =="moniteur":
        base = pd.read_csv(f"/opt/bazoulay/docker_gallicagram/gallicagram/moniteur{n}.csv")
        base.columns = ['total','annee', 'mois', 'jour']
    elif corpus =="figaro":
        base = pd.read_csv(f"/opt/bazoulay/docker_gallicagram/gallicagram/figaro{n}.csv")
        base.columns = ['total','annee', 'mois', 'jour']
    elif corpus =="paris":
        base = pd.read_csv(f"/opt/bazoulay/docker_gallicagram/gallicagram/paris{n}.csv")
        base.columns = ['total','annee', 'mois', 'jour']
    elif corpus =="temps":
        base = pd.read_csv(f"/opt/bazoulay/docker_gallicagram/gallicagram/temps{n}.csv")
        base.columns = ['total','annee', 'mois', 'jour']
    elif corpus =="petit_journal":
        base = pd.read_csv(f"/opt/bazoulay/docker_gallicagram/gallicagram/petit_journal{n}.csv")
        base.columns = ['total','annee', 'mois', 'jour']
    elif corpus =="journal_des_debats":
        base = pd.read_csv(f"/opt/bazoulay/docker_gallicagram/gallicagram/journal_des_debats{n}.csv")
        base.columns = ['total','annee', 'mois', 'jour']
    elif corpus=="presse":
        file = "/opt/bazoulay/docker_gallicagram/gallicagram/base_presse_mois_gallica_monogrammes.csv"
        base = pd.read_csv(file)
        base[["annee","mois"]] = base.date.str.split("/",expand=True)
        base.drop("date",axis=1,inplace=True)
        base = base.astype("int64")
        base.columns = ['total','annee', 'mois']
    elif corpus=="livres":
        base = pd.read_csv("/opt/bazoulay/docker_gallicagram/gallicagram/base_livres_gallica_monogrammes.csv")
        base.columns = ["annee","total"]
    else:
        base = pd.read_csv(f"/opt/bazoulay/docker_gallicagram/gallicagram/{corpus}{n}.csv")
        base = base.rename(columns={"n":"total"})
    return base

@app.route("/query")
def query():
    args = request.args
    word = args.get("mot")
    try:
        corpus=args["corpus"]
    except:
        corpus="presse"
    try:
        fr = args["from"]
    except:
        fr=1789
    try:
        to = args["to"]
    except:
        to = 2022
    try:
        resolution = args["resolution"]
    except:
        resolution = "default"
    try:
        rubrique = args["rubrique"]
    except:
        rubrique=None
    n = len(word.split(" "))
    conn = get_db(corpus,n)
    corpus_journaliers = ["lemonde","huma","paris","figaro","moniteur","temps","petit_journal","constitutionnel","journal_des_debats","la_presse","petit_parisien"]
    if resolution=="default" or resolution=="jour" or corpus=="livres" or (resolution=="mois" and corpus in ["presse","ddb"]):
        query = f"SELECT * FROM gram where gram=\"{word}\" and annee between {fr} and {to}"
    else:
        if resolution=="annee":
            base = "gram"
            if corpus in corpus_journaliers:
                base="gram_mois"
            query = f"SELECT sum(n) as n,annee,gram from {base} where gram=\"{word}\" and annee between {fr} and {to} group by annee"
            print('aaa')
        if resolution=="mois" and corpus in corpus_journaliers:
            query = f"SELECT * FROM gram_mois where gram=\"{word}\" and annee between {fr} and {to}"
    if rubrique is not None:
        by_rubrique = args["by_rubrique"]
        query = query.replace("and annee between",f'and rubrique in "{tuple(rubrique.split(' '))}" and annee between')
    db_df = pd.read_sql_query(query,conn)
    conn.close()
    base = get_base(corpus,n)
    base = base.loc[(base.annee>=int(fr))&(base.annee<=int(to))]
    if rubrique is not None and :
        base = base.loc[base.rubrique == rubrique]
    if resolution=="mois" and corpus in corpus_journaliers + ["presse"]:base = base.groupby(["annee","mois"]).agg({'total':'sum'}).reset_index()
    if resolution=="annee" and corpus in corpus_journaliers + ["presse","lemonde_rubriques"]:base = base.groupby(["annee"]).agg({'total':'sum'}).reset_index()
    db_df = pd.merge(db_df,base,how="right")
    db_df.n = db_df.n.fillna(0)
    db_df["gram"] = word
    if corpus=="livres":db_df = db_df.sort_values("annee")
    print(word)
    return db_df.to_csv(index=False) 

@app.route('/greet')
def say_hello():
  return 'Hello from Server'
if __name__ == "__main__":
    app.run(debug=True)

@app.route('/contain')
def contain():
    args = request.args
    try:
        corpus=args["corpus"]
    except:
        corpus="lemonde"
    print(corpus)
    mot1 = args.get("mot1").replace("'","")
    mot2 = args.get("mot2").replace("'","")
    base = "gram"
    try:
        fr = args["from"]
    except:
        fr=1789
    try:
        to = args["to"]
    except:
        to=2022
    try:
        count = args["count"]
    except:
        count = "True"
    try:
        resolution = args["resolution"]
    except:
        resolution = "default"

    if corpus=="presse":
        n=3
        time_steps = "annee,mois"
    if corpus=="livres":
        n=3
        time_steps = "annee"
    if corpus=="lemonde":
        n=4
        time_steps = "annee,mois"
        base = "gram_mois"
    conn = get_db(corpus,n)
    if count=="True":
        db_df = pd.read_sql_query(f"SELECT sum(n) as n,{time_steps} FROM {base} where rowid in (select rowid from full_text where gram match '{mot1} AND {mot2}') and annee between {fr} and {to} group by {time_steps}",conn)
    else:
        db_df = pd.read_sql_query(f"SELECT sum(n) as n,gram,{time_steps} FROM {base} where rowid in (select rowid from full_text where gram match '{mot1} AND {mot2}') and annee between {fr} and {to} group by gram,{time_steps}",conn)
    conn.close()
    base = get_base(corpus,n)
    if corpus=="lemonde":base = base.groupby(["annee","mois"]).agg({'total':'sum'}).reset_index()
    base = base.loc[(base.annee>=int(fr))&(base.annee<=int(to))]
    db_df = pd.merge(db_df,base,how="right")
    db_df.n = db_df.n.fillna(0)
    corpus_journaliers = ["lemonde","huma","paris","figaro","moniteur","temps","petit_journal","constitutionnel","journal_des_debats","la_presse","petit_parisien"]
    if resolution=="mois" and corpus in corpus_journaliers + ["presse","ddb"]:db_df = db_df.groupby(["annee","mois"]).agg({'n':'sum','total':'sum'}).reset_index()
    if resolution=="annee": db_df = db_df.groupby(["annee"]).agg({'total':'sum','n':'sum'}).reset_index()
    if count=="True":db_df["gram"] = mot1 + "&" + mot2
    print(mot1 + "&" + mot2)
    return db_df.to_csv(index=False)

@app.route("/joker")
def joker():
    args = request.args
    try:
        corpus=args["corpus"]
    except:
        corpus="lemonde"
    try:
        fr = args["from"]
    except:
        fr=1789
    try:
        to = args["to"]
    except:
        to=2022
    try:
        after = args["after"]
    except:
        after = "True" 
    try:
        n_joker = str(args["n_joker"])
    except:
        n_joker = 50
    mot = args.get("mot")
    try:
        n=args["length"]
    except:
        n=len(mot.split(" "))+1
    base = "gram"
    if corpus=="lemonde":base = "gram_mois"
    print(corpus)
    conn = get_db(corpus,n)
    if n_joker=="all":
        limit=""
    else:limit = f"limit {n_joker}"
    if after=="True":
        db_df = pd.read_sql_query(f'select sum(n) as tot, gram from {base} where annee between {fr} and {to} and rowid in (select rowid from full_text where gram match \'^{mot}\') group by gram order by tot desc {limit}',conn)
    else:
        db_df = pd.read_sql_query(f'select sum(n) as tot, gram from {base} where annee between {fr} and {to} and rowid in (select rowid from full_text where gram match "{mot}") group by gram order by tot desc {limit}',conn)
    conn.close()
    print(mot)
    return db_df.to_csv(index=False)

@app.route("/joker_mois")
def joker_mois():
    args = request.args
    try:
        corpus=args["corpus"]
    except:
        corpus="lemonde"
    try:
        year = args["year"]
    except:
        year=1945
    try:
        month = args["month"]
    except:
        month=1
    try:
        after = args["after"]
    except:
        after = "True"
    try:
        n_joker = str(args["n_joker"])
    except:
        n_joker = 50
    mot = args.get("mot")
    try:
        n=args["length"]
    except:
        n=len(mot.split(" "))+1
    base = "gram"
    if corpus=="lemonde":base = "gram_mois"
    print(corpus)
    conn = get_db(corpus,n)
    if n_joker=="all":
        limit=""
    else:limit = f"limit {n_joker}"    
    if after=="True":
        db_df = pd.read_sql_query(f'select sum(n) as tot, gram from {base} where annee={year} and mois={month} and rowid in (select rowid from full_text where gram match \'^{mot}\') group by gram order by tot desc {limit}',conn)
    else:
        db_df = pd.read_sql_query(f'select sum(n) as tot, gram from {base} where annee={year} and mois={month} and rowid in (select rowid from full_text where gram match "{mot}") group by gram order by tot desc {limit}',conn)
    conn.close()
    print(mot)
    return db_df.to_csv(index=False)


@app.route("/associated")
def associated():
    args = request.args
    try:
        corpus=args["corpus"]
    except:
        corpus="lemonde"
    try:
        fr = args["from"]
    except:
        fr=1789
    try:
        to = args["to"]
    except:
        to=2022
    try:
        n_joker = str(args["n_joker"])
    except:
        n_joker = 50
    mot = args.get("mot")
    try:
        n=args["length"]
    except:
        n=len(mot.split(" "))+1
    try:
        stopwords = int(args["stopwords"])
    except:
        stopwords=0
    base = "gram"
    if corpus=="lemonde":base = "gram_mois"
    print(corpus)
    conn = get_db(corpus,n)
    db_df = pd.read_sql_query(f'select sum(n) as tot, gram from {base} where annee between {fr} and {to} and rowid in (select rowid from full_text where gram match "{mot}") group by gram order by tot desc',conn)
    conn.close()
    print("associated")
    z = db_df.gram.str.endswith(f"{mot}")
    zz = db_df.gram.str.match(f"{mot}")
    db_df = db_df.loc[z+zz,]
    db_df["gram"] = db_df.gram.str.split(" ")
    db_df = db_df.explode("gram")
    if n_joker=="all":n_joker = len(db_df.index)
    z = [db_df.gram.values[i] not in mot.split(" ") for i in range(len(db_df.index))]
    db_df = db_df.loc[z,].groupby("gram").agg({"tot":"sum"}).sort_values("tot",ascending=False).reset_index()
    if stopwords>0:
        stopwords = pd.read_csv("/opt/bazoulay/docker_gallicagram/gallicagram/stopwords.csv").iloc[:stopwords,]
        z = db_df.gram.isin(stopwords.monogram)
        db_df = db_df.loc[~z,]
    db_df = db_df.iloc[:int(n_joker),]
    print(mot)
    return db_df.to_csv(index=False)

@app.route('/cooccur')
def cooccur():
    args = request.args
    try:
        fr = args["from"]
    except:
        fr=1945
    try:
        to = args["to"]
    except:
        to = 2022
    try:
        resolution = args["resolution"]
    except:
        resolution = "jour"
    mot1 = args.get("mot1").replace("'","").replace(" ","','")
    mot2 = args.get("mot2").replace("'","").replace(" ","','")
    conn = sqlite3.connect("/opt/bazoulay/ngram/1gram_lemonde_article.db")
    time_steps = "annee"
    if resolution in ["mois","jour"]:time_steps += ",mois"
    if resolution=="jour":time_steps += ",jour"
    query = f"select {time_steps},count(article_id) as n from (select distinct article_id,{time_steps} from gram where gram in ('{mot1}') and annee between {fr} and {to} INTERSECT select distinct article_id,{time_steps} from gram where gram in ('{mot2}') and annee between {fr} and {to}) group by {time_steps}"
    db_df = pd.read_sql(query,conn)
    print(query)
    conn.close()
    base = pd.read_csv("/opt/bazoulay/ngram/base_articles.csv")
    base = base.loc[(base.annee>=int(fr))&(base.annee<=int(to))]
    if resolution=="annee":base = base.groupby("annee").agg({"total":"sum"}).reset_index()
    if resolution=="mois":base = base.groupby(["annee","mois"]).agg({"total":"sum"}).reset_index()
    db_df = pd.merge(db_df,base,how="right")
    db_df.n = db_df.n.fillna(0)
    db_df["gram"] = mot1+"&"+mot2
    print("cooccut" + mot1 + mot2)
    return db_df.to_csv(index=False)


@app.route('/associated_article')
def associated_article():
    args = request.args
    try:
        fr = args["from"]
    except:
        fr=1945
    try:
        to = args["to"]
    except:
        to = 2022
    try:
        n_joker = args["n_joker"]
    except:
        n_joker = 100
    if n_joker=="all":
        n_joker = 10**10 
    else:n_joker = int(n_joker)
    try:
        stopwords = int(args["stopwords"])
    except:
        stopwords = 0
    mot = args.get("mot")
    conn = sqlite3.connect("/opt/bazoulay/ngram/1gram_lemonde_article.db")
    query = f"select gram,sum(n) as tot from gram where article_id in (select article_id from gram where gram=\"{mot}\" and annee between {fr} and {to}) group by gram order by tot desc limit {n_joker+stopwords}"
    print(query)
    db_df = pd.read_sql(query,conn)
    conn.close()
    if stopwords>0:
        stopwords = pd.read_csv("/opt/bazoulay/docker_gallicagram/gallicagram/stopwords.csv").iloc[:stopwords,]
        z = db_df.gram.isin(stopwords.monogram)
        db_df = db_df.loc[~z,]
    print("associated_article " + mot) 
    return db_df.to_csv(index=False)

@app.route('/query_article')
def query_article():
    args = request.args
    try:
        fr = args["from"]
    except:
        fr=1945
    try:
        to = args["to"]
    except:
        to = 2022
    mot = args.get("mot")
    conn = sqlite3.connect("/opt/bazoulay/ngram/1gram_lemonde_article.db")
    query = f"select count(*) as n,gram,annee,mois from gram where gram='{mot}' and annee between {fr} and {to} group by annee,mois"
    db_df = pd.read_sql(query,conn)
    conn.close()
    base = pd.read_csv("/opt/bazoulay/ngram/base_lemonde_articles.csv")
    base = base.loc[(base.annee>=int(fr))&(base.annee<=int(to))]
    db_df = pd.merge(db_df,base,how="left")
    db_df.n = db_df.n.fillna(0)
    return db_df.to_csv(index=False)

@app.route("/query_persee")
def query_persee():
    args = request.args
    word = args.get("mot")
    try:
        revue = args["revue"]
    except:
        revue="all"
    try:
        fr = args["from"]
    except:
        fr=1789
    try:
        to = args["to"]
    except:
        to = 2022
    n = len(word.split(" "))
    revue_condition = ""
    try:
        by_revue = eval(args["by_revue"])
    except:
        by_revue = False
    if revue !="all":
        if " " in revue:
            revue_condition = f"and revue in {tuple(revue.split(' '))}"
        else:
            revue_condition = f"and revue=\"{revue}\""
    print(revue_condition)
    if by_revue:
        group_by = ""
        take_revue = ",revue"
    else:
        group_by = "group by annee";
        take_revue = ""
    conn = sqlite3.connect(f"/opt/bazoulay/ngram/{n}gram_persee.db") 
    if by_revue:
        db_df = pd.read_sql_query(f"SELECT n,annee,gram,revue from gram where gram=\"{word}\" {revue_condition} and annee between {fr} and {to}",conn)
    else:
        db_df = pd.read_sql_query(f"SELECT sum(n) as n,annee,gram from gram where gram=\"{word}\" {revue_condition} and annee between {fr} and {to} group by annee",conn)
    conn.close()
    base = pd.read_csv(f"/opt/bazoulay/docker_gallicagram/gallicagram/persee{n}.csv")
    base.columns = ["total","annee","revue"]
    if revue != 'all':
        base = base.loc[np.isin(base.revue,revue.split(" "))] 
    if not by_revue:
        base = base.groupby("annee").agg({'total':'sum'}).reset_index().sort_values("annee")
    base = base.loc[(base.annee>=int(fr))&(base.annee<=int(to))]
    db_df = pd.merge(db_df,base,how="right")
    db_df.n = db_df.n.fillna(0)
    db_df["gram"] = word
    print(word)
    return db_df.to_csv(index=False)

@lru_cache(maxsize=1)
def load_corpus():
    return pd.read_csv("~/LRFAF/corpus.csv")

corpus_rap = load_corpus()

@app.route("/source_rap")
def source_rap():
    args = request.args
    word = args.get("mot")   
    word = r"\b" + word + r"\b" 
    word = word.replace(r"|",r"\b|\b")
    year = int(args.get("year"))
    print(word)
    print(year)
    corpus = corpus_rap.loc[corpus_rap.year==year]
    corpus = corpus.loc[corpus.lyrics.str.contains(word,case=False)]
    corpus["counts"] = corpus.lyrics.str.count(word,re.I)
    matchs = [re.search(word,lyrics,flags = re.IGNORECASE) for lyrics in corpus.lyrics.values]
    corpus["context_left"] = [corpus.lyrics.values[i][max(0,matchs[i].start()-30):(matchs[i].start())] for i in range(len(corpus.index))]
    corpus["pivot"] = [corpus.lyrics.values[i][matchs[i].start():matchs[i].end()] for i in range(len(corpus.index))]
    corpus["context_right"] = [corpus.lyrics.values[i][max(0,matchs[i].end()):(matchs[i].end() + 30)] for i in range(len(corpus.index))]

    corpus = corpus[["year","artist","title","url","pageviews","counts","context_left","pivot","context_right"]]
    corpus.url = "<a href='" + corpus.url + "' target='_blank'>" + corpus.url + "</a>"
    corpus = corpus.sort_values("pageviews",ascending=False)
    corpus = corpus.drop("pageviews",axis=1)
    corpus.year = corpus.year.astype("int")
    return corpus.set_index("title").to_csv()



import faiss
#from sentence_transformers import SentenceTransformer
import requests
import os
import torch
torch.set_num_threads(1)
os.environ["TOKENIZERS_PARALLELISM"] = "true"
#model = SentenceTransformer("OrdalieTech/Solon-embeddings-large-0.1")
#index = faiss.read_index("/opt/bazoulay/rap.index")
#corpus_faiss = pd.read_csv("/opt/bazoulay/api_gallicagram/corpus_faiss.csv")

def search(query_embedding, k=10):
    distances, indices = index.search(np.array([query_embedding]), k)
    return indices[0]

@app.route("/semantic_search_rap")
def semantic_search_rap():
    query = request.args.get("query")
    k = int(request.args.get("n"))
    print("semantic search" + query)
    query_embedding = model.encode(query)
    result_indices = search(query_embedding,k = k)
    return render_template('index.html', tables=[corpus_faiss.loc[result_indices].to_html(classes='data')], titles=corpus_faiss.columns.values)


