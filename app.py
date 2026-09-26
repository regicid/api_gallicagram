import io
from io import StringIO
import matplotlib.pyplot as plt
import base64
from flask import Flask,render_template,g,request,Response,make_response, send_file, jsonify
import sqlite3
import pandas as pd
import numpy as np
import re
import json
from waitress import serve
from functools import lru_cache
from flask_cors import CORS
import requests
import os

app = Flask(__name__)
CORS(app)

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
    print(DATABASE)
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

def process_input_words(words_input):
    input_words = [w.strip() for w in words_input.split(',')]
    words_to_search = []
    vowels = 'aeiouàâéèêëîïôùûü'
    for word in input_words:
        words_to_search.append(word)
        if word[0].lower() in vowels:
            words_to_search.append(f"l'{word}")
    return words_to_search

def build_query(words_to_search, fr, to, corpus, resolution, rubrique):
    placeholder = ','.join(['?'] * len(words_to_search))
    query_params = words_to_search + [fr, to]
    if resolution == "default" or resolution == "jour" or corpus == "livres" or (resolution == "mois" and corpus in ["presse", "ddb", "lemonde_rubriques","mediapart","lefigaro","lesechos","leparisien","lacroix"]):
        query = f"SELECT * FROM gram WHERE gram IN ({placeholder}) AND annee BETWEEN ? AND ?"
    elif resolution == "annee":
        base = "gram_mois" if corpus in ["libe","lemonde", "huma", "paris", "figaro", "moniteur", "temps", "petit_journal", "constitutionnel", "journal_des_debats", "la_presse", "petit_parisien"] else "gram"
        query = f"SELECT sum(n) as n, annee, gram FROM {base} WHERE gram IN ({placeholder}) AND annee BETWEEN ? AND ? GROUP BY annee, gram"
    elif resolution == "mois" and corpus in ["libe","lemonde", "huma", "paris", "figaro", "moniteur", "temps", "petit_journal", "constitutionnel", "journal_des_debats", "la_presse", "petit_parisien"]:
        query = f"SELECT * FROM gram_mois WHERE gram IN ({placeholder}) AND annee BETWEEN ? AND ?"
    if corpus == "lemonde_rubriques":
        query, query_params = handle_lemonde_rubriques(query, query_params, rubrique, resolution)
    return query, query_params

def handle_lemonde_rubriques(query, query_params, rubrique, resolution):
    by_rubrique = request.args.get("by_rubrique", "False").lower() == "true"
    if rubrique:
        rubrique_list = rubrique.split()
        if 1 < len(rubrique_list) < 8:
            rubrique_condition = f"AND rubrique IN ({','.join(['?'] * len(rubrique_list))})"
            query_params = query_params[:-2] + rubrique_list + query_params[-2:]
        elif len(rubrique_list) != 8:
            rubrique_condition = "AND rubrique = ?"
            query_params = query_params[:-2] + [rubrique] + query_params[-2:]
        else:
            rubrique_condition = ""
        query = query.replace("AND annee BETWEEN", f'{rubrique_condition} AND annee BETWEEN')
    
    # Add rubrique to SELECT and GROUP BY when by_rubrique is True and resolution is "annee"
    if by_rubrique and resolution == "annee":
        query = query.replace("SELECT sum(n) as n, annee, gram", "SELECT sum(n) as n, annee, gram, rubrique")
        query = query.replace("GROUP BY annee, gram", "GROUP BY annee, gram, rubrique")
    # Adjust the query for the "mois" resolution
    elif resolution == "mois":
        query = query.replace("SELECT sum(n) as n, annee, gram", "SELECT sum(n) as n, annee, mois, gram")
        query = query.replace("GROUP BY annee, gram", "GROUP BY annee, mois, gram")
    
    return query, query_params





def process_results(db_df, base, corpus, resolution, rubrique):
    if corpus == "lemonde_rubriques":
        print(db_df)
        by_rubrique = request.args.get("by_rubrique", "False").lower() == "true"   
        if rubrique:
            base = base[np.isin(base.rubrique, rubrique.split())] 
        grouping = ["annee"]
        if resolution == "mois":
            grouping.append("mois")
        if by_rubrique:
            grouping.append("rubrique")
        # Sum the total for base
        base = base.groupby(grouping).agg({"total": "sum"}).reset_index()       
        # Prepare db_df
        if "rubrique" not in db_df.columns and by_rubrique:
            # If rubrique is not in db_df but by_rubrique is True, 
            # we need to add it to properly merge with base
            merge_cols = ["annee", "mois"] if "mois" in base.columns else ["annee"]
            db_df = db_df.merge(base[merge_cols + ["rubrique"]], on=merge_cols, how="right")
        else:
            # Ensure db_df has all the necessary columns for merging
            for col in grouping:
                if col not in db_df.columns:
                    db_df[col] = None
        # Group db_df to remove potential duplicates
        db_df = db_df.groupby(grouping + ["gram"]).agg({"n": "sum"}).reset_index()
        # Merge db_df with base
        db_df = pd.merge(db_df, base, on=grouping, how="outer")
        # Fill NaN values
        db_df["n"] = db_df["n"].fillna(0)
        db_df["total"] = db_df["total"].fillna(0)
        db_df["gram"] = db_df["gram"].fillna(db_df["gram"].iloc[0] if len(db_df) > 0 else "") 
    elif resolution == "mois" and corpus in ["libe","lemonde", "huma", "paris", "figaro", "moniteur", "temps", "petit_journal", "constitutionnel", "journal_des_debats", "la_presse", "petit_parisien", "presse","mediapart"]:
        base = base.groupby(["annee", "mois"]).agg({'total': 'sum'}).reset_index()
        db_df = pd.merge(db_df, base, on=["annee", "mois"], how="outer")
        db_df["n"] = db_df["n"].fillna(0)
    elif resolution == "annee" and corpus in ["libe","lemonde", "huma", "paris", "figaro", "moniteur", "temps", "petit_journal", "constitutionnel", "journal_des_debats", "la_presse", "petit_parisien", "presse","mediapart","lefigaro","leparisien","lacroix","lesechos"]:
        base = base.groupby(["annee"]).agg({'total': 'sum'}).reset_index()
        db_df = pd.merge(db_df, base, on=["annee"], how="outer")
        db_df["n"] = db_df["n"].fillna(0)
    else:
        # For other cases, perform a simple outer merge
        db_df = pd.merge(db_df, base, how="outer")
        db_df["n"] = db_df["n"].fillna(0)
    if corpus == "livres":
        db_df = db_df.sort_values("annee")
    return db_df




@app.route("/query")
def query():
    args = request.args
    words_input = args.get("mot", "").lower()
    corpus = args.get("corpus", "presse")
    fr = args.get("from", 1789)
    to = args.get("to", 2022)
    resolution = args.get("resolution", "default")
    rubrique = args.get("rubrique")
    words_to_search = process_input_words(words_input)
    n = max(len(word.split()) for word in words_input.split(','))
    if corpus=="prenoms":n = 1
    conn = get_db(corpus, n)
    query, query_params = build_query(words_to_search, fr, to, corpus, resolution, rubrique)
    print(query)
    print(query_params)
    db_df = pd.read_sql_query(query, conn, params=query_params)
    conn.close()
    print(db_df)
    base = get_base(corpus, n)
    base = base[(base.annee >= int(fr)) & (base.annee <= int(to))]
    db_df = process_results(db_df, base, corpus, resolution, rubrique)
    # Instead of setting gram column to words_input, we'll ensure it's correctly populated in process_results
    return db_df.to_csv(index=False)

@app.route('/contain')
def contain():
    args = request.args
    try:
        corpus=args["corpus"]
    except:
        corpus="lemonde"
    print(corpus)
    mot1 = args.get("mot1").replace("'","").lower()
    mot2 = args.get("mot2").replace("'","").lower()
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
    mot = args.get("mot").lower()
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
    mot = args.get("mot").lower()
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
    mot = args.get("mot").lower()
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
    mot1 = args.get("mot1").replace("'","").replace(" ","','").lower()
    mot2 = args.get("mot2").replace("'","").replace(" ","','").lower()
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
        n_joker = str(args["n_joker"])
    except:
        n_joker = 50
    if n_joker=="all":
        n_joker = 10**10 
    else:n_joker = int(n_joker)
    try:
        stopwords = int(args["stopwords"])
    except:
        stopwords = 0
    mot = args.get("mot").lower()
    conn = sqlite3.connect("/opt/bazoulay/ngram/1gram_lemonde_article.db")
    query = f"select gram,sum(n) as tot from gram where article_id in (select article_id from gram where gram=\"{mot}\" and annee between {fr} and {to}) group by gram order by tot desc limit {n_joker+stopwords}"
    print(query)
    db_df = pd.read_sql(query,conn)
    conn.close()
    if stopwords>0:
        stopwords = pd.read_csv("/opt/bazoulay/docker_gallicagram/gallicagram/stopwords.csv").iloc[:stopwords,]
        z = db_df.gram.isin(stopwords.monogram)
        db_df = db_df.loc[~z,]
    db_df = db_df.iloc[:n_joker]
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
    resolution = args.get("resolution", "mois")
    mot = args.get("mot").lower()
    conn = sqlite3.connect("/opt/bazoulay/ngram/1gram_lemonde_article.db")
    time_steps = "annee"
    if resolution in ["mois","jour"]:time_steps += ",mois"
    if resolution=="jour":time_steps += ",jour"
    query = f"select count(*) as n,gram,{time_steps} from gram where gram='{mot}' and annee between {fr} and {to} group by {time_steps}"
    db_df = pd.read_sql(query,conn)
    conn.close()
    base = pd.read_csv("/opt/bazoulay/ngram/base_articles.csv")
    base = base.loc[(base.annee>=int(fr))&(base.annee<=int(to))]
    if resolution=="annee":base = base.groupby("annee").agg({"total":"sum"}).reset_index()
    if resolution=="mois":base = base.groupby(["annee","mois"]).agg({"total":"sum"}).reset_index()
    db_df = pd.merge(db_df,base,how="right")
    db_df.n = db_df.n.fillna(0)
    db_df["gram"] = mot
    return db_df.to_csv(index=False)

@app.route("/query_persee")
def query_persee():
    args = request.args
    word = args.get("mot").lower()
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
    if revue !="all" and len(revue.split(' ')) < 362:
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

cairn_revues = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "cairn_revues.csv"))
cairn_disciplines = {}
for code, disciplines in zip(cairn_revues["Code CAIRN"], cairn_revues["Disciplines"].fillna("")):
    for d in disciplines.split(";"):
        if d.strip():
            cairn_disciplines.setdefault(d.strip().lower(), set()).add(code)

@app.route("/query_cairn")
def query_cairn():
    # revue : codes CAIRN séparés par des espaces ; discipline : disciplines séparées par des virgules
    # Les deux se cumulent (union). Résolution annuelle uniquement.
    args = request.args
    word = args.get("mot", "").lower().strip()
    words = word.split()
    n = len(words)
    if n not in (1, 2, 3):
        return Response("mot doit contenir entre 1 et 3 mots", status=400)
    fr = int(args.get("from", 1958))
    to = int(args.get("to", 2030))
    by_revue = args.get("by_revue", "False").lower() in ("true", "1")
    revues = set(args.get("revue", "all").split()) - {"all"}
    for d in args.get("discipline", "").split(","):
        if not d.strip():
            continue
        if d.strip().lower() not in cairn_disciplines:
            return Response(f"discipline inconnue : {d.strip()}", status=400)
        revues |= cairn_disciplines[d.strip().lower()]
    revue_condition, revue_params = "", []
    if revues:
        revue_condition = f"and revue in ({','.join('?' * len(revues))})"
        revue_params = sorted(revues)
    table = ["unigram", "bigram", "trigram"][n - 1]
    group = "date, revue" if by_revue else "date"
    conn = sqlite3.connect("/opt/bazoulay/ngram/cairn_ngram.db")
    ids = [conn.execute("select id from token where word=?", (w,)).fetchone() for w in words]
    base = pd.read_sql_query(f"select date as annee, {'revue, ' if by_revue else ''}sum(total) as total from total_{table} where date between ? and ? {revue_condition} group by {group}", conn, params=[fr, to] + revue_params)
    if all(ids):
        w_condition = " and ".join(f"w{i+1}=?" for i in range(n))
        db_df = pd.read_sql_query(f"select date as annee, {'revue, ' if by_revue else ''}sum(n) as n from {table} where {w_condition} and date between ? and ? {revue_condition} group by {group}", conn, params=[i[0] for i in ids] + [fr, to] + revue_params)
    else:
        db_df = pd.DataFrame(columns=["annee", "revue", "n"] if by_revue else ["annee", "n"])
    conn.close()
    db_df = pd.merge(db_df, base, how="right")
    db_df.n = db_df.n.fillna(0).astype(int)
    db_df["gram"] = word
    db_df = db_df.sort_values(["annee", "revue"] if by_revue else "annee")
    print("cairn " + word)
    return db_df[["n", "annee", "gram"] + (["revue"] if by_revue else []) + ["total"]].to_csv(index=False)

tv_corpora = ("bfmtv", "cnews", "franceinfo")

@app.route("/query_tv")
def query_tv():
    # corpus : bfmtv, cnews ou franceinfo. Dates stockées en AAAAMMJJ (résolution journalière).
    # from/to : AAAA, AAAAMM ou AAAAMMJJ ; resolution : jour (défaut), mois ou annee.
    args = request.args
    corpus = args.get("corpus", "")
    if corpus not in tv_corpora:
        return Response(f"corpus doit être parmi : {', '.join(tv_corpora)}", status=400)
    word = args.get("mot", "").lower().strip()
    words = word.split()
    n = len(words)
    if n not in (1, 2, 3):
        return Response("mot doit contenir entre 1 et 3 mots", status=400)
    resolution = args.get("resolution", "jour")
    if resolution not in ("jour", "mois", "annee"):
        return Response("resolution doit être jour, mois ou annee", status=400)
    fr = args.get("from", "1900")
    to = args.get("to", "2100")
    if not (fr.isdigit() and to.isdigit() and len(fr) in (4, 6, 8) and len(to) in (4, 6, 8)):
        return Response("from/to doivent être au format AAAA, AAAAMM ou AAAAMMJJ", status=400)
    fr = int(fr.ljust(8, "0"))
    to = int(to + "9" * (8 - len(to)))
    table = ["unigram", "bigram", "trigram"][n - 1]
    conn = sqlite3.connect(f"/opt/bazoulay/ngram/{corpus}_ngram.db")
    ids = [conn.execute("select id from token where word=?", (w,)).fetchone() for w in words]
    base = pd.read_sql_query(f"select date, total from total_{table} where date between ? and ?", conn, params=[fr, to])
    if all(ids):
        w_condition = " and ".join(f"w{i+1}=?" for i in range(n))
        db_df = pd.read_sql_query(f"select date, n from {table} where {w_condition} and date between ? and ?", conn, params=[i[0] for i in ids] + [fr, to])
    else:
        db_df = pd.DataFrame(columns=["date", "n"])
    conn.close()
    db_df = pd.merge(db_df, base, how="right")
    db_df.n = db_df.n.fillna(0).astype(int)
    db_df["annee"] = db_df.date // 10000
    db_df["mois"] = db_df.date // 100 % 100
    db_df["jour"] = db_df.date % 100
    time_steps = {"annee": ["annee"], "mois": ["annee", "mois"], "jour": ["annee", "mois", "jour"]}[resolution]
    db_df = db_df.groupby(time_steps).agg({"n": "sum", "total": "sum"}).reset_index().sort_values(time_steps)
    db_df["gram"] = word
    print(corpus + " " + word)
    return db_df[["n"] + time_steps + ["gram", "total"]].to_csv(index=False)

ngram_tables = ["unigram", "bigram", "trigram", "quadrigram", "pentagram"]
ngram_steps = ["annee", "mois", "jour"]

@lru_cache(maxsize=None)
def ngram_schema(db_path, table):
    # Colonnes de temps d'une table token : annee[, mois[, jour]] séparées,
    # ou une seule colonne date au format AAAA, AAAAMM ou AAAAMMJJ (cairn, tv).
    # Renvoie (pas de temps disponibles, largeur de date ou None), ou None si la table n'existe pas.
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    cols = [c[1] for c in conn.execute(f"pragma table_info({table})")]
    if "date" in cols:
        d = conn.execute(f"select date from {table} limit 1").fetchone()
        conn.close()
        return (tuple(ngram_steps[:(len(str(d[0])) - 2) // 2]), len(str(d[0]))) if d else None
    conn.close()
    steps = tuple(s for s in ngram_steps if s in cols)
    return (steps, None) if steps else None

def ngram_time_sql(steps, width, resolution, fr, to):
    # select, group by et condition de période pour les pas de temps jusqu'à `resolution`.
    # fr/to sont des chaînes AAAAMMJJ complétées ; la condition porte sur la clé primaire.
    out = steps[:steps.index(resolution) + 1]
    bounds = lambda s: [int(s[0:4]), int(s[4:6]), int(s[6:8])][:len(steps)]
    if width is None:
        select = ", ".join(out)
        cols = f"({', '.join(steps)})"
        period = f"{cols} between ({', '.join('?' * len(steps))}) and ({', '.join('?' * len(steps))})"
        return select, ", ".join(out), period, bounds(fr) + bounds(to)
    scale = 10 ** (width - 4)
    exprs = {"annee": f"date / {scale}", "mois": f"date / {max(scale // 100, 1)} % 100", "jour": "date % 100"}
    select = ", ".join(f"{exprs[s]} as {s}" for s in out)
    return select, ", ".join(out), "date between ? and ?", [int(fr[:width]), int(to[:width])]

@app.route("/query_ngram")
def query_ngram():
    # Route générique pour les bases token {corpus}_ngram.db (token + unigram/bigram/... + total_*).
    # mot : ',' sépare des séries, '+' additionne des variantes ; 1 à 5 mots par n-gramme.
    # from/to : AAAA, AAAAMM ou AAAAMMJJ ; resolution : annee, mois ou jour (défaut : la plus fine disponible).
    # Si la table total_* manque, les totaux viennent des CSV de get_base.
    args = request.args
    corpus = args.get("corpus", "")
    if not re.fullmatch(r"[a-z0-9_]+", corpus):
        return Response("corpus invalide", status=400)
    db_path = f"/opt/bazoulay/ngram/{corpus}_ngram.db"
    if not os.path.exists(db_path):
        return Response(f"corpus inconnu : {corpus}", status=400)
    fr = args.get("from", "1000")
    to = args.get("to", "2100")
    if not (fr.isdigit() and to.isdigit() and len(fr) in (4, 6, 8) and len(to) in (4, 6, 8)):
        return Response("from/to doivent être au format AAAA, AAAAMM ou AAAAMMJJ", status=400)
    fr = fr.ljust(8, "0")
    to = to + "9" * (8 - len(to))
    series = [[v.lower().split() for v in s.split("+")] for s in args.get("mot", "").split(",")]
    if not all(v and all(0 < len(w) <= len(ngram_tables) for w in v) for v in series):
        return Response(f"mot doit contenir entre 1 et {len(ngram_tables)} mots par n-gramme", status=400)
    resolution = args.get("resolution", "default")
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    results = []
    for variants in series:
        n = len(variants[0])
        if any(len(w) != n for w in variants):
            return Response("les variantes d'une même série ('+') doivent avoir le même nombre de mots", status=400)
        table = ngram_tables[n - 1]
        schema = ngram_schema(db_path, table)
        if schema is None:
            return Response(f"pas de table {table} pour le corpus {corpus}", status=400)
        steps, width = schema
        res = steps[-1] if resolution == "default" else resolution
        if res not in steps:
            return Response(f"resolution doit être parmi : {', '.join(steps)}", status=400)
        select, group, period, period_params = ngram_time_sql(steps, width, res, fr, to)
        out = group.split(", ")
        # Comptes : une requête par variante (clé primaire w1, w2, ..., temps), sommées ensuite
        counts = []
        w_condition = " and ".join(f"w{i+1}=?" for i in range(n))
        for words in variants:
            ids = [conn.execute("select id from token where word=?", (w,)).fetchone() for w in words]
            if all(ids):
                counts.append(pd.read_sql_query(f"select {select}, sum(n) as n from {table} where {w_condition} and {period} group by {group}", conn, params=[i[0] for i in ids] + period_params))
        db_df = pd.concat(counts).groupby(out, as_index=False).n.sum() if counts else pd.DataFrame(columns=out + ["n"])
        # Totaux
        if ngram_schema(db_path, f"total_{table}") is not None:
            base = pd.read_sql_query(f"select {select}, sum(total) as total from total_{table} where {period} group by {group}", conn, params=period_params)
        else:
            base = get_base(corpus, n)
            if any(s not in base.columns for s in out):
                return Response(f"pas de totaux à la résolution {res} pour le corpus {corpus}", status=400)
            base_steps = [s for s in ngram_steps if s in base.columns]
            key = sum(base[s] * 100 ** (len(base_steps) - 1 - i) for i, s in enumerate(base_steps))
            k = 2 * len(base_steps) + 2
            base = base[(key >= int(fr[:k])) & (key <= int(to[:k]))]
            base = base.groupby(out, as_index=False).total.sum()
        db_df = pd.merge(db_df, base, how="right", on=out)
        db_df.n = db_df.n.fillna(0).astype(int)
        db_df["gram"] = "+".join(" ".join(w) for w in variants)
        results.append(db_df.sort_values(out)[["n"] + out + ["gram", "total"]])
    conn.close()
    print(corpus + " " + args.get("mot", ""))
    return pd.concat(results).to_csv(index=False)

corpus_rap =pd.read_csv("~/LRFAF/corpus.csv")

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



#import faiss
#from sentence_transformers import SentenceTransformer
import requests
import os
#import torch
#torch.set_num_threads(1)
#os.environ["TOKENIZERS_PARALLELISM"] = "true"
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


@app.route('/h_split')
def h_split():
    args = request.args
    author_id = args["author_id"]
    print(author_id)
    from ulysse_stuff.custom_index import get_all_h_indexes
    result = get_all_h_indexes(author_id)
    overall_index = result['overall']
    first_index = result['first']
    middle_index = result['middle']
    last_index = result['last']
    print(result)
    return jsonify(result)

@app.route('/app',methods=['GET', 'POST'])
def web_interface():
    print("Request received:", request.method)
    if request.method == 'POST':
        # Get form data
        speaker_1 = request.form.get('speaker_1')
        speaker_2 = request.form.get('speaker_2')
        beginning = int(request.form.get('beginning'))
        end = int(request.form.get('end'))
        print(speaker_1)
        # Process data
        verbs, odd_ratios,variance = process_data(speaker_1, speaker_2, beginning, end)
        
        return jsonify({
            "verbs": list(verbs),
            "odd_ratios": list(odd_ratios),  # This will be your list of numbers
            "variance":list(variance)
        })
    
    return render_template('app.html')

def process_data(speaker_1, speaker_2, beginning, end):
    verbs_1 = []
    verbs_2 = []
    cues = pd.read_csv("https://raw.githubusercontent.com/gillesbastin/old_fashion_nlp/refs/heads/main/cues_all.csv").iloc[:,:2]
    verbs = cues.lemmatized_cue.unique()
    for year in np.arange(beginning,end+1):
        for month in range(12):
            month = month+1
            try:
                files = os.listdir(f"/opt/bazoulay/quotes/quotes_{year}_{month}")
            except:
                continue
            genderized_files = [f for f in files if f.endswith("genderized.json")]
            for i in range(len(genderized_files)):
                with open(f"/opt/bazoulay/quotes/quotes_{year}_{month}/{genderized_files[i]}","r") as f:
                    a = json.load(f)
                for z in a:
                    if speaker_1 in z["speaker"]:
                        verbs_1.append(z["verb"])
                    if speaker_2 in z["speaker"]:
                        verbs_2.append(z["verb"])
            print(month)
        print(year)
    verbs_1 = pd.DataFrame(verbs_1,columns = ["cue"])
    verbs_2 = pd.DataFrame(verbs_2,columns = ["cue"])
    verbs_1= verbs_1.merge(cues,on="cue").lemmatized_cue.value_counts().reindex(verbs,fill_value=0)
    verbs_2 = verbs_2.merge(cues,on="cue").lemmatized_cue.value_counts().reindex(verbs,fill_value=0)
    odds_ratio = np.log2((verbs_1/verbs_1.sum())/(verbs_2/verbs_2.sum()))
    variance = 1/verbs_1 + 1/verbs_2
    z = np.logical_and(verbs_1 > 0, verbs_2 > 0)
    z = np.logical_and(verbs_1 + verbs_2 > 25,z)
    o = odds_ratio.loc[z].sort_values()
    o = pd.concat([o.head(10), o.tail(10)]).drop_duplicates()
    print(o)
    return o.index, o.values, variance[o.index].values



def forward_request(target_url):
    try:
        # stream=True est indispensable pour que les flux de données (comme SSE / Server-Sent Events)
        # ne soient pas mis en mémoire tampon (bufférisés) par le proxy.
        resp = requests.request(
            method=request.method,
            url=target_url,
            params=request.args,
            headers={key: value for key, value in request.headers if key.lower() != 'host'},
            data=request.get_data(),
            cookies=request.cookies,
            allow_redirects=False,
            timeout=30,
            stream=True  # Support du streaming de données
        )
        
        excluded_headers = ['content-encoding', 'content-length', 'transfer-encoding', 'connection']
        headers = [(name, value) for name, value in resp.raw.headers.items()
                   if name.lower() not in excluded_headers]
        
        # Si la réponse est un flux d'événements (Server-Sent Events pour MCP), 
        # on redirige le flux en continu vers le client.
        if 'text/event-stream' in resp.headers.get('Content-Type', ''):
            def generate():
                for chunk in resp.iter_content(chunk_size=1024):
                    yield chunk
            return Response(generate(), resp.status_code, headers)
            
        # Sinon, retour standard de la réponse
        return Response(resp.content, resp.status_code, headers)
    except requests.exceptions.RequestException as e:
        return {'error': f'Service indisponible : {str(e)}'}, 503


# 1. Serveur MCP (Port 8003) - Exposé sous /v2/mcp/
# Cette route est plus spécifique et doit intercepter les appels MCP en premier
@app.route('/v2/mcp/<path:subpath>', methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'])
def proxy_to_mcp(subpath):
    # Les appels vers /v2/mcp/mcp ou /v2/mcp/sse iront vers http://127.0.0.1:8003/mcp ou /sse
    url = f'http://127.0.0.1:8003/{subpath}'
    return forward_request(url)


# 2. Gallicagram API V2 (Port 8002) - Exposée sous /v2/
# Les routes de app.py commençant déjà par "/v2/", on conserve ce préfixe dans l'URL de destination.
@app.route('/v2/<path:subpath>', methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'])
def proxy_to_v2_api(subpath):
    # Les appels vers /v2/query iront vers http://127.0.0.1:8002/v2/query
    url = f'http://127.0.0.1:8002/v2/{subpath}'
    return forward_request(url)


# 3. (Optionnel) Ancienne API V1 si toujours nécessaire (Port 8001)
@app.route('/api/<path:subpath>', methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'])
def proxy_to_v1_api(subpath):
    url = f'http://127.0.0.1:8001/api/{subpath}'
    return forward_request(url)


# 4. API Agora (Port 8010) - Presse quotidienne stage-mids, exposée sous /agora/
# Le service tourne à part (tmux agora, code : github.com/corto2corto/ngram-press)
@app.route('/agora/<path:subpath>', methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'])
def proxy_to_agora(subpath):
    # Les appels vers /agora/query iront vers http://127.0.0.1:8010/query
    url = f'http://127.0.0.1:8010/{subpath}'
    return forward_request(url)