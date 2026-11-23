import io
from io import StringIO
import matplotlib.pyplot as plt
import base64
from flask import Flask,render_template,g,request,Response,make_response, send_file, jsonify
import sqlite3
import pandas as pd
import numpy as np
import re
import unicodedata
import json
from waitress import serve
from functools import lru_cache
from flask_cors import CORS
import requests
import os
from mots_du_jour import mots_du_jour as calcul_mots_du_jour

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
    vowels = 'aeiouyàâäéèêëîïôöùûüÿœæh'
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

ngram_dir = "/opt/bazoulay/ngram"
elias_dir = "/opt/bazoulay/stage-mids/bases"

def ngram_bases(args, corpus):
    # Bases d'un corpus : [{corpus}_ngram.db] dans ngram_dir, ou avec elias=true (presse web contemporaine,
    # mise à jour continue de stage-mids) la grosse base + les tampons donnés par version.json, relu à chaque
    # requête (la fusion mensuelle le remplace d'un coup). Liste vide si le corpus est inconnu.
    if args.get("elias", "false").lower() not in ("true", "1"):
        path = f"{ngram_dir}/{corpus}_ngram.db"
        return [path] if os.path.exists(path) else []
    with open(f"{elias_dir}/version.json") as f:
        e = json.load(f).get(corpus)
    return [f"{elias_dir}/{b}" for b in [e["grosse"], *e["tampons"]]] if e else []

def ngram_connect(bases):
    # Connexion en lecture seule à la première base ; les suivantes (tampons) sont attachées en t0, t1...
    # Les ids viennent du registre commun de stage-mids : un mot a le même id dans toutes les bases, et figure
    # dans la table token de chacune où il apparaît. La vue temporaire token (prioritaire sur main.token)
    # les réunit sans doublon ; les tables de comptes sont réunies par ngram_from.
    conn = sqlite3.connect(f"file:{bases[0]}?mode=ro", uri=True)
    schemas = ["main"] + [f"t{k}" for k in range(len(bases) - 1)]
    for s, b in zip(schemas[1:], bases[1:]):
        conn.execute(f"attach ? as {s}", (f"file:{b}?mode=ro",))
    if len(schemas) > 1:
        conn.execute("create temp view token as " + " union all ".join(
            f"select id, word from {s}.token" + "".join(f"{' where' if j == 0 else ' and'} id not in (select id from {p}.token)" for j, p in enumerate(schemas[:k]))
            for k, s in enumerate(schemas)))
    # Une seule transaction de lecture : comptes et totaux voient le même état, même si un ajout
    # est validé dans un tampon (mode WAL) pendant la requête
    conn.execute("begin")
    return conn

def ngram_from(conn, table, where, params):
    # Clause "from table where ..." et ses paramètres. Avec des tampons attachés, une sous-requête par base,
    # mises bout à bout (les comptes sont ensuite sommés par le group by) : une vue UNION ALL ne convient pas,
    # SQLite n'y pousse pas une condition contenant une sous-requête (w1 in (select ...) -> scan complet).
    schemas = [r[1] for r in conn.execute("pragma database_list") if r[1] != "temp"]
    if len(schemas) == 1:
        return f"from {table} where {where}", params
    return "from (" + " union all ".join(f"select * from {s}.{table} where {where}" for s in schemas) + ")", params * len(schemas)

ngram_tables = ["unigram", "bigram", "trigram", "fourgram", "fivegram"]  # noms écrits par convert_ngram.py
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

@lru_cache(maxsize=None)
def ngram_fields(db_path, table):
    # Champs propres au corpus (rubrique, revue...) : colonnes qui ne sont ni des mots (w1, w2...), ni du temps,
    # ni des comptes. Renvoie {champ: {label: code} ou None} ; un champ codé en entier par convert_ngram.py
    # (--encode) a une table de correspondance du même nom (id, label).
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    cols = [c[1] for c in conn.execute(f"pragma table_info({table})")]
    tables = {r[0] for r in conn.execute("select name from sqlite_master where type='table'")}
    fields = {}
    for c in cols:
        if re.fullmatch(r"w\d+", c) or c in ngram_steps or c in ("date", "n", "total"):
            continue
        fields[c] = dict(conn.execute(f'select label, id from "{c}"')) if c in tables else None
    conn.close()
    return fields

# Alias propres à un corpus : filtre -> (champ, {valeur: ensemble de valeurs du champ})
ngram_field_aliases = {"cairn": {"discipline": ("revue", cairn_disciplines)}}

def ngram_field_values(s, known=()):
    # Valeurs séparées par ',' ou ';' ('all' = pas de filtre). Des morceaux voisins sont recollés
    # quand ils forment une valeur connue contenant une virgule (discipline 'Info, Communication').
    parts = [p.strip() for p in re.split("[,;]", s)]
    values, i = [], 0
    while i < len(parts):
        j = next((j for j in range(len(parts), i + 1, -1) if ", ".join(parts[i:j]).lower() in known), i + 1)
        values.append(", ".join(parts[i:j]))
        i = j
    return {v for v in values if v and v.lower() != "all"}

def ngram_fields_sql(corpus, db_path, table, args):
    # Filtres optionnels sur les champs du corpus : rubrique=politique,culture ; revue=RFS,ARSS (voir
    # ngram_field_values). Un paramètre qui ne correspond à aucun champ du corpus est ignoré.
    # Renvoie (condition à ajouter à la période, params, None) ou (None, None, message d'erreur).
    fields = ngram_fields(db_path, table)
    labels = {c: {label.lower(): id for label, id in fields[c].items()} for c in fields if fields[c] is not None}
    values = {c: ngram_field_values(args.get(c, ""), labels.get(c, ())) for c in fields}
    for alias, (c, mapping) in ngram_field_aliases.get(corpus, {}).items():
        for v in ngram_field_values(args.get(alias, ""), mapping):
            if v.lower() not in mapping:
                return None, None, f"{alias} inconnu(e) : {v}"
            values[c] |= mapping[v.lower()]
    condition, params = "", []
    for c, vs in values.items():
        if not vs:
            continue
        if c in labels:
            unknown = {v for v in vs if v.lower() not in labels[c]}
            if unknown:
                return None, None, f"{c} inconnu(e) : {', '.join(sorted(unknown))} ; valeurs possibles : {', '.join(fields[c])}"
            vs = {labels[c][v.lower()] for v in vs}
        condition += f' and "{c}" in ({",".join("?" * len(vs))})'
        params += sorted(vs)
    return condition, params, None

ngram_elision = tuple("aeiouyàâäéèêëîïôöùûüÿœæh")  # 'orange' compte aussi 'l'orange', comme /query

ngram_max_forms = 500000       # nombre max de formes pour un motif joker (*, ?) : ~5 µs par forme sur presse
ngram_scan_limit = 2 * 10**9   # taille (octets) au-delà de laquelle on refuse les scans complets

def ngram_words_sql(conn, db_path, words):
    # Conditions sur w1, w2, ... pour un n-gramme dont chaque mot est exact, '_' (n'importe quel mot)
    # ou un motif glob avec * (0 ou plusieurs caractères) et ? (un caractère).
    # Renvoie (conditions, params, None) ou (None, None, message d'erreur).
    if all(w == "_" for w in words):
        return None, None, "le motif doit contenir au moins un mot autre que '_'"
    # Sans premier mot fixé, la clé primaire (w1, w2, ..., temps) ne sert pas : scan complet
    if os.path.getsize(db_path) > ngram_scan_limit and (words[0] == "_" or words[0][:1] in "*?"):
        return None, None, "sur ce corpus, le premier mot ne peut pas être '_' ni commencer par * ou ?"
    conditions, params = [], []
    table = ngram_tables[len(words) - 1]
    for i, w in enumerate(words):
        if w == "_":
            continue
        if ("*" in w or "?" in w) and len(conditions) == i > 0:
            # Mots précédents tous fixés : candidats = mots vus à cette position après eux (préfixe de la clé
            # primaire), filtrés par le motif. Sur presse, 'droit d*' : 34 000 candidats en 0,6 s, contre
            # 40 millions de formes d* dans token (2 min)
            sql, sql_params = ngram_from(conn, table, " and ".join(conditions), list(params))
            ids = [r[0] for r in conn.execute(
                f"select t.id from (select distinct w{i+1} as id {sql}) c cross join token t on t.id=c.id where t.word glob ?",
                sql_params + [w])]
            if len(ids) > ngram_max_forms:
                return None, None, f"le motif {w} correspond à plus de {ngram_max_forms} formes, le préciser"
            conditions.append(f"w{i+1} in ({','.join(map(str, ids)) or '-1'})")
            continue
        if "*" in w or "?" in w:
            forms = conn.execute("select count(*) from (select 1 from token where word glob ? limit ?)", (w, ngram_max_forms + 1)).fetchone()[0]
            if forms > ngram_max_forms:
                return None, None, f"le motif {w} correspond à plus de {ngram_max_forms} formes, le préciser"
            conditions.append(f"w{i+1} in (select id from token where word glob ?)")
            params.append(w)
        else:
            id = conn.execute("select id from token where word=?", (w,)).fetchone()
            conditions.append(f"w{i+1}=?")
            params.append(id[0] if id else -1)
    return conditions, params, None

@app.route("/query_ngram")
def query_ngram():
    # Route générique pour les bases token {corpus}_ngram.db (token + unigram/bigram/... + total_*).
    # mot : ',' sépare des séries, '+' additionne des variantes ; 1 à 5 mots par n-gramme.
    # Dans un n-gramme, '_' remplace n'importe quel mot ('guerre _ l'allemagne') et * / ? servent
    # de jokers dans un mot ('grèv*' additionne grève, grèves, grèvent...) ; chaque série reste une courbe.
    # from/to : AAAA, AAAAMM ou AAAAMMJJ ; resolution : annee, mois ou jour (défaut : la plus fine disponible).
    # Champs propres au corpus (rubrique sur lemonde_rubriques, revue sur cairn...) : rubrique=politique,culture
    # restreint comptes et totaux (voir ngram_fields_sql) ; by_rubrique=True sépare les courbes par rubrique.
    # Si la table total_* manque, les totaux viennent des CSV de get_base.
    # Un n-gramme dont le premier mot commence par une voyelle ou un h compte aussi sa forme élidée
    # ('orange' -> orange + l'orange, 'histoire de' -> + l'histoire de) ; la série garde le nom saisi.
    args = request.args
    corpus = args.get("corpus", "")
    if not re.fullmatch(r"[a-z0-9_]+", corpus):
        return Response("corpus invalide", status=400)
    bases = ngram_bases(args, corpus)
    if not bases:
        return Response(f"corpus inconnu : {corpus}", status=400)
    db_path = bases[0]  # schéma, champs et taille : ceux de la grosse base
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
    conn = ngram_connect(bases)
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
        fields = ngram_fields(db_path, table)
        condition, field_params, error = ngram_fields_sql(corpus, db_path, table, args)
        if error:
            conn.close()
            return Response(error, status=400)
        period += condition
        period_params += field_params
        by = [c for c in fields if args.get(f"by_{c}", "").lower() in ("true", "1")]
        select = ", ".join([select] + by)
        group = ", ".join([group] + by)
        out = group.split(", ")
        # Comptes : une requête par variante (clé primaire w1, w2, ..., temps), sommées ensuite
        series_label = "+".join(" ".join(w) for w in variants)
        elided = [["l'" + w[0]] + w[1:] for w in variants if w[0].startswith(ngram_elision)]
        variants = variants + [w for w in elided if w not in variants]
        counts = []
        for words in variants:
            conditions, params, error = ngram_words_sql(conn, db_path, words)
            if error:
                conn.close()
                return Response(error, status=400)
            sql, sql_params = ngram_from(conn, table, f"{' and '.join(conditions)} and {period}", params + period_params)
            counts.append(pd.read_sql_query(f"select {select}, sum(n) as n {sql} group by {group}", conn, params=sql_params))
        db_df = pd.concat(counts).groupby(out, as_index=False).n.sum() if counts else pd.DataFrame(columns=out + ["n"])
        # Totaux
        if ngram_schema(db_path, f"total_{table}") is not None:
            sql, sql_params = ngram_from(conn, f"total_{table}", period, period_params)
            base = pd.read_sql_query(f"select {select}, sum(total) as total {sql} group by {group}", conn, params=sql_params)
        else:
            if condition or by:
                conn.close()
                return Response(f"pas de totaux par {', '.join(fields)} pour le corpus {corpus}", status=400)
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
        db_df = db_df.sort_values(out)
        for c in by:
            if fields[c] is not None:
                db_df[c] = db_df[c].map({id: label for label, id in fields[c].items()})
        db_df["gram"] = series_label
        results.append(db_df[["n"] + out + ["gram", "total"]])
    conn.close()
    print(corpus + " " + args.get("mot", ""))
    return pd.concat(results).to_csv(index=False)

# /contain : nombre de fenêtres de W mots contenant deux mots, sur les bases token {corpus}_ngram.db (et les
# bases de stage-mids avec elias=true). L'ancienne route lisait le texte des n-grammes avec un index FTS
# (4gram_lemonde.db, 3gram_presse.db, 3gram.db), dont celui de Le Monde a disparu.
# Les bases token ne s'interrogent vite qu'en fixant le premier mot (clé primaire w1, w2, ...), et elles sont
# élaguées des n-grammes rares ('guerre froide X' disparaît bien plus que 'la guerre froide') : on ne peut pas
# retrouver les fenêtres en ne regardant que celles qui commencent par l'un des mots. On compte donc les paires
# à chaque distance dans le plus petit n-gramme qui les contient (bigram pour des mots voisins, trigram pour un
# mot d'écart...), dans les deux ordres, et une paire à distance d compte pour les W - d fenêtres de W mots qui
# la contiennent : n est un nombre de W-grammes, rapporté au total des W-grammes, comme l'ancienne route.
# La fenêtre maximale est le plus long n-gramme de la base (2 à 5 selon le corpus).
# lemonde : la base lemonde_ngram.db s'arrête aux 3-grammes ; on prend lemonde_rubriques_ngram.db, qui couvre
# tout le journal (1944-2024) jusqu'aux 5-grammes, en sommant les rubriques.
contain_corpus_db = {"lemonde": "lemonde_rubriques"}
# Fenêtre par défaut de l'ancienne route : 4 sur Le Monde, 3 ailleurs (presse, livres) ; bornée par la base
contain_default_window = {"lemonde": 4, "lemonde_rubriques": 4}
# L'index FTS coupait aux apostrophes : 'europe' trouvait aussi l'europe, d'europe...
contain_elisions = ("l'", "d'", "qu'", "n'", "s'", "j'", "m'", "t'", "c'", "jusqu'", "lorsqu'", "puisqu'")

@app.route("/contain")
@app.route("/contain_ngram")
def contain():
    # mot1, mot2 : un mot chacun. from/to : AAAA, AAAAMM ou AAAAMMJJ ; resolution : annee, mois ou jour
    # (défaut : la plus fine disponible). window : taille de la fenêtre en mots, de 2 au plus long n-gramme
    # de la base (défaut 4 sur Le Monde, 3 ailleurs). count=False : une ligne par n-gramme trouvé
    # ('guerre _ froide') au lieu de la somme. Champs du corpus comme pour /query_ngram : rubrique=politique,culture
    # restreint comptes et totaux, by_rubrique=True sépare les courbes (sauf corpus=lemonde, qui somme tout).
    args = request.args
    corpus = args.get("corpus", "lemonde")
    if not re.fullmatch(r"[a-z0-9_]+", corpus):
        return Response("corpus invalide", status=400)
    elias = args.get("elias", "false").lower() in ("true", "1")
    bases = ngram_bases(args, corpus if elias else contain_corpus_db.get(corpus, corpus))
    if not bases:
        return Response(f"corpus inconnu : {corpus}", status=400)
    db_path = bases[0]
    mots = [args.get(k, "").strip().lower() for k in ("mot1", "mot2")]
    if not all(mots) or any(len(m.split()) != 1 for m in mots):
        return Response("mot1 et mot2 doivent contenir un mot chacun", status=400)
    fr, to = args.get("from", "1000"), args.get("to", "2100")
    if not (fr.isdigit() and to.isdigit() and len(fr) in (4, 6, 8) and len(to) in (4, 6, 8)):
        return Response("from/to doivent être au format AAAA, AAAAMM ou AAAAMMJJ", status=400)
    fr, to = fr.ljust(8, "0"), to + "9" * (8 - len(to))
    tables = [t for t in ngram_tables[1:] if ngram_schema(db_path, t) and ngram_schema(db_path, f"total_{t}")]
    max_window = len(tables) + 1 if tables == ngram_tables[1:len(tables) + 1] else 0
    if max_window < 2:
        return Response(f"pas de bigrammes pour le corpus {corpus}", status=400)
    window = args.get("window", str(min(contain_default_window.get(corpus, 3), max_window)))
    if not window.isdigit() or not 2 <= int(window) <= max_window:
        return Response(f"window doit être entre 2 et {max_window} pour le corpus {corpus}", status=400)
    window = int(window)
    table_w = ngram_tables[window - 1]
    steps, width = ngram_schema(db_path, table_w)
    resolution = args.get("resolution", "default")
    res = steps[-1] if resolution == "default" else resolution
    if res not in steps:
        return Response(f"resolution doit être parmi : {', '.join(steps)}", status=400)
    select, group, period, period_params = ngram_time_sql(steps, width, res, fr, to)
    fields = ngram_fields(db_path, table_w)
    by = []
    if corpus not in contain_corpus_db or elias:
        condition, field_params, error = ngram_fields_sql(corpus, db_path, table_w, args)
        if error:
            return Response(error, status=400)
        period += condition
        period_params += field_params
        by = [c for c in fields if args.get(f"by_{c}", "").lower() in ("true", "1")]
    out = group.split(", ") + by
    select = ", ".join([select] + by)
    group = ", ".join(out)
    count = args.get("count", "True").lower() not in ("false", "0")
    conn = ngram_connect(bases)
    ids = []
    for m in mots:
        forms = [m] + ([e + m for e in contain_elisions] if m.startswith(ngram_elision) else [])
        ids.append(dict(conn.execute(f"select id, word from token where word in ({','.join('?' * len(forms))})", forms)))
    counts = []
    for first, second in ((ids[0], ids[1]), (ids[1], ids[0])):
        if not first or not second:
            continue
        for k in range(2, window + 1):
            # Mots à distance k - 1 : k-gramme w1 ... wk, compté dans window - (k - 1) fenêtres
            table, weight = ngram_tables[k - 1], window - k + 1
            words = f"w1 in ({','.join(map(str, first))}) and w{k} in ({','.join(map(str, second))})"
            sql, sql_params = ngram_from(conn, table, f"{words} and {period}", list(period_params))
            df = pd.read_sql_query(f"select {select}, w1, w{k} as wk, sum(n) * {weight} as n {sql} group by {group}, w1, w{k}", conn, params=sql_params)
            df["gram"] = df.w1.map(first) + " " + "_ " * (k - 2) + df.wk.map(second)
            counts.append(df.drop(columns=["w1", "wk"]))
    sql, sql_params = ngram_from(conn, f"total_{table_w}", period, period_params)
    base = pd.read_sql_query(f"select {select}, sum(total) as total {sql} group by {group}", conn, params=sql_params)
    conn.close()
    db_df = pd.concat(counts) if counts else pd.DataFrame(columns=out + ["gram", "n"])
    if count:
        db_df = db_df.groupby(out, as_index=False).n.sum()
        db_df = pd.merge(db_df, base, how="right", on=out)
        db_df["gram"] = mots[0] + "&" + mots[1]
    else:
        db_df = db_df.groupby(out + ["gram"], as_index=False).n.sum()
        db_df = pd.merge(db_df, base, how="left", on=out)
    db_df.n = db_df.n.fillna(0).astype(int)
    db_df = db_df.sort_values(out)
    for c in by:
        if fields[c] is not None:
            db_df[c] = db_df[c].map({id: label for label, id in fields[c].items()})
    print(corpus + " " + mots[0] + "&" + mots[1])
    return db_df[["n"] + out + ["total", "gram"]].to_csv(index=False)

def ngram_stoplist(args):
    # Les `stopwords` mots les plus fréquents (liste de /associated), ou None si le paramètre est invalide
    stopwords = args.get("stopwords", "0")
    if not stopwords.isdigit():
        return None
    return list(pd.read_csv("/opt/bazoulay/docker_gallicagram/gallicagram/stopwords.csv").monogram.iloc[:int(stopwords)])

ngram_score_methods = ("count", "llr", "pmi", "logdice")
# Les fréquences propres se lisent dans la table unigram (~1,4 ms par mot sur presse 1789-1950) : on ne note que
# les mots les plus fréquents du motif, max(1000, 20 * n_joker). Sur guerre _ (presse 1789-1950), les 50 premiers
# en logDice sont tous dans les 120 premiers en compte brut ; seul pmi, qui favorise les mots rares, peut en pâtir.
ngram_score_candidates = 1000

def ngram_score_args(args):
    # score et min_count (défaut 20 : les scores d'association sont instables sur les mots rares), ou message d'erreur
    score = args.get("score", "count")
    if score not in ngram_score_methods:
        return None, None, f"score doit être parmi : {', '.join(ngram_score_methods)}"
    min_count = args.get("min_count", "20")
    if not min_count.isdigit():
        return None, None, "min_count doit être un entier"
    return score, int(min_count), None

def ngram_score(conn, df, r1, fx, score, period, period_params, n_joker):
    # Ajoute à df (id, tot) la fréquence propre du mot (freq, table unigram) et le score d'association.
    # Table de contingence : O11 = tot (cooccurrences), R1 = r1 (positions disponibles autour du mot cherché),
    # C1 = freq, N = total des unigrammes sur la période ; fx = fréquence du mot cherché (logDice).
    # llr : log-vraisemblance de Dunning (1993), signée (négative si sous-représenté) ;
    # pmi : log2(O11 / E11) ; logdice : 14 + log2(2 * O11 / (fx + freq)) (Rychlý 2008), max 14.
    if n_joker != "all":
        df = df.nlargest(max(ngram_score_candidates, 20 * int(n_joker)), "tot")
    sql, sql_params = ngram_from(conn, "unigram", f"w1 in ({','.join(map(str, df.id))}) and {period}", period_params)
    freq = pd.read_sql_query(f"select w1 as id, sum(n) as freq {sql} group by w1", conn, params=sql_params)
    df = df.merge(freq, on="id", how="left").fillna({"freq": 0})
    sql, sql_params = ngram_from(conn, "total_unigram", period, period_params)
    N = conn.execute(f"select sum(total) {sql}", sql_params).fetchone()[0] or 0
    o11, c1 = df.tot.astype(float).values, np.maximum(df.freq.astype(float).values, df.tot.values)
    e11 = r1 * c1 / N
    if score == "pmi":
        df["score"] = np.log2(o11 / e11)
    elif score == "logdice":
        df["score"] = 14 + np.log2(2 * o11 / (fx + c1))
    else:
        o = [o11, r1 - o11, c1 - o11, N - r1 - c1 + o11]
        e = [e11, r1 * (N - c1) / N, (N - r1) * c1 / N, (N - r1) * (N - c1) / N]
        with np.errstate(divide="ignore", invalid="ignore"):
            llr = 2 * sum(np.where(oi > 0, oi * np.log(np.maximum(oi, 1e-300) / ei), 0) for oi, ei in zip(o, e))
        df["score"] = np.sign(o11 - e11) * llr
    df["score"] = df.score.round(3)
    df["freq"] = df.freq.astype(int)
    return df.sort_values("score", ascending=False)

def ngram_pattern(args, words):
    # Classement des n-grammes correspondant à un motif sur une base token {corpus}_ngram.db.
    # Chaque mot est exact, '_' (n'importe quel mot) ou un motif glob avec * et ? ('addict*').
    # stopwords=k écarte les k mots les plus fréquents aux positions '_' ; un joker (d*) fait partie de la requête et n'est pas filtré.
    # rubrique=..., revue=... : filtres sur les champs du corpus, comme pour /query_ngram.
    # Renvoie un CSV tot,gram trié par fréquence décroissante sur la période, ou une Response d'erreur.
    # score=llr|pmi|logdice (un seul '_', mots exacts) : classement par association entre le mot en '_' et le reste
    # du motif, parmi les formes vues au moins min_count fois ; renvoie alors tot,gram,freq,score.
    corpus = args.get("corpus", "")
    if not re.fullmatch(r"[a-z0-9_]+", corpus):
        return Response("corpus invalide", status=400)
    bases = ngram_bases(args, corpus)
    if not bases:
        return Response(f"corpus inconnu : {corpus}", status=400)
    db_path = bases[0]  # schéma, champs et taille : ceux de la grosse base
    n = len(words)
    if not 0 < n <= len(ngram_tables):
        return Response(f"le motif doit contenir entre 1 et {len(ngram_tables)} mots", status=400)
    fr = args.get("from", "1000")
    to = args.get("to", "2100")
    if not (fr.isdigit() and to.isdigit() and len(fr) in (4, 6, 8) and len(to) in (4, 6, 8)):
        return Response("from/to doivent être au format AAAA, AAAAMM ou AAAAMMJJ", status=400)
    fr = fr.ljust(8, "0")
    to = to + "9" * (8 - len(to))
    n_joker = args.get("n_joker", "50")
    if n_joker != "all" and not n_joker.isdigit():
        return Response("n_joker doit être un entier ou 'all'", status=400)
    stoplist = ngram_stoplist(args)
    if stoplist is None:
        return Response("stopwords doit être un entier", status=400)
    score, min_count, error = ngram_score_args(args)
    if error:
        return Response(error, status=400)
    if score != "count" and (words.count("_") != 1 or any("*" in w or "?" in w for w in words)):
        return Response("score n'est disponible qu'avec un seul '_' et des mots exacts", status=400)
    table = ngram_tables[n - 1]
    schema = ngram_schema(db_path, table)
    if schema is None:
        return Response(f"pas de table {table} pour le corpus {corpus}", status=400)
    steps, width = schema
    _, _, period, period_params = ngram_time_sql(steps, width, steps[0], fr, to)
    condition, field_params, error = ngram_fields_sql(corpus, db_path, table, args)
    if error:
        return Response(error, status=400)
    period += condition
    period_params += field_params
    conn = ngram_connect(bases)
    conditions, params, error = ngram_words_sql(conn, db_path, words)
    if error:
        conn.close()
        return Response(error, status=400)
    stop_ids = [r[0] for r in conn.execute(f"select id from token where word in ({','.join('?' * len(stoplist))})", stoplist)] if stoplist else []
    stop_conditions = [f"w{i+1} not in ({','.join(map(str, stop_ids))})" for i, w in enumerate(words) if w == "_"] if stop_ids else []
    if score != "count":
        j = words.index("_") + 1
        # Une seule lecture des n-grammes du motif : r1 = fx = nombre d'occurrences du motif,
        # toutes formes en '_' comprises (mots vides inclus) ; mots vides et min_count filtrés ensuite
        sql, sql_params = ngram_from(conn, table, f"{' and '.join(conditions)} and {period}", params + period_params)
        db_df = pd.read_sql_query(f"select w{j} as id, sum(n) as tot {sql} group by w{j}", conn, params=sql_params)
        r1 = db_df.tot.sum()
        db_df = db_df[(db_df.tot >= min_count) & ~db_df.id.isin(stop_ids)]
        if len(db_df):
            db_df = ngram_score(conn, db_df, r1, r1, score, period, period_params, n_joker)
            if n_joker != "all":
                db_df = db_df.iloc[:int(n_joker)]
            tokens = pd.read_sql_query(f"select id, word from token where id in ({','.join(map(str, db_df.id))})", conn)
            db_df = db_df.merge(tokens, on="id", sort=False)
            db_df["gram"] = [" ".join(t if w == "_" else w for w in words) for t in db_df.word]
        conn.close()
        print(corpus + " " + " ".join(words) + " " + score)
        return (db_df if len(db_df) else pd.DataFrame(columns=["tot", "gram", "freq", "score"]))[["tot", "gram", "freq", "score"]].to_csv(index=False)
    conditions += stop_conditions
    w_cols = ", ".join(f"w{i+1}" for i in range(n))
    # Mots résolus par sous-requête plutôt que par jointure : avec des tampons, token est une vue UNION ALL
    # qu'une jointure par position démultiplie (2^n branches recalculant chacune g)
    gram = " || ' ' || ".join(f"(select word from token where id=g.w{i+1})" for i in range(n))
    limit = "" if n_joker == "all" else f"limit {int(n_joker)}"
    sql, sql_params = ngram_from(conn, table, f"{' and '.join(conditions)} and {period}", params + period_params)
    db_df = pd.read_sql_query(
        f"select g.tot, {gram} as gram from (select {w_cols}, sum(n) as tot {sql} "
        f"group by {w_cols} order by tot desc {limit}) g order by g.tot desc",
        conn, params=sql_params)
    conn.close()
    print(corpus + " " + " ".join(words))
    return db_df.to_csv(index=False)

@app.route("/joker_ngram")
def joker_ngram():
    # Équivalent de /joker pour les bases token : mots les plus fréquents après (after=True, défaut)
    # ou avant (after=False) `mot`. length : taille du n-gramme cherché (défaut : nombre de mots + 1).
    # '_' peut aussi être placé explicitement dans mot ('la _ de'). Paramètres : corpus, mot, from, to, n_joker, stopwords,
    # score (count, llr, pmi ou logdice ; voir ngram_score) et min_count (défaut 20).
    args = request.args
    words = args.get("mot", "").lower().split()
    if "_" not in words:
        length = args.get("length", str(len(words) + 1))
        if not length.isdigit() or int(length) <= len(words):
            return Response("length doit être un entier supérieur au nombre de mots", status=400)
        jokers = ["_"] * (int(length) - len(words))
        words = words + jokers if args.get("after", "True") == "True" else jokers + words
    return ngram_pattern(args, words)

@app.route("/wildcard_ngram")
def wildcard_ngram():
    # Formes correspondant à un motif avec jokers glob : * (0 ou plusieurs caractères), ? (un caractère).
    # Ex. mot=addict* ; mot=addict* aux ; mot=*isme (corpus de petite taille uniquement).
    # Paramètres : corpus, mot, from, to, n_joker, stopwords. Renvoie tot,gram.
    args = request.args
    return ngram_pattern(args, args.get("mot", "").lower().split())

@app.route("/associated_ngram")
def associated_ngram():
    # Équivalent de /associated pour les bases token : mots les plus fréquents dans une fenêtre
    # de length - (nombre de mots de mot) positions après et/ou avant mot. Renvoie gram,tot.
    # side : after, before ou both (défaut : both, ou after sur les grosses bases où 'before' impose un scan complet).
    # Paramètres : corpus, mot, from, to, length (défaut : nombre de mots + 1), n_joker, stopwords,
    # score (count, llr, pmi ou logdice ; voir ngram_score) et min_count (défaut 20, pour les scores) ;
    # avec un score, renvoie gram,tot,freq,score trié par score.
    args = request.args
    corpus = args.get("corpus", "")
    if not re.fullmatch(r"[a-z0-9_]+", corpus):
        return Response("corpus invalide", status=400)
    bases = ngram_bases(args, corpus)
    if not bases:
        return Response(f"corpus inconnu : {corpus}", status=400)
    db_path = bases[0]  # schéma, champs et taille : ceux de la grosse base
    fr = args.get("from", "1000")
    to = args.get("to", "2100")
    if not (fr.isdigit() and to.isdigit() and len(fr) in (4, 6, 8) and len(to) in (4, 6, 8)):
        return Response("from/to doivent être au format AAAA, AAAAMM ou AAAAMMJJ", status=400)
    fr = fr.ljust(8, "0")
    to = to + "9" * (8 - len(to))
    words = args.get("mot", "").lower().split()
    k = len(words)
    length = args.get("length", str(k + 1))
    if not k or not length.isdigit() or not k < int(length) <= len(ngram_tables):
        return Response(f"mot doit contenir au moins un mot et length être compris entre {k + 1} et {len(ngram_tables)}", status=400)
    length = int(length)
    n_joker = args.get("n_joker", "50")
    if n_joker != "all" and not n_joker.isdigit():
        return Response("n_joker doit être un entier ou 'all'", status=400)
    stoplist = ngram_stoplist(args)
    if stoplist is None:
        return Response("stopwords doit être un entier", status=400)
    score, min_count, error = ngram_score_args(args)
    if error:
        return Response(error, status=400)
    big = os.path.getsize(db_path) > ngram_scan_limit
    side = args.get("side", "after" if big else "both")
    if side not in ("after", "before", "both"):
        return Response("side doit être after, before ou both", status=400)
    # Sans premier mot fixé, la clé primaire (w1, w2, ..., temps) ne sert pas : scan complet
    if big and side != "after":
        return Response(f"sur le corpus {corpus}, seul side=after est disponible", status=400)
    table = ngram_tables[length - 1]
    schema = ngram_schema(db_path, table)
    if schema is None:
        return Response(f"pas de table {table} pour le corpus {corpus}", status=400)
    steps, width = schema
    _, _, period, period_params = ngram_time_sql(steps, width, steps[0], fr, to)
    condition, field_params, error = ngram_fields_sql(corpus, db_path, table, args)
    if error:
        return Response(error, status=400)
    period += condition
    period_params += field_params
    conn = ngram_connect(bases)
    ids = [conn.execute("select id from token where word=?", (w,)).fetchone() for w in words]
    counts = []
    if all(ids):
        ids = [i[0] for i in ids]
        # after : mot occupe les positions 1..k ; before : les positions length-k+1..length
        for offset in {"after": [0], "before": [length - k], "both": [0, length - k]}[side]:
            fixed = range(offset + 1, offset + k + 1)
            free = [f"w{j}" for j in range(1, length + 1) if j not in fixed]
            w_condition = " and ".join(f"w{j}=?" for j in fixed)
            sql, sql_params = ngram_from(conn, table, f"{w_condition} and {period}", ids + period_params)
            df = pd.read_sql_query(f"select {', '.join(free)}, sum(n) as tot {sql} group by {', '.join(free)}", conn, params=sql_params)
            # Chaque mot de la fenêtre reçoit le compte du n-gramme
            counts += [df[[w, "tot"]].rename(columns={w: "id"}) for w in free]
    limit = None if n_joker == "all" else int(n_joker)
    kept = []
    if counts:
        db_df = pd.concat(counts).groupby("id", as_index=False).tot.sum()
        # r1 : nombre de positions de la fenêtre autour des occurrences de mot, avant tout filtre
        r1 = db_df.tot.sum()
        db_df = db_df[~db_df.id.isin(ids)].sort_values("tot", ascending=False)
        if score != "count":
            sql, sql_params = ngram_from(conn, ngram_tables[k - 1], f"{' and '.join(f'w{j+1}=?' for j in range(k))} and {period}", ids + period_params)
            fx = conn.execute(f"select sum(n) {sql}", sql_params).fetchone()[0] or 0
            db_df = db_df[db_df.tot >= min_count]
            if len(db_df):
                db_df = ngram_score(conn, db_df, r1, fx, score, period, period_params, n_joker)
        # Les mots ne sont résolus que par paquets, dans l'ordre, jusqu'à avoir n_joker mots retenus :
        # sur presse, résoudre tous les ids de la fenêtre dans la table token coûte plusieurs secondes
        step = 30000 if limit is None else max(2 * (limit + len(stoplist)), 1000)
        for start in range(0, len(db_df), step):
            chunk = db_df.iloc[start:start + step]
            tokens = pd.read_sql_query(f"select id, word as gram from token where id in ({','.join(map(str, chunk.id))})", conn)
            chunk = chunk.merge(tokens, on="id")[["gram", "tot"] + (["freq", "score"] if score != "count" else [])]
            # Sans lettre (ponctuation, nombres) ou mot vide : écarté
            chunk = chunk[chunk.gram.str.contains(r"[^\W\d_]") & ~chunk.gram.isin(stoplist)]
            kept.append(chunk)
            if limit is not None and sum(map(len, kept)) >= limit:
                break
    conn.close()
    key = "tot" if score == "count" else "score"
    db_df = pd.concat(kept).sort_values(key, ascending=False) if kept else pd.DataFrame(columns=["gram", "tot"] + (["freq", "score"] if score != "count" else []))
    if limit is not None:
        db_df = db_df.iloc[:limit]
    print("associated " + corpus + " " + " ".join(words))
    return db_df.to_csv(index=False)

@app.route("/mots_du_jour_ngram")
def mots_du_jour_ngram():
    # Mots du jour de la presse web contemporaine (bases elias de stage-mids) : mots en excès sur une période
    # par rapport aux `ref` mois qui précèdent son mois de début, groupés en sujets. Méthode : voir mots_du_jour.py.
    # corpus : médias séparés par ',' (défaut : tous ceux de version.json) ; from/to : AAAAMMJJ
    # (défaut : les 7 derniers jours complets) ; ref : mois de référence (défaut 12) ; min_titres : nombre de
    # titres où le mot doit être en excès (défaut 5, ramené au nombre de titres disponibles) ; z_min : seuil
    # d'excès dans un titre (défaut 5) ; min_n : occurrences minimales sur la période (défaut 10) ;
    # ratio_min : rapport minimal entre taux sur la période et taux de référence (défaut 1.5) ;
    # top : mots retenus (défaut 80) ; seuil : corrélation minimale pour grouper deux sujets (défaut 0.4) ;
    # tendance : séries comparées telles quelles (niveau), en écarts d'un jour à l'autre (diff) ou en écart
    # à la moyenne mobile sur 7 jours (passe_haut, défaut) ; seuil_articles : avec moins de 3 titres réguliers,
    # Le Monde jusqu'à 2022 est regroupé par articles communs (champ regroupement='articles'), seuil propre (défaut 0.3).
    # network=true : ajoute noeuds (mot, groupe = rang dans groupes, score, n) et liens (source, cible, poids, type :
    # articles, grille ou bigramme) pour un affichage en réseau.
    # Renvoie du JSON : groupes (étiquettes tirées des bigrammes, mots avec score, titres en excès, comptes).
    args = request.args
    corpora = [c for c in args.get("corpus", "").split(",") if c]
    if not all(re.fullmatch(r"[a-z0-9_]+", c) for c in corpora):
        return Response("corpus invalide", status=400)
    hier = pd.Timestamp.today().normalize() - pd.Timedelta(days=1)
    fr = args.get("from", (hier - pd.Timedelta(days=6)).strftime("%Y%m%d"))
    to = args.get("to", hier.strftime("%Y%m%d"))
    if not (re.fullmatch(r"\d{8}", fr) and re.fullmatch(r"\d{8}", to)):
        return Response("from/to doivent être au format AAAAMMJJ", status=400)
    try:
        params = dict(ref=int(args.get("ref", 12)), min_titres=int(args.get("min_titres", 5)),
                      z_min=float(args.get("z_min", 5)), min_n=int(args.get("min_n", 10)),
                      ratio_min=float(args.get("ratio_min", 1.5)),
                      top=int(args.get("top", 80)), seuil=float(args.get("seuil", 0.4)),
                      tendance=args.get("tendance", "passe_haut"),
                      seuil_articles=float(args.get("seuil_articles", 0.3)),
                      network=args.get("network", "false").lower() in ("true", "1"))
    except ValueError:
        return Response("ref, min_titres, min_n et top doivent être des entiers ; z_min, ratio_min, seuil et seuil_articles des nombres", status=400)
    if not (1 <= params["ref"] <= 120 and params["min_titres"] >= 1 and 1 <= params["top"] <= 300):
        return Response("ref doit être entre 1 et 120, min_titres >= 1, top entre 1 et 300", status=400)
    if params["tendance"] not in ("niveau", "diff", "passe_haut"):
        return Response("tendance doit être niveau, diff ou passe_haut", status=400)
    try:
        resultat = calcul_mots_du_jour(corpora or None, fr, to, **params)
    except ValueError as e:
        return Response(str(e), status=400)
    print("mots_du_jour " + ",".join(corpora or ["tous"]) + f" {fr}-{to} {resultat['duree_s']}s")
    return jsonify(resultat)

portraits_db = "/opt/bazoulay/portraits.sqlite"
portraits_filtres = ("newspaper", "section")

def portraits_cle(nom):
    # Clé de la table names : minuscules, sans accents, tout ce qui n'est ni lettre ni chiffre devient une espace
    # ('Jean-Luc Mélenchon' -> 'jean luc melenchon', "A'Salfo" -> 'a salfo').
    nom = unicodedata.normalize("NFKD", nom).encode("ascii", "ignore").decode().lower()
    return " ".join(re.sub(r"[^a-z0-9]+", " ", nom).split())

@app.route("/portraits")
def portraits():
    # Portraits de personnes dans la presse (portraits.sqlite) : une ligne par mois, journal et rubrique,
    # avec tous les champs de la table cells tels quels (comptes, sommes W, S_*, SS_*, WS_*...), sans agrégation.
    # nom : une ou plusieurs personnes séparées par ',' (accents, casse et tirets indifférents) ;
    # newspaper, section : filtres optionnels, valeurs séparées par ',' (ex. newspaper=lemonde,liberation) ;
    # from/to : AAAA ou AAAAMM, bornes incluses.
    # Renvoie du CSV : nom (forme affichée), puis les colonnes de cells.
    args = request.args
    noms = [n for n in args.get("nom", args.get("name", "")).split(",") if n.strip()]
    if not noms:
        return Response("nom requis", status=400)
    cles = list(dict.fromkeys(portraits_cle(n) for n in noms))
    where, params = [f"c.key in ({', '.join('?' * len(cles))})"], cles
    for f in portraits_filtres:
        valeurs = [v.strip() for v in args.get(f, "").split(",") if v.strip()]
        if valeurs:
            where.append(f"c.{f} in ({', '.join('?' * len(valeurs))})")
            params += valeurs
    for borne, op, defaut in (("from", ">=", "01"), ("to", "<=", "12")):
        d = args.get(borne)
        if d is None:
            continue
        if not re.fullmatch(r"\d{4}(\d{2})?", d):
            return Response("from/to doivent être au format AAAA ou AAAAMM", status=400)
        where.append(f"c.month {op} ?")
        params.append(f"{d[:4]}-{d[4:6] or defaut}")
    conn = sqlite3.connect(f"file:{portraits_db}?mode=ro", uri=True)
    db_df = pd.read_sql_query(
        "select n.name as nom, c.* from cells c join names n on n.key = c.key where "
        + " and ".join(where) + " order by c.key, c.month, c.newspaper, c.section", conn, params=params)
    conn.close()
    return db_df.to_csv(index=False)

@app.route("/names_portraits")
def names_portraits():
    # Personnes de portraits.sqlite citées dans au moins min_articles articles (défaut 100).
    # Renvoie du CSV : name (forme à passer à /portraits), key, n_articles, n_mentions, first_month, last_month,
    # par n_articles décroissant.
    try:
        seuil = int(request.args.get("min_articles", 100))
    except ValueError:
        return Response("min_articles doit être un entier", status=400)
    conn = sqlite3.connect(f"file:{portraits_db}?mode=ro", uri=True)
    db_df = pd.read_sql_query("select name, key, n_articles, n_mentions, first_month, last_month from names "
                              "where n_articles >= ? order by n_articles desc", conn, params=[seuil])
    conn.close()
    return db_df.to_csv(index=False)

# Scores de cells et suffixe de leurs poids : valence_nm a ses propres poids (W_nm, WW_nm), les autres W et WW
portraits_scores = {"valence": "", "extremisme": "", "dynamique": "", "journaliste": "", "valence_nm": "_nm"}
portraits_poids = sorted(set(portraits_scores.values()))

@lru_cache(maxsize=4)
def portraits_tables(mtime):
    # Tables présentes dans portraits.sqlite (cells_year et totals_month viennent de portraits_precompute.py)
    conn = sqlite3.connect(f"file:{portraits_db}?mode=ro", uri=True)
    tables = {r[0] for r in conn.execute("select name from sqlite_master where type = 'table'")}
    conn.close()
    return tables

def portraits_par_annee(filtres):
    # Vrai si les bornes tombent sur des années entières : la table cells_year suffit alors
    fr, to = filtres[2:]
    return (fr is None or fr.endswith("-01")) and (to is None or to.endswith("-12")) \
        and "cells_year" in portraits_tables(os.path.getmtime(portraits_db))

def portraits_filtres_args(args):
    # Filtres communs à /rankings_portraits et /baseline : (newspaper, section, from, to) en tuples hachables
    # (pour les caches), ou un message d'erreur. from/to : AAAA ou AAAAMM -> 'AAAA-MM'.
    filtres = []
    for f, v in (("newspaper", args.get("newspaper", args.get("newspapers", ""))), ("section", args.get("section", ""))):
        filtres.append(tuple(sorted({x.strip() for x in v.split(",") if x.strip()})))
    for borne, defaut in (("from", "01"), ("to", "12")):
        d = args.get(borne)
        if d is not None and not re.fullmatch(r"\d{4}(\d{2})?", d):
            return "from/to doivent être au format AAAA ou AAAAMM"
        filtres.append(f"{d[:4]}-{d[4:6] or defaut}" if d else None)
    return tuple(filtres)

def portraits_filtres_sql(filtres, annee=False):
    # Conditions SQL (alias c pour cells, ou cells_year si annee) et paramètres correspondant aux filtres.
    journaux, sections, fr, to = filtres
    where, params = [], []
    for col, valeurs in (("newspaper", journaux), ("section", sections)):
        if valeurs:
            where.append(f"c.{col} in ({', '.join('?' * len(valeurs))})")
            params += valeurs
    for op, d in ((">=", fr), ("<=", to)):
        if d:
            where.append(f"c.{'year' if annee else 'month'} {op} ?")
            params.append(d[:4] if annee else d)
    return where, params

@lru_cache(maxsize=4)
def portraits_totaux_mois(mtime):
    # Sommes de tout le monde par mois × journal × rubrique (poids et S de chaque score), mises en cache tant
    # que le fichier n'est pas modifié. Base de /baseline, des scores attendus et de /newspapers_portraits.
    # Lues dans totals_month (portraits_precompute.py), sinon calculées sur cells (parcours complet, ~10 s).
    sommes = ", ".join([f"sum(W{w}) as W{w}" for w in portraits_poids] + [f"sum(S_{s}) as S_{s}" for s in portraits_scores])
    source = "totals_month" if "totals_month" in portraits_tables(mtime) else "cells"
    conn = sqlite3.connect(f"file:{portraits_db}?mode=ro", uri=True)
    df = pd.read_sql_query(f"select month, newspaper, section, sum(n_articles) as n_articles, {sommes} "
                           f"from {source} group by month, newspaper, section", conn)
    conn.close()
    return df

def portraits_totaux_annee(filtres):
    # Sommes de tout le monde par année × journal × rubrique, restreintes aux filtres (mois compris).
    journaux, sections, fr, to = filtres
    df = portraits_totaux_mois(os.path.getmtime(portraits_db))
    garde = pd.Series(True, index=df.index)
    if journaux:
        garde &= df["newspaper"].isin(journaux)
    if sections:
        garde &= df["section"].isin(sections)
    if fr:
        garde &= df["month"] >= fr
    if to:
        garde &= df["month"] <= to
    df = df[garde].assign(year=df["month"].str[:4])
    return df.drop(columns="month").groupby(["year", "newspaper", "section"], as_index=False).sum()

@app.route("/baseline")
def baseline():
    # Sommes de toutes les personnes de portraits.sqlite par année × journal × rubrique : le score moyen d'une
    # cellule est S_{score} / W. Filtres facultatifs : newspaper, section (séparés par ','), from/to (AAAA ou AAAAMM).
    # Renvoie du CSV : year, newspaper, section, n_articles, W, W_nm, puis S_{score} pour chaque score
    # (valence, extremisme, dynamique, journaliste, valence_nm ; valence_nm se rapporte à W_nm).
    filtres = portraits_filtres_args(request.args)
    if isinstance(filtres, str):
        return Response(filtres, status=400)
    return portraits_totaux_annee(filtres).to_csv(index=False)

@app.route("/rankings_portraits")
def rankings_portraits():
    # Classement des personnes sur une période, tous journaux confondus (ou filtrés).
    # Dans cells, chaque article a compte pour w_a (somme des poids de ses segments) et s_a (somme des scores
    # pondérés) : W = Σw_a, S = Σs_a, WW = Σw_a², SS = Σs_a², WS = Σw_a·s_a. Le score moyen d'une personne est
    # le rapport R = S / W (score moyen par segment, pondéré) ; son écart-type est celui d'un estimateur par
    # le ratio avec l'article comme grappe : se² = n/(n-1) · (SS - 2R·WS + R²·WW) / W², n = nombre d'articles.
    # {score}_expected : score qu'aurait la personne si elle avait, dans chaque cellule année × journal × rubrique
    # où elle apparaît (dans les filtres), le score moyen des autres personnes de la cellule :
    # Σ W_p · (S_all − S_p) / (W_all − W_p) ÷ Σ W_p (cellules où la personne est seule exclues).
    # {score}_mean − {score}_expected est donc son score relatif aux journaux, rubriques et années où on parle d'elle.
    # from/to : AAAA ou AAAAMM (bornes incluses) ; min : articles minimum sur la période et le périmètre (défaut 100) ;
    # newspaper (ou newspapers), section : filtres séparés par ',' ; score : tri (valence par défaut,
    # extremisme, dynamique, journaliste, valence_nm) ; order : desc (défaut) ou asc.
    # valence_nm a ses propres poids (W_nm, WW_nm), qui remplacent W et WW dans les formules ci-dessus.
    # Avec des bornes en années entières, les sommes viennent de cells_year (rapide), sinon de cells.
    # Renvoie du CSV : name, key, n_articles, n_mentions, n_segments, puis pour chaque score
    # {score}_mean, {score}_se, {score}_low, {score}_high (intervalle à 95 %), {score}_expected, trié par {score}_mean.
    args = request.args
    try:
        seuil = int(args.get("min", 100))
    except ValueError:
        return Response("min doit être un entier", status=400)
    score = args.get("score", "valence")
    if score not in portraits_scores:
        return Response(f"score doit être parmi : {', '.join(portraits_scores)}", status=400)
    order = args.get("order", "desc")
    if order not in ("asc", "desc"):
        return Response("order doit être asc ou desc", status=400)
    filtres = portraits_filtres_args(args)
    if isinstance(filtres, str):
        return Response(filtres, status=400)
    out = portraits_classement(filtres, seuil, os.path.getmtime(portraits_db))
    out = out.sort_values(f"{score}_mean", ascending=(order == "asc"))
    return out.to_csv(index=False)

@lru_cache(maxsize=64)
def portraits_classement(filtres, seuil, mtime):
    # Agrégation par personne pour /rankings_portraits, mise en cache par jeu de filtres tant que le fichier
    # n'est pas modifié (mtime fait partie de la clé). Pour chaque score s, df contient W, WW, S, SS, WS et
    # E_s, EW_s (sommes des contributions au score attendu : attendu = E_s / EW_s).
    annee = portraits_par_annee(filtres)
    where, params = portraits_filtres_sql(filtres, annee)
    # Le nombre total d'articles d'une personne (names) borne celui de la période : on ne parcourt que ses cellules
    where, params = ["c.key in (select key from names where n_articles >= ?)"] + where, [seuil] + params
    comptes = ["n_articles", "n_mentions", "n_segments"] + [f"{x}{w}" for w in portraits_poids for x in ("W", "WW")] \
        + [f"{x}_{s}" for s in portraits_scores for x in ("S", "SS", "WS")]
    conn = sqlite3.connect(f"file:{portraits_db}?mode=ro", uri=True)
    if annee:
        # Bornes en années entières : cells_year a déjà, par cellule année × journal × rubrique, les contributions
        # au score attendu (portraits_precompute.py), qui ne dépendent pas des filtres. Tout se somme en SQL.
        cols = comptes + [f"{x}_{s}" for s in portraits_scores for x in ("E", "EW")]
        df = pd.read_sql_query(f"select c.key, {', '.join(f'sum({x}) as {x}' for x in cols)} from cells_year c "
                               f"where {' and '.join(where)} group by c.key having sum(n_articles) >= ?",
                               conn, params=params + [seuil]).set_index("key")
    else:
        # Bornes au mois : sommes par personne × année × journal × rubrique sur cells, comparées aux totaux des
        # mêmes cellules restreints aux filtres (portraits_totaux_annee), puis agrégées par personne
        cel = pd.read_sql_query(
            f"select c.key, substr(c.month, 1, 4) as year, c.newspaper, c.section, "
            f"{', '.join(f'sum({x}) as {x}' for x in comptes)} from cells c where {' and '.join(where)} "
            f"group by c.key, year, c.newspaper, c.section", conn, params=params)
        cel = cel.merge(portraits_totaux_annee(filtres), on=["year", "newspaper", "section"], suffixes=("", "_all"))
        for s, w in portraits_scores.items():
            autres_W = cel[f"W{w}_all"] - cel[f"W{w}"]
            pas_seul = autres_W > 1e-9 * cel[f"W{w}_all"].abs().clip(lower=1)  # la personne n'est pas seule
            cel[f"E_{s}"] = (cel[f"W{w}"] * (cel[f"S_{s}_all"] - cel[f"S_{s}"]) / autres_W).where(pas_seul, 0)
            cel[f"EW_{s}"] = cel[f"W{w}"].where(pas_seul, 0)
        df = cel.groupby("key")[comptes + [f"{x}_{s}" for s in portraits_scores for x in ("E", "EW")]].sum()
    noms = pd.read_sql_query("select name, key from names where n_articles >= ?", conn, params=[seuil])
    conn.close()
    df = df[(df["n_articles"] >= seuil) & (df["W"] > 0)]
    out = noms[noms["key"].isin(df.index)].reset_index(drop=True)
    df = df.loc[out["key"]]
    out = out.join(df[["n_articles", "n_mentions", "n_segments"]].reset_index(drop=True))
    n = df["n_articles"].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):  # personne sans poids nm : scores nm vides
        for s, w in portraits_scores.items():
            W, WW = df[f"W{w}"].to_numpy(), df[f"WW{w}"].to_numpy()
            r = df[f"S_{s}"].to_numpy() / W
            var = (df[f"SS_{s}"].to_numpy() - 2 * r * df[f"WS_{s}"].to_numpy() + r ** 2 * WW) / W ** 2
            se = np.sqrt(np.clip(var, 0, None) * n / np.maximum(n - 1, 1))
            out[f"{s}_mean"], out[f"{s}_se"], out[f"{s}_low"], out[f"{s}_high"] = r, se, r - 1.96 * se, r + 1.96 * se
            out[f"{s}_expected"] = df[f"E_{s}"].to_numpy() / df[f"EW_{s}"].to_numpy()
    return out

@lru_cache(maxsize=4)
def portraits_journaux(mtime):
    # Couples (journal, rubrique) présents dans cells, tirés des totaux mensuels (mis en cache)
    return portraits_totaux_mois(mtime)[["newspaper", "section"]].drop_duplicates()

@app.route("/newspapers_portraits")
def newspapers_portraits():
    # Journaux de portraits.sqlite. section : ne garde que les journaux ayant au moins une de ces rubriques
    # (séparées par ',', ex. section=sport).
    # Renvoie du CSV : newspaper, sections (toutes les rubriques du journal, séparées par ',').
    db_df = portraits_journaux(os.path.getmtime(portraits_db))
    sections = [s.strip() for s in request.args.get("section", "").split(",") if s.strip()]
    journaux = db_df.groupby("newspaper")["section"].apply(lambda s: ",".join(sorted(s))).reset_index(name="sections")
    if sections:
        journaux = journaux[journaux["newspaper"].isin(db_df.loc[db_df["section"].isin(sections), "newspaper"])]
    return journaux.to_csv(index=False)

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