# Tables précalculées de portraits.sqlite pour les routes /rankings_portraits et /baseline de app.py.
# À relancer après chaque reconstruction de la base : python portraits_precompute.py [chemin]
#   cells_year   : sommes de cells par personne × année × journal × rubrique (WITHOUT ROWID, clé (key, year,
#                  newspaper, section) : les lignes d'une personne sont contiguës sur le disque)
#                  Plus, pour chaque score s, sa contribution au score attendu de /rankings_portraits :
#                  E_s = W_p · (S_all − S_p) / (W_all − W_p) et EW_s = W_p, où _p désigne la personne et _all tout
#                  le monde dans la cellule année × journal × rubrique (0 et 0 si la personne y est seule ; W_nm
#                  et S_valence_nm pour valence_nm). Score attendu = ΣE_s / ΣEW_s sur les cellules retenues.
#                  Index cells_year_section (key, section, year, newspaper, n_articles) : articles par rubrique.
#   totals_month : sommes de tout le monde par mois × journal × rubrique (W, W_nm et S_* seulement)
# Les tables sont remplacées dans une seule transaction ; les colonnes de comptes et de sommes sont reprises
# telles quelles de cells (agent, patient... compris), seules key/month/newspaper/section sont regroupées.
import sqlite3
import sys
import time

chemin = sys.argv[1] if len(sys.argv) > 1 else "/opt/bazoulay/portraits.sqlite"
conn = sqlite3.connect(chemin, isolation_level=None, timeout=120)  # attend que les lectures en cours (API) finissent
cols = [c[1] for c in conn.execute("pragma table_info(cells)")]
cles = ("key", "month", "newspaper", "section")
sommes = [c for c in cols if c not in cles]
totaux = [c for c in sommes if c == "n_articles" or c.startswith("W") and not c.startswith("WS")
          and not c.startswith("WW") or c.startswith("S_")]
# Scores et colonne de poids associée (mêmes conventions que portraits_scores dans app.py)
scores = {c[2:]: "W_nm" if c.endswith("_nm") else "W" for c in sommes if c.startswith("S_")}

t = time.time()
conn.execute("begin")
conn.execute("drop table if exists cells_year")
conn.execute("create temp table annee as select key, substr(month, 1, 4) as year, newspaper, section, "
             + ", ".join(f'sum("{c}") as "{c}"' for c in sommes) + " from cells group by key, year, newspaper, section")
poids = sorted(set(scores.values()))
conn.execute("create temp table tot as select year, newspaper, section, "
             + ", ".join([f'sum("{w}") as "{w}"' for w in poids] + [f'sum("S_{s}") as "S_{s}"' for s in scores])
             + " from annee group by year, newspaper, section")
attendus = []
for s, w in scores.items():
    pas_seul = f'(t."{w}" - a."{w}") > 1e-9 * max(abs(t."{w}"), 1)'
    attendus += [f'case when {pas_seul} then a."{w}" * (t."S_{s}" - a."S_{s}") / (t."{w}" - a."{w}") else 0 end',
                 f'case when {pas_seul} then a."{w}" else 0 end']
noms = sommes + [x for s in scores for x in (f"E_{s}", f"EW_{s}")]
conn.execute("create table cells_year (key text, year text, newspaper text, section text, "
             + ", ".join(f'"{c}" real' for c in noms) + ", primary key (key, year, newspaper, section)) without rowid")
conn.execute("insert into cells_year select a.key, a.year, a.newspaper, a.section, "
             + ", ".join([f'a."{c}"' for c in sommes] + attendus)
             + " from annee a join tot t using (year, newspaper, section) order by a.key, a.year, a.newspaper, a.section")
conn.execute("drop table if exists totals_month")
conn.execute("create table totals_month (month text, newspaper text, section text, "
             + ", ".join(f'"{c}" real' for c in totaux) + ", primary key (month, newspaper, section)) without rowid")
conn.execute("insert into totals_month select month, newspaper, section, "
             + ", ".join(f'sum("{c}")' for c in totaux) + " from cells group by month, newspaper, section")
# Index étroit pour compter les articles par personne × rubrique sans tri (section_share de /rankings_portraits)
conn.execute("create index cells_year_section on cells_year(key, section, year, newspaper, n_articles)")
conn.execute("commit")
try:
    conn.execute("analyze")  # statistiques du planificateur : utile, pas indispensable
except sqlite3.OperationalError as e:
    print("analyze non fait :", e)
for table in ("cells", "cells_year", "totals_month"):
    print(table, conn.execute(f"select count(*) from {table}").fetchone()[0], "lignes")
print(f"{time.time() - t:.0f} s")
