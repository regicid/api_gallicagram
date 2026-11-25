"""Tests par simulation de portraits_prior.py (σ² poolé, composantes de variance, Paule–Mandel).

Lancer : /opt/bazoulay/venvs/bin/python -m pytest -q tests/test_portraits_prior.py   (ou python tests/test_portraits_prior.py
pour afficher les valeurs estimées). Les personnes sont tirées du modèle à composantes avec des variances connues,
des nombres d'articles à longue traîne (la plupart < 20, quelques-unes à plusieurs milliers), des poids aléatoires
et des scores d'articles continus ou dans {−1, 0, 1} ; on construit cells puis on vérifie les estimations.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import portraits_prior as P  # noqa: E402

VRAI = {"a": 0.02, "b": 0.006, "c": 0.004, "d": 0.003, "m": 0.005}


def simule(n_pers=4000, comp=VRAI, sigma2=0.05, discret=False, poids="lognormal", seed=0, n_journaux=8,
           annees=range(2010, 2020), mu_sd=0.05):
    """Cellules (key × month × newspaper) simulées ; renvoie (cells, σ² vrai)."""
    rng = np.random.default_rng(seed)
    annees = np.array(list(annees))
    n = np.minimum(np.floor(np.exp(rng.normal(2.2, 1.6, n_pers))).astype(int) + 1, 5000)
    pers = np.repeat(np.arange(n_pers), n)
    N = len(pers)
    # journal : un journal "maison" une fois sur deux, sinon au hasard ; années dans une fenêtre par personne
    maison = rng.integers(0, n_journaux, n_pers)
    o = np.where(rng.random(N) < 0.5, maison[pers], rng.integers(0, n_journaux, N))
    debut = rng.integers(0, len(annees), n_pers)
    duree = rng.integers(1, len(annees) + 1, n_pers)
    t = (debut[pers] + rng.integers(0, 10 ** 6, N) % duree[pers]) % len(annees)
    mois = rng.integers(1, 13, N)

    def effet(var, *cles):
        if var == 0:
            return np.zeros(N)
        code = P._codes(*cles)
        return rng.normal(0, np.sqrt(var), code.max() + 1)[code]

    mu_ot = rng.normal(0, mu_sd, (n_journaux, len(annees)))[o, t]
    theta = (mu_ot + effet(comp["a"], pers) + effet(comp["b"], pers, t) + effet(comp["c"], pers, o)
             + effet(comp["d"], pers, o, t) + effet(comp["m"], pers, t * 100 + mois))
    if discret:
        # s ∈ {−1, 0, 1}, moyenne θ (bornée), variance 2q − θ² ≈ σ² avec q = 0.25 et θ petit
        th = np.clip(theta, -0.45, 0.45)
        u = rng.random(N)
        s = np.where(u < 0.25 + th / 2, 1.0, np.where(u < 0.5, -1.0, 0.0))
        sigma2 = float(np.mean(0.5 - th ** 2))
    else:
        s = theta + rng.normal(0, np.sqrt(sigma2), N)
    # Poids d'articles : CV² ≈ 1,2 comme dans portraits.sqlite (lognormal σ = 0,89, ou gamma de forme 0,83)
    w = {"lognormal": lambda: rng.lognormal(0, 0.89, N), "gamma": lambda: rng.gamma(0.83, 1.2, N),
         "egaux": lambda: np.ones(N)}[poids]()
    S = w * s
    df = pd.DataFrame({"key": pers.astype(str), "month": [f"{y}-{m:02d}" for y, m in zip(annees[t], mois)],
                       "newspaper": o.astype(str), "n_articles": 1.0, "W": w, "WW": w * w,
                       "W3": w ** 3})
    for sc in P.SCORES_PRIOR:  # mêmes scores pour les trois (le test porte sur la méthode)
        df[f"S_{sc}"], df[f"SS_{sc}"], df[f"WS_{sc}"] = S, S * S, w * S
    cells = df.groupby(["key", "month", "newspaper"], as_index=False).sum()
    return cells, sigma2


def proche(est, vrai, rel=0.3, abs_=6e-4):
    return abs(est - vrai) <= rel * vrai + abs_


def test_sigma2_poids_egaux_sans_biais():
    # Unités de 2 à 40 articles de poids égaux : la correction de Kish est exacte (n − 1)
    rng = np.random.default_rng(1)
    rapports = []
    for _ in range(200):
        n = rng.integers(2, 40, 300)
        u = np.repeat(np.arange(300), n)
        s = rng.normal(0, 1, len(u)) * np.sqrt(0.05) + rng.normal(0, 0.3, 300)[u]
        w = np.full(len(u), 2.0)
        agg = lambda x: np.bincount(u, x)
        sig, _ = P.sigma2_pooled(agg(w), agg(w * w), agg(w * s)[None], agg((w * s) ** 2)[None], agg(w * w * s)[None])
        rapports.append(sig[0] / 0.05)
    assert abs(np.mean(rapports) - 1) < 0.005, np.mean(rapports)


def sigma2_unites(n, tirage, rng):
    u = np.repeat(np.arange(len(n)), n)
    w = tirage(len(u))
    s = rng.normal(0, np.sqrt(0.05), len(u))
    agg = lambda x: np.bincount(u, x)
    args = (agg(w), agg(w * w), agg(w * s)[None], agg((w * s) ** 2)[None], agg(w * w * s)[None])
    return P.sigma2_pooled(*args)[0][0] / 0.05, P.sigma2_pooled(*args, n=n)[0][0] / 0.05


def test_sigma2_poids_aleatoires():
    # Poids dispersés comme dans portraits.sqlite (CV² ≈ 1,2). Unités de 2 articles : exact avec n.
    # Personnes de 30 à 70 articles : Kish sous-estime de moins de 9 % ; de 200 à 2000 : de moins de 2 %.
    rng = np.random.default_rng(2)
    for nom, tirage in (("lognormal", lambda k: rng.lognormal(0, 0.89, k)), ("gamma", lambda k: rng.gamma(0.83, 1.2, k))):
        kish, exact = sigma2_unites(np.full(100000, 2), tirage, rng)
        assert abs(exact - 1) < 0.02 and kish < 0.8, (nom, kish, exact)
        kish, _ = sigma2_unites(rng.integers(30, 70, 4000), tirage, rng)
        assert 0.91 < kish < 1.0, (nom, kish)
        kish, _ = sigma2_unites(rng.integers(200, 2000, 3000), tirage, rng)
        assert 0.98 < kish < 1.01, (nom, kish)


def test_sigma2_ignore_scores_manquants():
    W = np.array([3.0, 2.0, 0.0]); WW = np.array([3.5, 2.5, 0.0])
    S = np.array([[1.0, np.nan, 0.0]]); SS = np.array([[1.0, 0.5, 0.0]]); WS = np.array([[1.2, 1.0, 0.0]])
    sig, n = P.sigma2_pooled(W, WW, S, SS, WS)
    assert n[0] == 1 and np.isfinite(sig[0])


def ajuste(cells, w3=True):
    return P.fit_components(cells if w3 else cells.drop(columns="W3"), min_prior=30)


def test_composantes_sans_W3_biaisees():
    # Documente pourquoi W3 est requis : sans lui, poids dispersés → σ² sous-estimé d'environ 20 % sur les
    # cellules fines, et le bruit manquant part dans s2_d et s2_m
    cells, s2 = simule()
    r = ajuste(cells, w3=False)["valence"]
    assert r["sigma2"] < 0.9 * s2 and r["s2_m"] > 1.4 * VRAI["m"]


def verifie(res, comp, sigma2, rel=0.3, abs_=6e-4, tol_sigma=0.04):
    r = res["valence"]
    assert abs(r["sigma2"] / sigma2 - 1) < tol_sigma, (r["sigma2"], sigma2)
    for k, v in comp.items():
        assert proche(r[f"s2_{k}"], v, rel, abs_), (k, r[f"s2_{k}"], v)


def test_composantes_continues():
    cells, s2 = simule()
    verifie(ajuste(cells), VRAI, s2)


def test_composantes_nulles():
    comp = {"a": 0.02, "b": 0.0, "c": 0.004, "d": 0.0, "m": 0.0}
    cells, s2 = simule(comp=comp, seed=3)
    verifie(ajuste(cells), comp, s2, abs_=1e-3)  # composantes nulles : estimées à moins de 0,001 (≈ 2 % de σ²)


def test_composantes_poids_gamma():
    cells, s2 = simule(seed=8, poids="gamma")
    verifie(ajuste(cells), VRAI, s2, tol_sigma=0.06)


def test_composantes_discretes():
    cells, s2 = simule(discret=True, seed=4)
    # Scores {−1, 0, 1} : σ² ≈ 0,46, dix fois le cas continu, et hétéroscédastique (0,5 − θ²) : composantes de
    # l'ordre de 0,005 estimées à ±0,005 près (≈ 1 % de σ²) avec ~900 personnes
    verifie(ajuste(cells), VRAI, s2, rel=0.4, abs_=5e-3, tol_sigma=0.05)


def test_paule_mandel_et_cross_check():
    # Classement poolé : PM sur les personnes (≥ 30 articles) ; τ² attendu = moyenne pondérée PM de τ²_p implicite
    comp = {"a": 0.02, "b": 0.006, "c": 0.004, "d": 0.0, "m": 0.0}
    # Baselines μ_ot nulles : le score attendu E vaut 0 et le score relatif est le score absolu
    cells, s2 = simule(comp=comp, seed=5, mu_sd=0.0)
    agg = cells.groupby("key").sum(numeric_only=True)
    sc = "valence"
    col = lambda x: agg[x].to_numpy()[None]
    pr = P.prior_entre(agg["n_articles"], agg["W"], agg["WW"], col(f"S_{sc}"), col(f"SS_{sc}"), col(f"WS_{sc}"),
                       np.zeros((1, len(agg))), scores=(sc,), min_prior=30)
    assert pr["K"] > 500
    res = ajuste(cells)[sc]
    # τ²_p implicite : s2_a + s2_b Σv² + s2_c Σu² pour chaque personne ≥ 30 articles
    c = cells.copy(); c["year"] = c["month"].str[:4]
    Wp = c.groupby("key")["W"].sum()
    parts = {k: (c.groupby(["key", g])["W"].sum() / Wp).pow(2).groupby(level=0).sum()
             for k, g in (("b", "year"), ("c", "newspaper"))}
    sel = agg.index[agg["n_articles"] >= 30]
    t2p = P.tau2_implique(res, {k: v.loc[sel].to_numpy() for k, v in parts.items()})
    t2_pm = pr["tau2_rel"][sc]
    assert abs(pr["tau2"][sc] - t2_pm) < 1e-12
    assert abs(t2_pm - np.mean(t2p)) < 0.35 * np.mean(t2p) + 0.003, (t2_pm, np.mean(t2p))


def test_paule_mandel_zero_et_simple():
    rng = np.random.default_rng(6)
    v = rng.uniform(0.001, 0.01, 2000)
    y0 = rng.normal(0, np.sqrt(v))
    y1 = rng.normal(0.1, np.sqrt(v + 0.004))
    t2, mu, K = P.paule_mandel(np.vstack([y0, y1]), np.vstack([v, v]))
    assert t2[0] < 3e-4 and abs(t2[1] - 0.004) < 0.0008 and abs(mu[1] - 0.1) < 0.01 and (K == 2000).all()


def test_bords():
    # K < 3 → valeurs nulles et "reason" ; W = 0 ignoré ; personne ne correspond aux filtres
    pr = P.prior_entre([40, 50], [1, 2], [0.5, 1], [[0.1, 0.2]], [[0.1, 0.1]], [[0.1, 0.2]], [[0, 0]], scores=("valence",))
    assert pr["K"] == 2 and pr["tau2"]["valence"] is None and "reason" in pr
    vide = P.prior_entre([], [], [], np.zeros((1, 0)), np.zeros((1, 0)), np.zeros((1, 0)), np.zeros((1, 0)), scores=("valence",))
    assert vide["K"] == 0 and "reason" in vide
    pr = P.prior_entre([40, 50, 60, 70], [0, 2, 3, 4], [0, 1, 2, 2], [[0, 0.2, 0.3, -0.1]], [[0, 0.1, 0.1, 0.1]],
                       [[0, 0.2, 0.3, 0.1]], [[np.nan, 0, 0, 0]], scores=("valence",))
    assert pr["K"] == 3 and pr["sigma2"]["valence"] is not None
    cells, _ = simule(n_pers=30, seed=7)
    cells["n_articles"] = 1.0  # personne n'atteint 30 articles par cellule… on force un seuil inatteignable
    res = P.fit_components(cells, min_prior=10 ** 9)
    assert res["valence"]["K"] == 0 and res["valence"]["s2_a"] is None


if __name__ == "__main__":
    for nom, comp, disc, seed, poids in (("continu", VRAI, False, 0, "lognormal"), ("nulles", {"a": 0.02, "b": 0.0, "c": 0.004, "d": 0.0, "m": 0.0}, False, 3, "lognormal"),
                                  ("discret", VRAI, True, 4, "lognormal"), ("gamma", VRAI, False, 8, "gamma"), ("egaux", VRAI, False, 9, "egaux")):
        cells, s2 = simule(comp=comp, discret=disc, seed=seed, poids=poids)
        r = ajuste(cells)["valence"]
        r0 = ajuste(cells, w3=False)["valence"]
        print("   sans W3 : σ² %.4f" % r0["sigma2"], " ".join(f"{k}→{r0['s2_' + k]:.4f}" for k in comp))
        print(nom, "cellules", len(cells), "σ² vrai %.4f estimé %.4f" % (s2, r["sigma2"]),
              " ".join(f"{k}: {v:.4f}→{r['s2_' + k]:.4f}" for k, v in comp.items()), "K", r["K"])
