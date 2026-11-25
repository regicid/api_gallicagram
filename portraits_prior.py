"""Priors empirical Bayes des scores de portraits.sqlite (numpy seulement, sans Flask).

Utilisé par app.py (prior entre personnes en direct dans /rankings_portraits, lecture de /variance_components),
par portraits_precompute.py (ajustement hors ligne des composantes de variance) et par tests/test_portraits_prior.py.

Notations (cellule = personne × mois × journal × rubrique, d = article) : W = Σ_d W_d, WW = Σ_d W_d²,
S = Σ_d S_d, SS = Σ_d S_d², WS = Σ_d W_d·S_d, avec S_d = W_d·s_d (s_d : score moyen de l'article). Les cellules se
somment. Pour une unité poolée : R = S/W, Q = SS − 2R·WS + R²·WW = Σ_d (S_d − R·W_d)², n_eff = W²/WW.
"""
import numpy as np
import pandas as pd
from scipy.optimize import nnls

SCORES_PRIOR = ("valence", "extremisme", "dynamique")
MIN_PRIOR = 30
COMPOSANTES = ("a", "b", "c", "d", "m")


def ddl_sigma2(W, WW, n=None, W3=None):
    """Dénominateur de σ̂² pour chaque unité : E[Q] = σ² · ddl (voir sigma2_pooled).

    Exact : ddl = WW − 2·W3/W + WW²/W², W3 = Σ_d W_d³, absent de cells. On prend la forme de Kish
    WW − WW²/W² = WW (1 − 1/n_eff), exacte pour des poids égaux (W3 donné : formule exacte) ; avec n (nombre d'articles), elle est remplacée
    par la valeur exacte pour 1 article (0) et 2 articles (W et WW déterminent les deux poids :
    W3 = 1,5·W·WW − 0,5·W³). Une estimation de W3 par relation de moments (lognormale) a été essayée pour
    n ≥ 3 et écartée : très biaisée sur les petites cellules (voir PROFILE.md).
    """
    W = np.asarray(W, float); WW = np.asarray(WW, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        if W3 is not None:  # colonne W3 = Σ_d W_d³ présente : formule exacte
            return np.clip(WW - 2 * np.asarray(W3, float) / W + WW * WW / (W * W), 0, None)
        kish = WW - WW * WW / (W * W)
        if n is None:
            return kish
        n = np.asarray(n, float)
        deux = WW - 2 * (1.5 * WW - 0.5 * W * W) + WW * WW / (W * W)
        return np.where(n <= 1, 0.0, np.where(n < 2.5, np.clip(deux, 0, kish), kish))


def sigma2_pooled(W, WW, S, SS, WS, n=None, W3=None):
    """Variance intra-personne σ² poolée sur des unités (personnes, ou cellules les plus fines).

    W, WW (et n, nombre d'articles, facultatif) : (n_u,) ; S, SS, WS : (k, n_u) pour k scores. Renvoie σ² (k,)
    et le nombre d'unités utilisées par score. Unités où W ≤ 0 ou un score non fini ignorées pour ce score.

    Modèle : s_d = θ_u + ε_d, ε_d indépendants de variance σ², et R = Σ W_d s_d / W la moyenne pondérée de
    l'unité u. Alors Var(R) = σ²·WW/W² (d'où se_model² = σ²·WW/W²) et
        Q = Σ_d W_d² (s_d − R)²,
        E[Q] = σ² Σ_d W_d² (1 − 2 W_d/W + WW/W²) = σ² (WW − 2·W3/W + WW²/W²),   W3 = Σ_d W_d³,
    donc σ̂² = ΣQ / Σ ddl est sans biais si ddl = WW − 2·W3/W + WW²/W² (poids égaux : w²(n − 1), la correction
    n − 1 habituelle). W3 n'est pas dans cells : ddl_sigma2 prend la forme de Kish WW (1 − 1/n_eff), trop grande
    (W3/W ≥ WW²/W², Cauchy–Schwarz), sauf pour 1 ou 2 articles où elle est exacte avec n. σ̂² sous-estime donc σ²
    d'un facteur 1 − O(CV²/n_eff) : avec des poids aussi dispersés que dans portraits.sqlite (CV² ≈ 1,2), environ
    −7 % pour des personnes de 30 à 60 articles, −3 à −5 % de 60 à 200, < 1 % au-delà (simulation), mais bien
    plus sur des cellules de 3 à 20 articles. Une unité d'un seul article ne compte pas.
    Une dérive du vrai score dans l'unité gonfle σ̂² : pooler sur des cellules fines donne la variance
    d'échantillonnage pure (tests/test_portraits_prior.py vérifie le biais par simulation).
    """
    W = np.asarray(W, float); WW = np.asarray(WW, float)
    S, SS, WS = (np.atleast_2d(np.asarray(x, float)) for x in (S, SS, WS))
    with np.errstate(divide="ignore", invalid="ignore"):
        R = S / W
        Q = SS - 2 * R * WS + R * R * WW
        ddl = ddl_sigma2(W, WW, n, W3)
    ok = (W > 0) & np.isfinite(Q) & np.isfinite(ddl)
    num = np.where(ok, Q, 0).sum(1)
    den = np.where(ok, ddl, 0).sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, num / den, np.nan), ok.sum(1)


def paule_mandel(y, v, rtol=1e-6, iters=50):
    """τ² de Paule–Mandel, pour plusieurs problèmes à la fois (lignes de y, v ; mêmes K colonnes).

    Résout Σ_i w_i (y_i − μ_w)² = K − 1, w_i = 1/(τ² + v_i), τ² ≥ 0 ; τ² = 0 si la statistique est déjà
    ≤ K − 1 en τ² = 0. μ = μ_w à la solution. f(τ²) = Σ w (y − μ_w)² − (K − 1) est décroissante et convexe :
    Newton (dérivée Σ w² (y − μ_w)², μ_w stationnaire) part de l'estimateur de DerSimonian–Laird, ou de 0 si
    DL dépasse déjà la racine, et converge de façon monotone (6 itérations environ sur les données réelles).
    Colonnes non finies ignorées par ligne. Renvoie τ² (m,), μ (m,), K (m,).
    """
    y = np.atleast_2d(np.asarray(y, float)); v = np.atleast_2d(np.asarray(v, float))
    ok = np.isfinite(y) & np.isfinite(v) & (v > 0)
    y = np.where(ok, y, 0.0); v = np.where(ok, v, 1.0)
    K = ok.sum(1)
    m = ok.astype(float)

    def stat(t2):
        w = m / (t2[:, None] + v)
        sw = w.sum(1)
        mu = (w * y).sum(1) / np.where(sw > 0, sw, 1)
        r2 = (y - mu[:, None]) ** 2
        return w, mu, r2

    w, mu, r2 = stat(np.zeros(len(y)))
    sw = w.sum(1)
    Q0 = (w * r2).sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        t2 = np.maximum(0, (Q0 - (K - 1)) / (sw - (w * w).sum(1) / sw))
    t2 = np.where(np.isfinite(t2), t2, 0.0)
    w, mu, r2 = stat(t2)
    t2 = np.where((w * r2).sum(1) - (K - 1) < 0, 0.0, t2)  # DL au-delà de la racine : repartir de 0
    for _ in range(iters):
        w, mu, r2 = stat(t2)
        wr2 = w * r2
        f = wr2.sum(1) - (K - 1)
        d = (w * wr2).sum(1)
        with np.errstate(divide="ignore", invalid="ignore"):
            pas = np.where((f > 0) & (d > 0), f / d, 0.0)
        t2 = t2 + pas
        if (pas <= rtol * np.maximum(t2, 1e-15)).all():
            break
    w, mu, _ = stat(t2)
    return t2, mu, K


def prior_entre(n_articles, W, WW, S, SS, WS, E, scores=SCORES_PRIOR, min_prior=MIN_PRIOR):
    """Prior entre personnes du classement : population = personnes avec n_articles ≥ min_prior.

    Arrays par personne : n_articles, W, WW (n,) ; S, SS, WS, E (k, n), E = score attendu (baseline relative,
    ΣE_s/ΣEW_s, NaN si inconnue). σ² poolé sur les personnes (sigma2_pooled, W3 estimé), se_i² = σ²·WW_i/W_i², puis
    Paule–Mandel sur R_i (mu, tau2) et sur R_i − E_i (mu_rel, tau2_rel), avec les mêmes se_i.
    Renvoie le dict "prior" de /rankings_portraits ; valeurs nulles et "reason" si K < 3.
    """
    n_articles = np.asarray(n_articles, float); W = np.asarray(W, float); WW = np.asarray(WW, float)
    sel = (n_articles >= min_prior) & (W > 0)
    S, SS, WS, E = (np.atleast_2d(np.asarray(x, float))[:, sel] for x in (S, SS, WS, E))
    W, WW, n_articles = W[sel], WW[sel], n_articles[sel]
    out = {"K": int(sel.sum()), "min_prior": min_prior}
    vides = {k: {s: None for s in scores} for k in ("sigma2", "mu", "tau2", "mu_rel", "tau2_rel")}
    if out["K"] < 3:
        return {**out, "K_rel": 0, **vides, "reason": f"moins de 3 personnes avec au moins {min_prior} articles"}
    sigma2, _ = sigma2_pooled(W, WW, S, SS, WS, n_articles)
    R = S / W
    se2 = sigma2[:, None] * (WW / (W * W))[None, :]
    t2, mu, K = paule_mandel(np.vstack([R, R - E]), np.vstack([se2, se2]))
    k = len(scores)
    out["K_rel"] = int(K[k:].min())
    nombre = lambda x: None if not np.isfinite(x) else float(x)
    out.update({"sigma2": dict(zip(scores, map(nombre, sigma2))), "mu": dict(zip(scores, map(nombre, mu[:k]))),
                "tau2": dict(zip(scores, map(nombre, t2[:k]))), "mu_rel": dict(zip(scores, map(nombre, mu[k:]))),
                "tau2_rel": dict(zip(scores, map(nombre, t2[k:])))})
    if out["K_rel"] < 3:
        out["mu_rel"] = {s: None for s in scores}; out["tau2_rel"] = {s: None for s in scores}
        out["reason"] = "moins de 3 personnes avec un score attendu (seules dans toutes leurs cellules)"
    return out


def _codes(*cols):
    """Code entier dense d'une combinaison de colonnes (factorize successifs, sans dépassement)."""
    code = np.zeros(len(cols[0]), np.int64)
    for c in cols:
        c = pd.factorize(c)[0] if not np.issubdtype(np.asarray(c).dtype, np.integer) else np.asarray(c)
        code = pd.factorize(code * (int(c.max()) + 1) + c)[0]
    return code


def fit_components(c, scores=SCORES_PRIOR, min_prior=MIN_PRIOR):
    """Composantes de variance de l'option B, par méthode des moments, pour une rubrique et une période.

    c : DataFrame des cellules les plus fines (une ligne par key × month × newspaper, mois 'AAAA-MM'), colonnes
    key, month, newspaper, n_articles, W, WW, W3 (Σ_d W_d³ ; sans elle σ² est biaisé, voir sigma2_pooled) et
    S_s, SS_s, WS_s pour chaque score. Toutes les personnes doivent y
    être (elles servent à la baseline) ; seules celles avec ≥ min_prior articles servent à l'estimation.

    Modèle du vrai score de la personne p dans le journal o, l'année t, le mois τ (dans t) :
        θ_poτ = μ_ot + a_p + b_pt + c_po + d_pot + m_pτ,
    effets indépendants, centrés, de variances s2_a (niveau stable), s2_b (dérive annuelle), s2_c (traitement par
    journal), s2_d (interaction année × journal), s2_m (mois dans l'année) ; articles : s_d = θ + ε_d, Var ε = σ².

    Étapes :
    1. μ_ot : baseline journal × année des autres personnes (leave-one-out, comme E_s de /rankings_portraits) ;
       x = S − W·μ par cellule, cellules où la personne est seule exclues. x̄_u = Σx/ΣW pour toute unité u.
       Les x̄ ont une moyenne commune mu_rel (écart moyen des personnes à leur baseline), qui s'élimine partout.
    2. σ² : sigma2_pooled sur les cellules les plus fines (variance d'échantillonnage pure).
    3. Pour une unité u (ensemble de cellules) et une composante k de groupes g (p ; p×t ; p×o ; p×o×t ; p×τ),
       la part de k dans x̄_u est Σ_g (W_ug/W_u)·effet_g, donc
           Var(x̄_u) = s2_a + Σ_k s2_k·A_uk + σ²·WW_u/W_u²,   A_uk = Σ_g W_ug² / W_u².
    4. Équations d'estimation, chaque personne pesant autant (sinon les plus citées dominent) :
       - entre personnes : Q_a = Σ_p (x̄_p − moyenne)², E[Q_a] = (K−1)/K · Σ_p Var(x̄_p) ;
       - dans la personne, pour les niveaux L = année, journal, année × journal, mois :
           Q_L = Σ_p Σ_{u∈L, u⊂p} s_u (x̄_u − x̄_p)²,  s_u = W_u/W_p,
           E[Q_L] = Σ_u s_u Var(x̄_u) − Σ_p Var(x̄_p)      (car x̄_p = Σ s_u x̄_u ; s2_a s'élimine).
       Cinq équations linéaires en (s2_a, s2_b, s2_c, s2_d, s2_m) une fois σ² fixé : moindres carrés sous
       contrainte ≥ 0 (NNLS, lignes normalisées), c.-à-d. des estimateurs de type DerSimonian–Laird pour
       plusieurs niveaux emboîtés et croisés, tronqués à 0. Une interaction personne × mois × journal, absente du
       modèle, se répartirait sur s2_d et s2_m.
    Renvoie {score: {sigma2, s2_a, ..., s2_m, mu_rel, K, min_prior}} ; valeurs None et "reason" si K < 3.
    """
    n_p = c.groupby("key")["n_articles"].sum()  # articles de la personne dans la rubrique et la période
    c = c[(c["W"] > 0)].reset_index(drop=True)
    year = c["month"].str[:4].to_numpy()
    p_all, cles = pd.factorize(c["key"])
    n_p = n_p.reindex(cles).to_numpy(float)
    ot = _codes(c["newspaper"].to_numpy(), year)
    pot_all = _codes(p_all, ot)
    W = c["W"].to_numpy(float)
    W_ot = np.bincount(ot, W)
    W_pot = np.bincount(pot_all, W)
    autres = W_ot[ot] - W_pot[pot_all]
    pas_seul = autres > 1e-9 * np.maximum(np.abs(W_ot[ot]), 1)
    garde = pas_seul & (n_p[p_all] >= min_prior)
    K = int(np.unique(p_all[garde]).size)
    res = {}
    if K < 3:
        return {s: {"sigma2": None, **{f"s2_{k}": None for k in COMPOSANTES}, "mu_rel": None, "K": K,
                    "min_prior": min_prior, "reason": f"moins de 3 personnes avec au moins {min_prior} articles"}
                for s in scores}
    # μ_ot leave-one-out par score, calculé avec tout le monde, puis restriction aux cellules retenues
    mus = {}
    for s in scores:
        S = c[f"S_{s}"].to_numpy(float)
        S_ot, S_pot = np.bincount(ot, S), np.bincount(pot_all, S)
        with np.errstate(divide="ignore", invalid="ignore"):
            mus[s] = ((S_ot[ot] - S_pot[pot_all]) / autres)[garde]
    c = c[garde].reset_index(drop=True)
    year, month, news = year[garde], c["month"].to_numpy(), c["newspaper"].to_numpy()
    p = pd.factorize(c["key"])[0]
    W, WW, n_c = c["W"].to_numpy(float), c["WW"].to_numpy(float), c["n_articles"].to_numpy(float)
    W3 = c["W3"].to_numpy(float) if "W3" in c else None  # nécessaire pour un σ² sans biais sur les cellules fines
    groupes = {"a": p, "b": _codes(p, year), "c": _codes(p, news), "d": _codes(p, news, year), "m": _codes(p, month)}
    niveaux = {"annee": groupes["b"], "journal": groupes["c"], "annee_journal": groupes["d"], "mois": groupes["m"]}

    def sommes(u):
        n = int(u.max()) + 1
        return np.bincount(u, W, n), np.bincount(u, WW, n)

    def coefs(u, Wu):
        # A_uk pour k = b, c, d, m (s2_a : coefficient 1)
        A = []
        for k in COMPOSANTES[1:]:
            pair = _codes(u, groupes[k])
            Wug = np.bincount(pair, W)
            u_pair = np.zeros(len(Wug), np.int64); u_pair[pair] = u
            A.append(np.bincount(u_pair, Wug ** 2, len(Wu)) / Wu ** 2)
        return np.array(A)  # (4, n_u)

    Wp, WWp = sommes(p)
    Ap = coefs(p, Wp)
    lignes, cible_A, cible_e, stats = [], [], [], {}
    # Équation entre personnes
    lignes.append(("a", None))
    # Équations dans la personne
    pre = {}
    for nom, u in niveaux.items():
        Wu, WWu = sommes(u)
        p_u = np.zeros(len(Wu), np.int64); p_u[u] = p
        su = Wu / Wp[p_u]
        pre[nom] = (u, Wu, WWu, p_u, su, coefs(u, Wu))
    for s in scores:
        S, SS, WS = (c[f"{x}_{s}"].to_numpy(float) for x in ("S", "SS", "WS"))
        sigma2 = float(sigma2_pooled(W, WW, S[None], SS[None], WS[None], n_c, W3)[0][0])
        X = S - W * mus[s]
        xp = np.bincount(p, X) / Wp
        mu_rel = float(xp.mean())
        A_rows, rhs = [], []
        # entre personnes
        f = (K - 1) / K
        A_rows.append(f * np.r_[K, Ap.sum(1)]); rhs.append(((xp - mu_rel) ** 2).sum() - f * sigma2 * (WWp / Wp ** 2).sum())
        for nom, (u, Wu, WWu, p_u, su, Au) in pre.items():
            xu = np.bincount(u, X, len(Wu)) / Wu
            Q = (su * (xu - xp[p_u]) ** 2).sum()
            coef = np.r_[0.0, (su[None] * Au).sum(1) - Ap.sum(1)]
            bruit = sigma2 * ((su * WWu / Wu ** 2).sum() - (WWp / Wp ** 2).sum())
            A_rows.append(coef); rhs.append(Q - bruit)
        A = np.array(A_rows); b = np.array(rhs)
        norme = np.linalg.norm(A, axis=1); norme[norme == 0] = 1
        theta, _ = nnls(A / norme[:, None], b / norme)
        res[s] = {"sigma2": sigma2, **{f"s2_{k}": float(t) for k, t in zip(COMPOSANTES, theta)},
                  "mu_rel": mu_rel, "K": K, "min_prior": min_prior}
    return res


def tau2_implique(comp, parts):
    """Variance a priori τ²_p d'un score poolé, d'après les composantes (option B).

    parts : dict de Σ v², Σ u², Σ w² (et Σ mois²) des parts du poids W de la personne par année ("b"), journal
    ("c"), année × journal ("d"), mois ("m") dans la sélection ; tableaux numpy ou scalaires.
    """
    return comp["s2_a"] + sum(comp[f"s2_{k}"] * parts[k] for k in ("b", "c", "d", "m") if k in parts)
