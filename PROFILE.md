# Empirical-Bayes priors for portraits: Phase 0 profile

Measured on 2026-11-25 on the production server with the production database, read-only, while production traffic was running:

- **Database:** `/opt/bazoulay/portraits.sqlite`, 1.9 GB.
- **Code:** `app.py` as deployed.
- **Server:** 4 cores, 31 GB RAM, load average about 3. gunicorn runs 20 workers with `--max-requests 100`.

The script and raw results are in the session scratchpad: `profile_prior.py`, `profile_full.log` and `profile_results.json`.

## How the numbers were measured

- **Runs:** 20 per case. Figures are p50 / p95 in milliseconds unless marked otherwise.
- **Cold:** the app's `lru_cache`s are cleared before each run, so every SQL query runs again. The OS page cache could not be dropped, because that needs root. The 1.9 GB file fits in the roughly 14 GB of page cache, and the db is read all the time, so production "cold" is in practice this "cold".
- **Warm:** the same call again, answered from the LRU cache.
- **Queries:** all three use section `politique`.
  - `full_all`: 2008–2026, all newspapers.
  - `2024_all`: 2024, all newspapers.
  - `full_lemonde`: 2008–2026, `lemonde` only.

  All three are year-aligned, so they use the fast `cells_year` path.

## (a) Data sizes

`cells` has 4,121,109 rows and `cells_year` 1,684,965. `names` has 19,770 people; the db only keeps people with at least 100 articles overall.

People counts are articles within the window, section `politique`:

| window | people | ≥10 | ≥30 | ≥50 | ≥100 | ≥500 |
|---|---|---|---|---|---|---|
| 2008–2026, all newspapers | 13,762 | 8,314 | 6,427 | — | 4,309 | 922 |
| 2024, all newspapers | 6,072 | 1,913 | 962 | 617 | 303 | 71 |
| 2008–2026, lemonde | 8,748 | 3,054 | 1,257 | 812 | 464 | 106 |

## (b) Current `/rankings_portraits`, broken down

| query | min | rows | SQL | post-proc. | CSV | (JSON) | **cold total** | **warm total** |
|---|---|---|---|---|---|---|---|---|
| full_all | 500 | 922 | 1391 / 1428 | 15 / 19 | 47 / 52 | 15 / 16 | **1452 / 1490** | 48 / 52 |
| full_all | 50 | 5,597 | 2901 / 2948 | 25 / 31 | 256 / 277 | 89 / 96 | **3179 / 3229** | 259 / 275 |
| full_all | 30 | 6,427 | 2911 / 2938 | 26 / 30 | 293 / 299 | 101 / 105 | **3227 / 3261** | 297 / 310 |
| 2024_all | 500 | 71 | 147 / 149 | 12 / 13 | 5 / 5 | 1 / 2 | **164 / 166** | 5 / 6 |
| 2024_all | 50 | 617 | 392 / 398 | 16 / 18 | 30 / 31 | 10 / 10 | **438 / 444** | 31 / 32 |
| 2024_all | 30 | 962 | 398 / 402 | 16 / 17 | 47 / 48 | 15 / 15 | **461 / 468** | 48 / 49 |
| full_lemonde | 500 | 106 | 386 / 390 | 11 / 12 | 6 / 6 | 2 / 2 | **404 / 408** | 7 / 7 |
| full_lemonde | 50 | 812 | 960 / 968 | 16 / 19 | 40 / 41 | 13 / 13 | **1016 / 1024** | 41 / 42 |
| full_lemonde | 30 | 1,257 | 940 / 1000 | 17 / 22 | 62 / 66 | 20 / 21 | **1019 / 1084** | 63 / 70 |

- **Totals:** "Cold total" and "warm total" include sort and CSV, which is what the route returns today.
- **JSON column:** `to_json(orient="records")` on the same frame, for comparison only.
- **Why min=500 is faster:** the SQL only reads the cells of people with at least `min` articles overall (from `names`).

## (c) Live between-person prior on top of (b)'s aggregate

This step computes, for valence, extremisme and dynamique, both absolute and relative (R − E):

- the pooled σ²;
- se_model² = σ²·WW/W²;
- the Paule–Mandel τ² and μ, solved as six problems at once in numpy.

The population is people with at least 30 articles in the window.

| query | K | Newton from τ²=0 (profiled run) | Newton from DerSimonian–Laird, rtol 1e-6 (final) |
|---|---|---|---|
| full_all | 6,427 | 32 / 34 | **≈ 7 / 10** (6 iterations) |
| 2024_all | 962 | 6 / 6 | ≈ 2 / 3 |
| full_lemonde | 1,257 | 7 / 8 | ≈ 2 / 3 |

The final solver agrees with an 80-step bisection to within 1e-14 relative, on simulated data with τ² = 0 and τ² > 0. The final full_all figure was measured while the profiling run competed for the CPU. The 2024 and lemonde figures are scaled by K from the profiled run, since cost is linear in K.

Values at min_prior = 30, for valence / extremisme / dynamique:

| query | K | σ² | τ² | μ | τ²_rel | μ_rel |
|---|---|---|---|---|---|---|
| full_all | 6,427 | .0565 / .0030 / .0118 | .0161 / .0017 / .00056 | −.035 / .009 / .009 | .0148 / .0017 / .00053 | .045 / −.003 / −.001 |
| 2024_all | 962 | .0518 / .0043 / .0075 | .0210 / .0021 / .0019 | −.032 / .015 / .014 | .0189 / .0020 / .0018 | .064 / −.007 / .010 |
| full_lemonde | 1,257 | .0360 / .0024 / .0064 | .0096 / .0015 / .00033 | −.058 / .010 / .004 | .0099 / .0015 / .00034 | .037 / −.007 / −.001 |

## (d) Live between-person prior without the ranking

This is one aggregation by person under the filters, with the min_prior = 30 HAVING, followed by the estimation.

| query | aggregation | estimation | total |
|---|---|---|---|
| full_all | 1797 / 1946 | 32 / 33 | 1829 / 1978 |
| 2024_all | 293 / 297 | 6 / 6 | 299 / 303 |
| full_lemonde | 828 / 833 | 7 / 8 | 835 / 840 |

## (e) Live within-person prior for time series

The population is the 6,427 people with at least 30 articles in politique, 2008–2026, all newspapers. The data are aggregated by person × unit, then a pooled DerSimonian–Laird τ² is computed across units within each person.

| unit | units | aggregation | estimation | total | σ² (val / ext / dyn) | τ²_within (val / ext / dyn) |
|---|---|---|---|---|---|---|
| year (`cells_year`) | 82,715 | 2378 / 2604 | 25 / 33 | 2403 / 2629 | .0509 / .0029 / .0110 | .0062 / .00030 / .0010 |
| quarter (`cells`) | 210,698 | 15503 / 15807 | 99 / 151 | 15623 / 15981 | .0490 / .0029 / .0107 | .0086 / .00055 / .0012 |
| month (`cells`) | 382,877 | 14298 / 15415 | 168 / 266 | 14497 / 15615 | .0475 / .0029 / .0106 | .0111 / .00086 / .0014 |
| newspaper (`cells_year`) | 118,290 | 3999 / 4617 | 37 / 41 | 4036 / 4653 | .0528 / .0029 / .0119 | .0031 / .00036 / .00010 |

## (f) Offline Option B fit

This was run once on a 10 % random sample of people (1,977 of them), all sections. Each level is a separate `GROUP BY` on `cells`.

| level | rows | aggregation (s) | moment estimation (s) |
|---|---|---|---|
| person | 7,656 | 10.0 | 0.01 |
| person × year | 83,435 | 6.9 | 0.05 |
| person × newspaper | 31,207 | 6.2 | 0.02 |
| person × year × newspaper | 164,875 | 8.7 | 0.14 |
| person × month | 248,873 | 10.0 | 0.19 |

The sample took about 42 s. Most of that is scan cost, which does not scale with the sample size: each query reads `cells`.

For the full data, the plan is one pass that reads the finest grain (person × month × newspaper × section, which is `cells` itself, 4.1 M rows) into numpy, then rolls up the five levels with `bincount`. My estimate is 1–3 minutes for all sections and the three scores; I have not measured the full run. That is fine to run inside `portraits_precompute.py` on each reload.

## Recommendation

| case | live cost (p95) | fits 20 ms? | design |
|---|---|---|---|
| Between-person prior inside `/rankings_portraits` (c) | 3–10 ms | yes | **Option A**: live Paule–Mandel (DL start), LRU-cached with the ranking |
| Between-person prior without a ranking (d) | 0.3–2.0 s | no | **Option B** |
| Within-person prior, person × period (e) | 2.4–16 s | no | **Option B** |
| Within-person prior, person × newspaper (e) | 4.7 s | no | **Option B** |

This is the outcome you expected: live Paule–Mandel inside `/rankings_portraits`, Option B for everything else. Option B's offline fit (f) costs minutes, at reload time only, and it is served as an O(1) lookup.

### Is the 20 ms budget realistic?

The budget is realistic for the prior's marginal cost, but it is the wrong thing to worry about. The route around it is already 10–300× over budget:

- **Cold `/rankings_portraits`** takes 0.16–3.3 s, almost all of it in SQL.
- **Warm at min = 30–50**, CSV serialisation alone takes 260–300 ms, on every request, because the CSV is re-serialised from the cached frame each time. JSON is 3× cheaper, and caching the serialised bytes would bring it to about 1 ms.
- **"Warm" is rare in production.** `lru_cache` is per worker, there are 20 workers, and `--max-requests 100` recycles each worker after 100 requests. Most filter changes on the page are therefore cold, about 1–3 s.

If the page has to feel snappy at min ≈ 30–50, that needs dealing with whether or not the prior is added. Options, by impact:

- a cache shared across workers (for example a small sqlite or disk cache keyed by the normalised parameters, invalidated on db mtime);
- caching the serialised response and returning fewer or rounded float columns;
- precomputing the default filter combinations at reload.

I'd suggest folding that into Phase 1, since the brief says the route must be fast at min ≈ 30–50.

### When `min` is above `min_prior`

The prior's population is everyone with at least `min_prior` (30) articles. If the page asks for min = 500, the route currently scans only the people with at least 500 articles. Computing the prior then needs the min = 30 aggregate, which adds +1.5 s cold for full_all (1.4 s → 2.9 s).

Fix: always aggregate at `min(min, min_prior)` and cache that, then filter rows for output. Once the front end moves to min ≈ 30–50, this costs nothing.

## Issues to settle before Phase 1

1. **The exact degrees-of-freedom correction needs Σ_d W_d³, which `cells` does not store.**
   - Under the model, with s_d = S_d/W_d having variance σ² and R the W-weighted mean, E[Q_p] = σ²·(WW − 2·W3/W + WW²/W²), where W3 = Σ_d W_d³.
   - **With W3:** the correction is exact and unbiased.
   - **Without W3:** the best available is the Kish form WW·(1 − WW/W²) = WW·(1 − 1/n_eff). It is exact for equal weights. By Cauchy–Schwarz, W3/W ≥ WW²/W², so this underestimates σ², with a per-person relative bias of order 1/n_eff. That is likely a few percent at 30+ articles.
   - The simulation test would show the bias.
   - **Recommendation:** add `W3` (and `W3_nm`) to the db build. It is one more `sum(w**3)`. The build script isn't on this server, so that's on your side. I'd use the Kish form only as a fallback.

2. **Where σ² is pooled changes it materially.**
   - A person's Q over a window includes the true drift of their score across years and newspapers, not only article noise.
   - The measured σ² for valence falls from 0.0565 (pooled by person, full window) to 0.0528 (person × newspaper), 0.0509 (person × year), 0.0490 (person × quarter) and 0.0475 (person × month).
   - If σ² absorbs the drift, that drift is counted twice: once in se² and again in τ².
   - **Recommendation:** estimate σ² once, offline, at the finest grain (person × month × newspaper cells, which Option B reads anyway). Serve it per section, and era if split. The ranking route then uses that σ² instead of re-pooling at person level live. That also makes σ² consistent between the ranking and the Option B views, and removes one step from the live path.
   - The brief currently says to pool Q_p by person under the route's filters, so this is a change to the spec I'm asking you to approve.

3. **The prior's population is truncated.** The db only holds people with at least 100 articles overall. In narrow windows (one year, one newspaper), the min_prior = 30 population is "≥ 30 in the window **and** ≥ 100 overall". That leans toward well-covered people, so τ² will be somewhat off for narrow filters. The simulation can still use long-tailed counts, but real-data checks inherit this.

4. **The deployed API differs from the brief.**
   - `ngramsci_routes.py` and `political_portraits.html` are not on this server.
   - Here `/rankings_portraits` returns **CSV**, not JSON, so there is no JSON object to add a `"prior"` key to.
   - Options:
     - (a) add `format=json`, returning `{"rows": [...], "prior": {...}}`, and leave the CSV untouched;
     - (b) a separate `/rankings_prior` route sharing the same cached aggregate;
     - (c) tell me where the JSON version lives and I'll work there.
   - I'd go with (a), since JSON is also 3× cheaper to serialise.

5. **Small items for Phase 1:**
   - DerSimonian–Laird is undefined when no person has two or more units (seen as a NaN warning on a sparse sample level). It will return null with a `reason`, as will K < 3.
   - The 12,194 cells with W = 0 are skipped per score.

Phase 0 is done. I'm stopping here and waiting for your go-ahead, and your answers on items 1, 2 and 4, before starting Phase 1.

---

# Phase 1, part 1: live prior in `/rankings_portraits` (Option A, built)

Decisions taken after Phase 0:
- **Output:** JSON via `format=json`.
- **σ² for the ranking:** pooled by person, live under the route's filters. The Kish degrees-of-freedom correction is used, with no third moment.

## What changed

**New module `portraits_prior.py`**, pure numpy:
- `sigma2_pooled` and `ddl_sigma2`, which hold the derivation;
- `paule_mandel`: Newton from a DerSimonian–Laird start, vectorised over the scores;
- `prior_entre`;
- `fit_components` and `tau2_implique`, for Option B.

**`/rankings_portraits?format=json`** returns `{"rows": [...], "prior": {...}}`:
- `rows` are the same rows as the CSV, plus `n_eff` = W²/WW. The front end needs it for se_model² = sigma2 / n_eff.
- `prior` is `{"K", "K_rel", "min_prior", "sigma2", "mu", "tau2", "mu_rel", "tau2_rel"}`, for valence, extremisme and dynamique. Values are null and a `reason` is given when K < 3.
- `min_prior` (default 30) is independent of `min`. The aggregate is computed once at `min(min, min_prior)`.

**CSV output is byte-identical to before.** I checked five queries: year-aligned, monthly bounds, `section_flag`, `order=asc` with another score, and a filter matching nobody.

**Internal changes:**
- `portraits_classement` is split into `portraits_agregat` (cached sums) and the ranking.
- The response body is cached per normalised parameters in `portraits_reponse`, an LRU of 8 entries invalidated by the db mtime.

## Benchmarks after the change

End to end through Flask's test client, 20 runs, p50 / p95 in ms. Cold means every cache is cleared before the run.

| query | min | CSV cold | CSV warm | JSON cold | JSON warm | prior alone |
|---|---|---|---|---|---|---|
| full_all | 500 | 1361 / 1414 | 1 / 2 | 2668 / 2709 | 1 / 1 | 8 / 8 |
| full_all | 50 | 2896 / 2921 | 5 / 5 | 2772 / 2847 | 2 / 3 | 8 / 8 |
| full_all | 30 | 2945 / 2993 | 6 / 6 | 2760 / 2781 | 2 / 2 | 8 / 8 |
| 2024_all | 500 | 198 / 199 | 1 / 1 | 392 / 395 | 1 / 1 | 2 / 2 |
| 2024_all | 50 | 417 / 438 | 1 / 1 | 404 / 408 | 1 / 1 | 2 / 2 |
| 2024_all | 30 | 438 / 440 | 1 / 1 | 410 / 419 | 1 / 1 | 2 / 2 |
| full_lemonde | 500 | 415 / 450 | 1 / 1 | 909 / 1006 | 1 / 1 | 3 / 3 |
| full_lemonde | 50 | 964 / 1032 | 1 / 3 | 933 / 975 | 1 / 1 | 3 / 3 |
| full_lemonde | 30 | 995 / 1056 | 1 / 2 | 945 / 1018 | 1 / 2 | 3 / 3 |

**The prior costs 2–8 ms p95, within the 20 ms budget.**

- **At min = 30–50**, JSON cold is as fast as CSV cold or faster: JSON serialises in about a third of the time, which pays for the prior.
- **At min = 500**, JSON cold is about 2× CSV, because the prior needs the min_prior = 30 population. This is the cost predicted in Phase 0. It disappears once the page uses min ≤ 30.
- **Warm requests drop from 48–310 ms to 1–6 ms** thanks to the response cache, for CSV too. The per-worker cache and `--max-requests 100` caveat from Phase 0 still applies.

# Phase 1, part 2: Option B blocked on W3

Phase 0 said the Kish correction underestimates σ² by "a few percent". **That holds for persons, not for the finest cells.** The real article weights are very dispersed: the CV² of weights within a cell is about 1.2 (from n·WW/W² − 1 on politique cells with 20+ articles).

With weights that dispersed, simulation gives these Kish biases on σ̂²:

| unit | articles | σ̂² / σ² |
|---|---|---|
| person | 30–60 | 0.93–0.95 |
| person | 60–200 | 0.95–0.98 |
| person | 200+ | 0.99+ |
| finest cell (person × month × newspaper) | 1 to a few, which is what Option B needs σ² from | ≈ 0.80 |

The ranking prior is fine. Option B is not.

**Effect on the fit.** Simulated with realistic weights, 4,000 people of whom about 920 have 30+ articles:

| | σ² | a | b | c | d | m |
|---|---|---|---|---|---|---|
| true | 0.0500 | 0.0200 | 0.0060 | 0.0040 | 0.0030 | 0.0050 |
| without W3 (Kish) | 0.0403 | 0.0192 | 0.0059 | 0.0044 | **0.0053** | **0.0098** |
| with W3 | 0.0516 | 0.0194 | 0.0067 | 0.0042 | 0.0025 | 0.0040 |

The underestimated noise reappears as spurious person × year × newspaper (d) and month (m) variance. That would under-shrink time series and heatmaps.

**What I tried without W3:**
- An exact formula for 1- and 2-article cells. This is kept.
- Estimating W3 from n, W and WW with a lognormal moment relation. This was dropped: it overestimates σ² by about 50 % on cells of 3–20 articles.

**Real data also show σ² is not constant across cell sizes** (politique, valence):

| cell size | estimator | σ² |
|---|---|---|
| 2 articles | exact | 0.038 |
| 20–49 | Kish, a lower bound | 0.047 |
| 50+ | Kish, a lower bound | 0.051 |

Either heavily covered people have noisier article scores, or their true score moves within a month. With W3 this can be measured exactly and handled: the noise term of the fit becomes Σ_cells σ²(cell size)·WW_c/W_u², which the estimating equations accept unchanged.

**Code status.** `fit_components` uses an exact W3 column automatically when `cells` has one. The tests run with and without W3, and `test_composantes_sans_W3_biaisees` documents the bias. All 11 tests pass:
- σ² unbiased with equal weights;
- the Kish bias bounded at person level;
- σ², all five components and Paule–Mandel recovered with W3, for continuous, gamma-weighted and {−1, 0, 1} scores and with zero components;
- the cross-check between the components' implied τ² and the live Paule–Mandel τ²;
- the edge cases (K < 3, W = 0, missing scores, nobody matching).

**Still to do once W3 is available:**
- the precomputed table and the `/variance_components` route;
- the era comparison (2008–2016 against 2017–2026);
- the real-data cross-check between the implied τ² and the live Paule–Mandel τ².
