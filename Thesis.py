import pyarrow
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy import stats
import warnings

warnings.filterwarnings('ignore')
np.random.seed(42)

# ---------------- config ----------------

TRAIN_WINDOW_YEARS = 10
HORIZONS = [1, 2]
STEP_SIZE = 1
ANALYSIS_START_YEAR = 1990
ANALYSIS_END_YEAR = 2023
OBS_PER_PARAM = 10
N_BOOTSTRAP = 999
MERGE_SAME_TIME_INDEX = False
MAX_TRANSITION_GAP = 11  # time_index units, 10 = one year
EXCLUDED_YEARS = {2018}  # sparse season coverage

# ---------------- load data ----------------

df = pd.read_parquet('Vogue Runway Dataset.parquet')
embeddings = np.load('Vogue Runway Image.npy')
assert len(df) == embeddings.shape[0], "Metadata and embeddings are misaligned!"

# ---------------- prepare data ----------------

season_order = {
    'Spring': 1, 'Summer': 1, 'Spring/Summer': 1, 'SS': 1,
    'Fall': 2, 'Winter': 2, 'Fall/Winter': 2, 'FW': 2,
    'Resort': 1.5, 'Pre-Fall': 1.8, 'Holiday': 2.5
}

df['season_num'] = df['season'].map(lambda x: season_order.get(x, 1))
df['time_index'] = df['year'] * 10 + df['season_num']

mask = df['year'].between(ANALYSIS_START_YEAR, ANALYSIS_END_YEAR) & ~df['year'].isin(EXCLUDED_YEARS)
df = df[mask].copy()
embeddings = embeddings[mask.values]

# ---------------- helpers ----------------

def calculate_optimal_k(n_transitions, var_order=2, obs_per_param=10):
    k = int(np.floor(np.sqrt(n_transitions / (var_order * obs_per_param))))
    return max(k, 3)


def compute_aic_bic(residuals, n, p, k_dim):
    sigma = np.cov(residuals.T)
    if k_dim == 1:
        log_det = np.log(float(sigma))
    else:
        sign, log_det = np.linalg.slogdet(sigma)
        if sign <= 0:
            sigma += np.eye(k_dim) * 1e-10
            sign, log_det = np.linalg.slogdet(sigma)
    log_lik = -n / 2 * (k_dim * np.log(2 * np.pi) + log_det + k_dim)
    return 2 * p - 2 * log_lik, p * np.log(n) - 2 * log_lik


def rao_wilks_f(logdet_restricted, logdet_unrestricted, n, k):
    p = k
    df_h = k
    df_e = n - 2 * k * k

    wilks_lambda = np.exp(logdet_unrestricted - logdet_restricted)

    denom = (p ** 2 + df_h ** 2 - 5)
    s = np.sqrt((p ** 2 * df_h ** 2 - 4) / denom) if denom != 0 else 1.0

    df1 = p * df_h
    df2 = s * (df_e - (p - df_h + 1) / 2) - (p * df_h - 2) / 2

    lam_s = wilks_lambda ** (1 / s)
    F_stat = ((1 - lam_s) / lam_s) * (df2 / df1)
    p_value = 1 - stats.f.cdf(F_stat, df1, df2)
    return wilks_lambda, F_stat, df1, df2, p_value


def clark_west_test(y_true, y_pred_restricted, y_pred_unrestricted):
    # squared Euclidean loss, one-sided
    e1 = np.sum((y_true - y_pred_restricted) ** 2, axis=1)
    e2 = np.sum((y_true - y_pred_unrestricted) ** 2, axis=1)
    adj = np.sum((y_pred_restricted - y_pred_unrestricted) ** 2, axis=1)
    f = e1 - (e2 - adj)

    T = len(f)
    if T < 2:
        return np.nan, np.nan
    f_mean, f_std = np.mean(f), np.std(f, ddof=1)
    if f_std == 0 or np.isnan(f_std):
        return np.nan, np.nan
    cw_stat = f_mean / (f_std / np.sqrt(T))
    return cw_stat, 1 - stats.norm.cdf(cw_stat)


def build_triplet_data(agg_df, pc_cols, max_gap=MAX_TRANSITION_GAP):
    X1_list, X2_list, y_list, meta_list = [], [], [], []
    n_dropped_gap = 0
    for brand in agg_df['designer'].unique():
        bd = (agg_df[agg_df['designer'] == brand]
              .sort_values('time_index').reset_index(drop=True))
        if len(bd) < 3:
            continue
        states = bd[pc_cols].values
        years = bd['year'].values
        ti = bd['time_index'].values
        n_imgs = bd['n_images'].values if 'n_images' in bd.columns else np.full(len(bd), np.nan)
        for i in range(len(states) - 2):
            if ti[i + 1] - ti[i] > max_gap or ti[i + 2] - ti[i + 1] > max_gap:
                n_dropped_gap += 1
                continue
            X1_list.append(states[i + 1])
            X2_list.append(np.concatenate([states[i], states[i + 1]]))
            y_list.append(states[i + 2])
            meta_list.append({
                'brand': brand,
                'year_tm1': years[i], 'year_t': years[i + 1], 'year_tp1': years[i + 2],
                'ti_tp1': ti[i + 2],
                'n_images_tp1': n_imgs[i + 2],
            })
    if not meta_list:
        return None
    return {
        'X1': np.vstack(X1_list),
        'X2': np.vstack(X2_list),
        'y':  np.vstack(y_list),
        'meta': pd.DataFrame(meta_list),
        'n_dropped': n_dropped_gap,
    }


def build_quadruple_data(agg_df, pc_cols, max_gap=MAX_TRANSITION_GAP):
    Stm1, St, Stp1, Stp2, meta_list = [], [], [], [], []
    for brand in agg_df['designer'].unique():
        bd = (agg_df[agg_df['designer'] == brand]
              .sort_values('time_index').reset_index(drop=True))
        if len(bd) < 4:
            continue
        states = bd[pc_cols].values
        years = bd['year'].values
        ti = bd['time_index'].values
        for i in range(len(states) - 3):
            gaps = [ti[i+1]-ti[i], ti[i+2]-ti[i+1], ti[i+3]-ti[i+2]]
            if any(g > max_gap for g in gaps):
                continue
            Stm1.append(states[i])
            St.append(states[i + 1])
            Stp1.append(states[i + 2])
            Stp2.append(states[i + 3])
            meta_list.append({
                'brand': brand,
                'year_tm1': years[i], 'year_t': years[i + 1],
                'year_tp1': years[i + 2], 'year_tp2': years[i + 3],
            })
    if not meta_list:
        return None
    return {
        'Stm1': np.vstack(Stm1), 'St': np.vstack(St),
        'Stp1': np.vstack(Stp1), 'Stp2': np.vstack(Stp2),
        'meta': pd.DataFrame(meta_list),
    }


def icc_oneway(x, groups):
    d = pd.DataFrame({'x': x, 'g': np.asarray(groups)})
    N, G = len(d), d['g'].nunique()
    sizes = d.groupby('g').size()
    m0 = (N - (sizes ** 2).sum() / N) / (G - 1)
    grand = d['x'].mean()
    gmeans = d.groupby('g')['x'].transform('mean')
    msb = ((gmeans - grand) ** 2).sum() / (G - 1)
    msw = ((d['x'] - gmeans) ** 2).sum() / (N - G)
    icc = (msb - msw) / (msb + (m0 - 1) * msw)
    anova_p = 1 - stats.f.cdf(msb / msw, G - 1, N - G)
    return icc, anova_p, m0


def clustering_summary(resid, groups):
    iccs, n_sig, m0 = [], 0, None
    for j in range(resid.shape[1]):
        icc_j, p_j, m0 = icc_oneway(resid[:, j], groups)
        iccs.append(icc_j)
        n_sig += int(p_j < 0.05)
    iccs = np.array(iccs)
    deff = 1 + (m0 - 1) * max(iccs.mean(), 0)
    return {
        'n_groups': pd.Series(groups).nunique(), 'avg_size': m0,
        'icc_mean': iccs.mean(), 'icc_min': iccs.min(), 'icc_max': iccs.max(),
        'n_sig': n_sig, 'deff': deff, 'eff_n': len(resid) / deff,
    }


def ols_residual_maker(X):
    Xc = np.hstack([np.ones((X.shape[0], 1)), X])
    P = np.linalg.pinv(Xc)
    return lambda Y: Y - Xc @ (P @ Y)


def nested_test_stats(r1, r2, n, k, p1, p2):
    rss1, rss2 = np.sum(r1 ** 2), np.sum(r2 ** 2)
    F = ((rss1 - rss2) / (p2 - p1)) / (rss2 / (n * k - p2 - k))
    _, ld1 = np.linalg.slogdet(np.cov(r1.T) + np.eye(k) * 1e-12)
    _, ld2 = np.linalg.slogdet(np.cov(r2.T) + np.eye(k) * 1e-12)
    _, WF, _, _, _ = rao_wilks_f(ld1, ld2, n, k)
    return F, WF


def wild_cluster_bootstrap(X1, X2, Y, clusters, k, p1, p2, B=999, seed=42):
    # restricted wild bootstrap, Rademacher weights, one draw per brand
    n = len(Y)
    res1, res2 = ols_residual_maker(X1), ols_residual_maker(X2)
    r1, r2 = res1(Y), res2(Y)
    F_obs, W_obs = nested_test_stats(r1, r2, n, k, p1, p2)
    fitted1 = Y - r1

    codes, uniq = pd.factorize(np.asarray(clusters))
    rng = np.random.default_rng(seed)
    F_b, W_b = np.empty(B), np.empty(B)
    for b in range(B):
        w = rng.choice([-1.0, 1.0], size=len(uniq))[codes]
        Y_star = fitted1 + r1 * w[:, None]
        F_b[b], W_b[b] = nested_test_stats(res1(Y_star), res2(Y_star), n, k, p1, p2)

    return {
        'F_obs': F_obs, 'F_p': (1 + np.sum(F_b >= F_obs)) / (B + 1),
        'F_boot_95': np.quantile(F_b, 0.95),
        'W_obs': W_obs, 'W_p': (1 + np.sum(W_b >= W_obs)) / (B + 1),
        'W_boot_95': np.quantile(W_b, 0.95),
        'n_clusters': len(uniq),
    }


def var_stability(model, k, order):
    # largest companion-matrix eigenvalue modulus, < 1 = stable
    C = model.coef_
    if order == 1:
        comp = C
    else:
        A2, A1 = C[:, :k], C[:, k:]  # features are [s_{t-1}, s_t]
        comp = np.zeros((2 * k, 2 * k))
        comp[:k, :k] = A1
        comp[:k, k:] = A2
        comp[k:, :k] = np.eye(k)
    return np.max(np.abs(np.linalg.eigvals(comp)))


# ---------------- PCA ----------------

brand_counts = df['designer'].value_counts()
mask_brand = df['designer'].isin(brand_counts[brand_counts >= 4].index)
df = df[mask_brand].copy()
embeddings = embeddings[mask_brand.values]

# temporary PCA just to count transitions for k
temp_k = 5
temp_scores = PCA(n_components=temp_k).fit_transform(embeddings)
temp_df = df.copy()
for i in range(temp_k):
    temp_df[f'tPC{i+1}'] = temp_scores[:, i]
temp_agg = (temp_df.groupby(['designer', 'year', 'season', 'time_index'])
            .agg({f'tPC{i+1}': 'mean' for i in range(temp_k)}).reset_index())

n_transitions_est = sum(max(0, len(g) - 2) for _, g in temp_agg.groupby('designer'))
k = calculate_optimal_k(n_transitions_est, var_order=2, obs_per_param=OBS_PER_PARAM)

pca = PCA(n_components=k)
scores = pca.fit_transform(embeddings)
var_exp = pca.explained_variance_ratio_.sum()

pc_cols = [f'PC{i+1}' for i in range(k)]
for i, col in enumerate(pc_cols):
    df[col] = scores[:, i]

agg_df = (df.groupby(['designer', 'year', 'season', 'time_index'])
          .agg(**{pc: (pc, 'mean') for pc in pc_cols},
               n_images=(pc_cols[0], 'size'))
          .reset_index().sort_values(['designer', 'time_index']))

# ---------------- same time_index check ----------------

dup_mask = agg_df.duplicated(['designer', 'time_index'], keep=False)
n_dup_rows = int(dup_mask.sum())
n_collections_raw = len(agg_df)

if MERGE_SAME_TIME_INDEX and n_dup_rows:
    # image-weighted mean
    tmp = agg_df.copy()
    for pc in pc_cols:
        tmp[pc] = tmp[pc] * tmp['n_images']
    agg_df = (tmp.groupby(['designer', 'year', 'time_index'])
              .agg(**{pc: (pc, 'sum') for pc in pc_cols},
                   n_images=('n_images', 'sum'),
                   season=('season', lambda x: '+'.join(sorted(set(x)))))
              .reset_index())
    for pc in pc_cols:
        agg_df[pc] = agg_df[pc] / agg_df['n_images']
    agg_df = agg_df.sort_values(['designer', 'time_index'])

# ---------------- in-sample tests ----------------

triplets = build_triplet_data(agg_df, pc_cols)
X_2, y_2 = triplets['X2'], triplets['y']
X_1_aligned = X_2[:, k:]
n = len(y_2)
p1, p2 = k * k, 2 * k * k

var2 = LinearRegression().fit(X_2, y_2)
resid_2 = y_2 - var2.predict(X_2)
rss_2 = np.sum(resid_2 ** 2)
mse_2 = mean_squared_error(y_2, var2.predict(X_2))

var1 = LinearRegression().fit(X_1_aligned, y_2)
resid_1 = y_2 - var1.predict(X_1_aligned)
rss_1 = np.sum(resid_1 ** 2)
mse_1 = mean_squared_error(y_2, var1.predict(X_1_aligned))

# pooled F, df2 counts all n*k scalar residuals
f_df1 = p2 - p1
f_df2 = n * k - p2 - k
f_stat = ((rss_1 - rss_2) / f_df1) / (rss_2 / f_df2)
f_pvalue = 1 - stats.f.cdf(f_stat, f_df1, f_df2)

_, logdet1 = np.linalg.slogdet(np.cov(resid_1.T) + np.eye(k) * 1e-12)
_, logdet2 = np.linalg.slogdet(np.cov(resid_2.T) + np.eye(k) * 1e-12)
lr_stat = n * (logdet1 - logdet2)
lr_pvalue = 1 - stats.chi2.cdf(lr_stat, df=p2 - p1)

wilks_lambda, wilks_f_stat, wilks_df1, wilks_df2, wilks_pvalue = rao_wilks_f(logdet1, logdet2, n, k)

AIC_1, BIC_1 = compute_aic_bic(resid_1, n, p1, k)
AIC_2, BIC_2 = compute_aic_bic(resid_2, n, p2, k)
mse_improvement = (1 - mse_2 / mse_1) * 100

stab_1 = var_stability(var1, k, order=1)
stab_2 = var_stability(var2, k, order=2)

# ---------------- residual diagnostics ----------------

meta_is = triplets['meta'].reset_index(drop=True)

clust_season = clustering_summary(resid_2, meta_is['ti_tp1'].values)
clust_brand = clustering_summary(resid_2, meta_is['brand'].values)

# lag-1 correlation within brand
order = meta_is.sort_values(['brand', 'ti_tp1']).index.values
rs = resid_2[order]
bs = meta_is.loc[order, 'brand'].values
same = bs[1:] == bs[:-1]
e_now, e_prev = rs[1:][same], rs[:-1][same]
lag1 = np.array([np.corrcoef(e_now[:, j], e_prev[:, j])[0, 1] for j in range(k)])
lag1_band = 2 / np.sqrt(same.sum())

# error size vs image count
err_size = np.sum(resid_2 ** 2, axis=1)
imgs = meta_is['n_images_tp1'].values
rho, rho_p = stats.spearmanr(imgs, err_size)
by_q = pd.Series(err_size).groupby(pd.qcut(imgs, 4, duplicates='drop'), observed=True).mean()

dim_var = resid_2.var(axis=0)

# normality
skews = stats.skew(resid_2, axis=0)
kurts = stats.kurtosis(resid_2, axis=0)
jb_p = np.array([stats.jarque_bera(resid_2[:, j]).pvalue for j in range(k)])

centred = resid_2 - resid_2.mean(axis=0)
S_inv = np.linalg.inv(np.cov(centred.T, bias=True))
d2 = np.einsum('ij,jk,ik->i', centred, S_inv, centred)
mardia_b2 = np.mean(d2 ** 2)
mardia_expected = k * (k + 2)
mardia_z = (mardia_b2 - mardia_expected) / np.sqrt(8 * k * (k + 2) / n)
share_3sd = np.mean(np.abs(centred / centred.std(axis=0)) > 3)

# ---------------- season-demeaned robustness ----------------

agg_dm = agg_df.copy()
agg_dm[pc_cols] = agg_dm[pc_cols] - agg_dm.groupby('time_index')[pc_cols].transform('mean')
trip_dm = build_triplet_data(agg_dm, pc_cols)

X2_dm, y_dm = trip_dm['X2'], trip_dm['y']
X1_dm = X2_dm[:, k:]
n_dm = len(y_dm)

var2_dm = LinearRegression().fit(X2_dm, y_dm)
var1_dm = LinearRegression().fit(X1_dm, y_dm)
r2_dm = y_dm - var2_dm.predict(X2_dm)
r1_dm = y_dm - var1_dm.predict(X1_dm)
rss1_dm, rss2_dm = np.sum(r1_dm ** 2), np.sum(r2_dm ** 2)

f_dm = ((rss1_dm - rss2_dm) / (p2 - p1)) / (rss2_dm / (n_dm * k - p2 - k))
f_dm_p = 1 - stats.f.cdf(f_dm, p2 - p1, n_dm * k - p2 - k)

_, ld1_dm = np.linalg.slogdet(np.cov(r1_dm.T) + np.eye(k) * 1e-12)
_, ld2_dm = np.linalg.slogdet(np.cov(r2_dm.T) + np.eye(k) * 1e-12)
lr_dm_p = 1 - stats.chi2.cdf(n_dm * (ld1_dm - ld2_dm), df=p2 - p1)
_, wf_dm, _, _, wp_dm = rao_wilks_f(ld1_dm, ld2_dm, n_dm, k)
mse_gain_dm = (1 - np.mean(r2_dm ** 2) / np.mean(r1_dm ** 2)) * 100
stab_1_dm = var_stability(var1_dm, k, 1)
stab_2_dm = var_stability(var2_dm, k, 2)
clust_season_dm = clustering_summary(r2_dm, trip_dm['meta']['ti_tp1'].values)

# ---------------- wild cluster bootstrap ----------------

boot_orig = wild_cluster_bootstrap(X_1_aligned, X_2, y_2, meta_is['brand'].values,
                                   k, p1, p2, B=N_BOOTSTRAP)
boot_dm = wild_cluster_bootstrap(X1_dm, X2_dm, y_dm, trip_dm['meta']['brand'].values,
                                 k, p1, p2, B=N_BOOTSTRAP)

# ---------------- rolling-window validation ----------------

quadruples = build_quadruple_data(agg_df, pc_cols)

last_start = ANALYSIS_END_YEAR - TRAIN_WINDOW_YEARS - max(HORIZONS) + 1
window_starts = list(range(ANALYSIS_START_YEAR, last_start + 1, STEP_SIZE))
min_train_obs = p2 + 5

window_results = {h: [] for h in HORIZONS}
train_sizes = []
window_stability = []
tmeta = triplets['meta']

for w in window_starts:
    train_years = set(range(w, w + TRAIN_WINDOW_YEARS)) - EXCLUDED_YEARS
    train_end = w + TRAIN_WINDOW_YEARS - 1

    train_mask = (tmeta['year_tm1'].isin(train_years) &
                  tmeta['year_t'].isin(train_years) &
                  tmeta['year_tp1'].isin(train_years))
    n_train = train_mask.sum()
    if n_train < min_train_obs:
        continue

    X2_train = triplets['X2'][train_mask.values]
    y_train = triplets['y'][train_mask.values]
    X1_train = X2_train[:, k:]

    var1_w = LinearRegression().fit(X1_train, y_train)
    var2_w = LinearRegression().fit(X2_train, y_train)
    window_stability.append((train_end, var_stability(var1_w, k, 1), var_stability(var2_w, k, 2)))
    train_sizes.append(n_train)

    # h = 1
    if 1 in HORIZONS:
        test_year = train_end + 1
        h1_mask = (tmeta['year_tm1'].isin(train_years) &
                   tmeta['year_t'].isin(train_years) &
                   (tmeta['year_tp1'] == test_year))
        if h1_mask.sum() >= 3:
            X2_test = triplets['X2'][h1_mask.values]
            y_test = triplets['y'][h1_mask.values]
            pred1 = var1_w.predict(X2_test[:, k:])
            pred2 = var2_w.predict(X2_test)

            mse1 = mean_squared_error(y_test, pred1)
            mse2 = mean_squared_error(y_test, pred2)
            cw_stat, cw_p = clark_west_test(y_test, pred1, pred2)
            window_results[1].append({
                'train_end': train_end, 'test_year': test_year, 'n_train': n_train,
                'n_test': h1_mask.sum(), 'mse_var1': mse1, 'mse_var2': mse2,
                'gap_pct': (1 - mse2 / mse1) * 100, 'cw_stat': cw_stat, 'cw_pvalue': cw_p,
                'y_test': y_test, 'pred1': pred1, 'pred2': pred2,
            })

    # h = 2, iterated (actual t+1 never used)
    if 2 in HORIZONS and quadruples is not None:
        test_year = train_end + 2
        qmeta = quadruples['meta']
        h2_mask = (qmeta['year_tm1'].isin(train_years) &
                   qmeta['year_t'].isin(train_years) &
                   (qmeta['year_tp2'] == test_year))
        if h2_mask.sum() >= 3:
            Stm1 = quadruples['Stm1'][h2_mask.values]
            St = quadruples['St'][h2_mask.values]
            y_test = quadruples['Stp2'][h2_mask.values]

            pred1 = var1_w.predict(var1_w.predict(St))
            pred2_step1 = var2_w.predict(np.hstack([Stm1, St]))
            pred2 = var2_w.predict(np.hstack([St, pred2_step1]))

            mse1 = mean_squared_error(y_test, pred1)
            mse2 = mean_squared_error(y_test, pred2)
            cw_stat, cw_p = clark_west_test(y_test, pred1, pred2)
            window_results[2].append({
                'train_end': train_end, 'test_year': test_year, 'n_train': n_train,
                'n_test': h2_mask.sum(), 'mse_var1': mse1, 'mse_var2': mse2,
                'gap_pct': (1 - mse2 / mse1) * 100, 'cw_stat': cw_stat, 'cw_pvalue': cw_p,
                'y_test': y_test, 'pred1': pred1, 'pred2': pred2,
            })

# ---------------- pooled out-of-sample ----------------

pooled_summary = {}
for h in HORIZONS:
    wr = window_results[h]
    if not wr:
        continue
    y_all = np.vstack([r['y_test'] for r in wr])
    p1_all = np.vstack([r['pred1'] for r in wr])
    p2_all = np.vstack([r['pred2'] for r in wr])

    mse1_p = mean_squared_error(y_all, p1_all)
    mse2_p = mean_squared_error(y_all, p2_all)
    cw_stat_p, cw_p_p = clark_west_test(y_all, p1_all, p2_all)
    wins = sum(1 for r in wr if r['mse_var2'] < r['mse_var1'])

    pooled_summary[h] = {
        'n_windows': len(wr), 'n_obs': len(y_all),
        'mse_var1': mse1_p, 'mse_var2': mse2_p,
        'gap_pct': (1 - mse2_p / mse1_p) * 100,
        'cw_stat': cw_stat_p, 'cw_pvalue': cw_p_p, 'wins': wins,
        'sign_p': stats.binomtest(wins, len(wr), 0.5, alternative='greater').pvalue,
    }

# ============================================================================
# RESULTS
# ============================================================================

print("=" * 80)
print("RESULTS")
print("=" * 80)

print("\nSAMPLE")
print(f"  Images: {len(df):,}   Brands: {df['designer'].nunique():,}   "
      f"Collections: {len(agg_df):,}")
print(f"  k = {k}   variance explained = {var_exp:.1%}")
print(f"  VAR(2) transitions: {n:,}   dropped by gap filter: {triplets['n_dropped']:,}")
print(f"  obs/param (system) = {n/p2:.1f}:1   obs/regressor (per equation) = {n/(2*k):.1f}:1")
print(f"  Collections sharing a brand + time_index: {n_dup_rows:,} of {n_collections_raw:,} "
      f"(merged = {MERGE_SAME_TIME_INDEX})")

print("\nIN-SAMPLE")
print(f"  Pooled F:  F = {f_stat:.4f}  (df = {f_df1}, {f_df2})  p = {f_pvalue:.3e}")
print(f"  Wilks:     Lambda = {wilks_lambda:.4f}  F = {wilks_f_stat:.4f}  "
      f"(df = {wilks_df1:.1f}, {wilks_df2:.1f})  p = {wilks_pvalue:.3e}")
print(f"  LR:        stat = {lr_stat:,.1f}  p = {lr_pvalue:.3e}")
print(f"  AIC:       VAR(1) = {AIC_1:,.1f}   VAR(2) = {AIC_2:,.1f}")
print(f"  BIC:       VAR(1) = {BIC_1:,.1f}   VAR(2) = {BIC_2:,.1f}")
print(f"  MSE:       VAR(1) = {mse_1:.6f}   VAR(2) = {mse_2:.6f}   ({mse_improvement:.1f}% improvement)")
print(f"  Stability: VAR(1) = {stab_1:.4f}   VAR(2) = {stab_2:.4f}")

print("\nRESIDUAL DIAGNOSTICS (VAR(2))")
for label, c in [("By season", clust_season), ("By brand", clust_brand)]:
    print(f"  {label}: ICC mean = {c['icc_mean']:.4f} (min {c['icc_min']:.4f}, max {c['icc_max']:.4f})  "
          f"sig dims = {c['n_sig']}/{k}  design effect = {c['deff']:.2f}  effective n = {c['eff_n']:,.0f}")
print(f"  Within-brand lag-1 corr: mean = {lag1.mean():+.4f} (min {lag1.min():+.4f}, max {lag1.max():+.4f})  "
      f"dims beyond +/-{lag1_band:.4f}: {int(np.sum(np.abs(lag1) > lag1_band))}/{k}")
print(f"  Spearman(n_images, squared error): rho = {rho:+.4f}  p = {rho_p:.3e}")
for q, v in by_q.items():
    print(f"    images in {q}: mean squared error = {v:.6f}")
print(f"  Residual variance ratio (max/min dim): {dim_var.max() / dim_var.min():.1f}x   "
      f"share of RSS from PC1-PC5: {dim_var[:5].sum() / dim_var.sum():.1%}")
print(f"  Skewness: mean = {skews.mean():+.3f} (min {skews.min():+.3f}, max {skews.max():+.3f})")
print(f"  Excess kurtosis: mean = {kurts.mean():+.3f} (min {kurts.min():+.3f}, max {kurts.max():+.3f})")
print(f"  Jarque-Bera rejections: {int(np.sum(jb_p < 0.05))}/{k}")
print(f"  Mardia kurtosis: {mardia_b2:.1f} vs {mardia_expected} ({mardia_b2 / mardia_expected:.2f}x, z = {mardia_z:.1f})")
print(f"  Residuals beyond 3 SD: {share_3sd:.2%} (Gaussian = 0.27%)")

print("\nSEASON-DEMEANED")
print(f"  n = {n_dm:,}")
print(f"  Pooled F = {f_dm:.4f}  p = {f_dm_p:.3e}")
print(f"  Wilks F  = {wf_dm:.4f}  p = {wp_dm:.3e}")
print(f"  LR p = {lr_dm_p:.3e}")
print(f"  MSE improvement = {mse_gain_dm:.1f}%")
print(f"  Stability: VAR(1) = {stab_1_dm:.4f}   VAR(2) = {stab_2_dm:.4f}")
print(f"  Season ICC mean = {clust_season_dm['icc_mean']:.4f}")

print(f"\nWILD CLUSTER BOOTSTRAP (by brand, B = {N_BOOTSTRAP})")
for label, b in [("Original", boot_orig), ("Season-demeaned", boot_dm)]:
    print(f"  {label} ({b['n_clusters']:,} clusters):")
    print(f"    Pooled F = {b['F_obs']:.4f}  95th pct = {b['F_boot_95']:.4f}  p = {b['F_p']:.4f}")
    print(f"    Wilks F  = {b['W_obs']:.4f}  95th pct = {b['W_boot_95']:.4f}  p = {b['W_p']:.4f}")

print("\nOUT-OF-SAMPLE (pooled)")
for h, s in pooled_summary.items():
    print(f"  h = {h}: windows = {s['n_windows']}  obs = {s['n_obs']:,}  "
          f"VAR(2) wins = {s['wins']}/{s['n_windows']} (sign p = {s['sign_p']:.4f})")
    print(f"         MSE VAR(1) = {s['mse_var1']:.6f}  VAR(2) = {s['mse_var2']:.6f}  "
          f"gap = {s['gap_pct']:+.1f}%  CW = {s['cw_stat']:.2f}  p = {s['cw_pvalue']:.3e}")
if window_stability:
    ws = np.array([(a, b) for _, a, b in window_stability])
    print(f"  Max stability across windows: VAR(1) = {ws[:, 0].max():.4f}   VAR(2) = {ws[:, 1].max():.4f}")
    print(f"  Training transitions per window: {min(train_sizes):,} to {max(train_sizes):,}")

print("\nOUT-OF-SAMPLE (per window)")
for h in HORIZONS:
    if not window_results[h]:
        continue
    print(f"  h = {h}")
    print(f"  {'Train end':>10} {'Test yr':>8} {'Train n':>8} {'n':>6} {'Gap %':>8} {'CW':>8} {'CW p':>10}")
    for r in window_results[h]:
        print(f"  {r['train_end']:>10} {r['test_year']:>8} {r['n_train']:>8} {r['n_test']:>6} "
              f"{r['gap_pct']:>+7.1f}% {r['cw_stat']:>8.2f} {r['cw_pvalue']:>10.2e}")
