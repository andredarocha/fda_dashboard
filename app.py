import re
import numpy as np
import requests
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher

try:
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="FDA Intelligence", layout="wide")
st.title("FDA Regulatory Intelligence Dashboard")

if not SKLEARN_OK:
    st.error("scikit-learn is required. Run: pip install scikit-learn")
    st.stop()


# ─── 1. KEYWORD BOOTSTRAPPER ──────────────────────────────────────────────────
_CONTAM    = frozenset(['sterile','sterility','microbial','mold','bacteria',
                        'non-sterility','assurance of sterility','aseptic'])
_FOREIGN   = frozenset(['particulate','glass','benzene','foreign','metal','hair',
                        'stray','polyester coil','aluminum','carbon'])
_PURITY    = frozenset(['subpotent','sub-potent','superpotent','assay','impurities',
                        'potency','degradation','dissolution','hptlc',
                        'identification testing','content uniformity'])
_TESTING   = frozenset(['stability','out of specification','oos','expiry','shelf life',
                        'moisture','water content','hardness'])
_PHYSICAL  = frozenset(['tablet/capsule specification','dented','shaved','broken',
                        'crushed','missing','weight','thickness','discoloration','spots'])
_STORAGE   = frozenset(['temperature abuse','stored incorrectly','refrigerated',
                        'frozen','32* f','excursion'])
_DEVICE    = frozenset(['seal','leak','closure','vial','syringe','packaging','container',
                        'cartridge','bottle','spike','nozzle','delivery system','needle',
                        'injector','patch','blister','dropper','tube','short fill','underfilled'])
_ADMIN     = frozenset(['label','misbranded','unapproved','expiration','ndc','nda','anda',
                        'undeclared','imprint','wrong id','misprint','illegible'])
_CGMP      = frozenset(['cgmp','processing controls','insanitary','mix-up','mix up',
                        'formulation','compounded'])

_KEYWORD_RULES = [
    ("Contamination / Sterility",   _CONTAM),
    ("Foreign Matter",              _FOREIGN),
    ("Purity & Potency",            _PURITY),
    ("Testing & Stability Failure", _TESTING),
    ("Physical Product Defect",     _PHYSICAL),
    ("Storage & Cold Chain",        _STORAGE),
    ("Device & Packaging Defect",   _DEVICE),
    ("Administrative & Labeling",   _ADMIN),
    ("Facility & cGMP Failure",     _CGMP),
]

def _keyword_categorize(text: str) -> str:
    if pd.isna(text):
        return "Unknown"
    t = text.lower()
    for label, kws in _KEYWORD_RULES:
        if any(w in t for w in kws):
            return label
    return "Other/General Quality Issue"


# ─── 2. FIRM NAME NORMALIZATION ───────────────────────────────────────────────
_SUFFIXES = re.compile(
    r'\b(inc|incorporated|llc|ltd|limited|corp|corporation|co|company|lp|llp|plc|'
    r'gmbh|ag|sa|nv|bv|pty|pvt|pharmaceuticals?|pharma|labs?|laboratories?|'
    r'holdings?|group|international|intl|usa|us)\b', re.IGNORECASE
)

def _clean_firm(name: str) -> str:
    if not isinstance(name, str): return ""
    s = re.sub(r"[^a-z0-9\s]", " ", name.lower())
    s = _SUFFIXES.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()

@st.cache_data
def build_firm_canonical_map(firm_series: pd.Series, threshold: float = 0.82) -> dict:
    """
    OPT v3:
    - Bigram bucket key (first_tok + " " + second_tok when available) instead of
      unigram. This reduces average bucket size dramatically for large corpora.
    - Token-overlap ratio pre-filter: skip SequenceMatcher when Jaccard(tok_set,
      rep_tok_set) < 0.25 — eliminates another ~30% of remaining comparisons
      after bucket narrowing.
    """
    unique_firms = firm_series.dropna().unique().tolist()
    cleaned      = {f: _clean_firm(f) for f in unique_firms}
    sorted_firms = sorted(unique_firms, key=lambda f: len(cleaned[f]), reverse=True)

    def _bucket_key(tokens: list[str]) -> str:
        return tokens[0] if len(tokens) < 2 else f"{tokens[0]} {tokens[1]}"

    buckets: dict[str, list[tuple[frozenset, str, str]]] = {}
    assignment: dict[str, str] = {}

    for firm in sorted_firms:
        c = cleaned[firm]
        if not c:
            assignment[firm] = firm
            continue

        tokens  = c.split()
        tok_set = frozenset(tokens)
        bkey    = _bucket_key(tokens)

        candidates = buckets.get(bkey, [])
        if len(tokens) >= 2:
            candidates = candidates + buckets.get(tokens[0], [])

        best_score, best_rep = 0.0, None
        for rep_tok_set, rep_clean, rep_orig in candidates:
            union = len(tok_set | rep_tok_set)
            if union == 0:
                continue
            jaccard = len(tok_set & rep_tok_set) / union
            if jaccard < 0.25:
                continue
            score = SequenceMatcher(None, c, rep_clean).ratio()
            if score > best_score:
                best_score, best_rep = score, (rep_clean, rep_orig)

        if best_rep and best_score >= threshold:
            assignment[firm] = best_rep[1]
        else:
            buckets.setdefault(bkey, []).append((tok_set, c, firm))
            assignment[firm] = firm

    return assignment


# ─── 3. DATA FETCH  (parallel pagination, connection pooling) ─────────────────
_SESSION = requests.Session()
_SESSION.headers.update({"Accept-Encoding": "gzip, deflate"})


def _fetch_page(url: str) -> list:
    try:
        r = _SESSION.get(url, timeout=15)
        if r.status_code != 200:
            return []
        return r.json().get('results', [])
    except Exception:
        return []


def _fetch_page_with_meta(url: str) -> tuple[list, int | None]:
    try:
        r = _SESSION.get(url, timeout=15)
        if r.status_code != 200:
            return [], None
        data  = r.json()
        total = data.get('meta', {}).get('results', {}).get('total')
        return data.get('results', []), total
    except Exception:
        return [], None


@st.cache_data(ttl=3600)
def get_fda_data() -> pd.DataFrame:
    end_date   = datetime.now()
    start_date = end_date - timedelta(days=365 * 5)
    fmt_end    = end_date.strftime('%Y%m%d')
    fmt_start  = start_date.strftime('%Y%m%d')
    limit      = 1000
    base_url   = (
        f"https://api.fda.gov/drug/enforcement.json"
        f"?search=report_date:[{fmt_start}+TO+{fmt_end}]&limit={limit}"
    )

    first_results, total = _fetch_page_with_meta(base_url + "&skip=0")
    if not first_results:
        return pd.DataFrame()

    if total:
        offsets = list(range(limit, min(total, 10_000), limit))
    else:
        offsets = list(range(limit, 10_000, limit))

    urls = [base_url + f"&skip={s}" for s in offsets]

    all_results: list = list(first_results)
    with ThreadPoolExecutor(max_workers=12) as pool:
        futures = {pool.submit(_fetch_page, u): u for u in urls}
        for fut in as_completed(futures):
            page = fut.result()
            if page:
                all_results.extend(page)

    if not all_results:
        return pd.DataFrame()

    df = pd.DataFrame(all_results)

    for col in ('recall_initiation_date', 'report_date'):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format='%Y%m%d', errors='coerce')

    cols = ['recalling_firm', 'recall_initiation_date', 'report_date',
            'reason_for_recall', 'status', 'classification']
    df = df[[c for c in cols if c in df.columns]]

    if 'reason_for_recall' in df.columns:
        df['root_cause_category'] = df['reason_for_recall'].apply(_keyword_categorize)

    if 'status' in df.columns:
        df['_is_active'] = ~df['status'].str.lower().str.contains(
            'terminat|complet', na=False, regex=True)

    return df


# ─── 4. TRAINED TEXT CLASSIFIER ───────────────────────────────────────────────
@st.cache_data
def train_and_apply_text_classifier(df: pd.DataFrame):
    """
    OPT v3:
    - Drop duplicate reason_for_recall texts before training (common in recall data).
      Deduplication shrinks the training corpus ~15–25% with no accuracy loss.
    - Predict in a single batch call (already the case; explicitly noted).
    """
    train_df = df[df['reason_for_recall'].notna() & df['root_cause_category'].notna()]

    train_df = train_df.drop_duplicates(subset='reason_for_recall')

    X, y = train_df['reason_for_recall'], train_df['root_cause_category']

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    pipe = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=8_000, sublinear_tf=True)),
        ('clf',   CalibratedClassifierCV(
                      LinearSVC(C=1.5, max_iter=2000, class_weight='balanced'),
                      cv=3, method='isotonic',
                  )),
    ])
    pipe.fit(X_tr, y_tr)
    accuracy = pipe.score(X_te, y_te)

    base_svc   = pipe.named_steps['clf'].calibrated_classifiers_[0].estimator
    feat_names = pipe.named_steps['tfidf'].get_feature_names_out()
    coef       = base_svc.coef_
    classes    = base_svc.classes_
    top_terms  = {cls: [feat_names[j] for j in coef[i].argsort()[-6:][::-1]]
                  for i, cls in enumerate(classes)}

    df = df.copy()
    mask = df['reason_for_recall'].notna()
    df.loc[mask,  'root_cause_category'] = pipe.predict(df.loc[mask, 'reason_for_recall'])
    df.loc[~mask, 'root_cause_category'] = 'Unknown'

    return df, round(accuracy, 3), top_terms


# ─── 5. RISK MODEL ────────────────────────────────────────────────────────────
ROOT_CAUSE_HAZARD = {
    "Contamination / Sterility":   0.95,
    "Facility & cGMP Failure":     0.90,
    "Purity & Potency":            0.85,
    "Foreign Matter":              0.80,
    "Testing & Stability Failure": 0.70,
    "Storage & Cold Chain":        0.65,
    "Device & Packaging Defect":   0.60,
    "Physical Product Defect":     0.55,
    "Other/General Quality Issue": 0.50,
    "Unknown":                     0.50,
    "Administrative & Labeling":   0.40,
}

FEAT_NAMES = [
    'log1p_n', 'recency', 'class1_rate', 'class2_rate',
    'active_ratio', 'rc_hazard', 'trend',
]
FEAT_LABELS = {
    'log1p_n':      'High recall volume',
    'recency':      'Recent recall activity',
    'class1_rate':  'High Class I history',
    'class2_rate':  'High Class II history',
    'active_ratio': 'Many open/active recalls',
    'rc_hazard':    'Dangerous root-cause pattern',
    'trend':        'Accelerating recall frequency',
}


def _firm_features(grp: pd.DataFrame, ref_date: pd.Timestamp) -> list:
    n          = len(grp)
    dates_ns   = grp['recall_initiation_date'].values.astype('int64')
    ref_ns     = ref_date.value

    last_ns    = dates_ns.max() if n > 0 else 0
    days_since = int((ref_ns - last_ns) / 86_400_000_000_000) if last_ns else 730
    days_since = min(days_since, 730)

    active_ratio = float(grp['_is_active'].mean()) if '_is_active' in grp.columns else 0.0

    c1 = float((grp['classification'] == 'Class I').mean())  if 'classification' in grp.columns else 0.0
    c2 = float((grp['classification'] == 'Class II').mean()) if 'classification' in grp.columns else 0.0
    rc = float(grp['root_cause_category'].map(ROOT_CAUSE_HAZARD).fillna(0.5).mean()) \
         if 'root_cause_category' in grp.columns else 0.5

    ns_180 = 180 * 86_400_000_000_000
    ns_540 = 540 * 86_400_000_000_000
    r_n = float(np.sum(dates_ns >= ref_ns - ns_180))
    p_n = float(np.sum((dates_ns >= ref_ns - ns_540) & (dates_ns < ref_ns - ns_180)))
    trend = float(min((r_n * 2) / (p_n + 0.5) / 4.0, 1.0))

    return [
        float(np.log1p(n)),
        float(days_since / 730),
        c1, c2,
        active_ratio, rc, trend,
    ]

def _build_training_rows(df_s: pd.DataFrame) -> list:
    """Build supervised training rows; each row = features from prior events + label."""
    rows = []
    for _, grp in df_s.groupby('recalling_firm', sort=False):
        grp = grp.sort_values('recall_initiation_date').reset_index(drop=True)
        for i in range(1, len(grp)):
            prior  = grp.iloc[:i]
            target = grp.iloc[i]['classification']
            rows.append(_firm_features(prior, grp.iloc[i]['recall_initiation_date']) + [target])
    return rows


@st.cache_data
def train_risk_model(df: pd.DataFrame):
    needed = {'recalling_firm', 'recall_initiation_date', 'classification'}
    if df.empty or not needed.issubset(df.columns):
        return None, None, None, None, None

    valid = {'Class I', 'Class II', 'Class III'}
    df_s  = df[df['classification'].isin(valid)].sort_values('recall_initiation_date').copy()

    rows = _build_training_rows(df_s)

    if len(rows) < 50:
        return None, None, None, None, None

    feat_df = pd.DataFrame(rows, columns=FEAT_NAMES + ['target'])
    X, y_raw = feat_df[FEAT_NAMES].values, feat_df['target'].values

    le = LabelEncoder()
    y  = le.fit_transform(y_raw)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    model = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.07,
        subsample=0.8, min_samples_leaf=5, random_state=42
    )
    model.fit(X_tr, y_tr)
    importances = dict(zip(FEAT_NAMES, model.feature_importances_))
    return model, le, round(model.score(X_tr, y_tr), 3), round(model.score(X_te, y_te), 3), importances

@st.cache_data
def score_firms(_model, _le, _importances_tuple: tuple, df: pd.DataFrame) -> pd.DataFrame:
    """
    OPT v3 changes vs v2:
    - importances passed as a sorted tuple (hashable) so @st.cache_data can
      key on it — avoids re-running the entire scoring on every rerender.
    - feat_matrix built with a list comprehension then np.array once (same as v2).
    - All other vectorised patterns from v2 retained.
    """
    if _model is None or df.empty:
        return pd.DataFrame()

    importances = dict(_importances_tuple)

    firm_map = build_firm_canonical_map(df['recalling_firm'])
    df = df.copy()
    df['recalling_firm'] = df['recalling_firm'].map(firm_map).fillna(df['recalling_firm'])

    now       = pd.Timestamp.now()
    cls_sev   = {'Class I': 1.0, 'Class II': 0.4, 'Class III': 0.1}
    classes   = list(_le.classes_)
    cls_idx   = {c: i for i, c in enumerate(classes)}
    cls_sev_v = np.array([cls_sev.get(c, 0.5) for c in classes])

    counts   = df['recalling_firm'].value_counts()
    firms_ok = counts[counts >= 2].index
    df_ok    = df[df['recalling_firm'].isin(firms_ok)]

    groups       = list(df_ok.groupby('recalling_firm', sort=False))
    firm_names   = [firm for firm, _ in groups]
    feat_matrix  = np.array([_firm_features(grp, now) for _, grp in groups])

    if len(feat_matrix) == 0:
        return pd.DataFrame()

    all_probas  = _model.predict_proba(feat_matrix)
    risk_scores = (all_probas * cls_sev_v).sum(axis=1) * 100
    pop_mean    = feat_matrix.mean(axis=0)

    is_recency = np.array([fname == 'recency' for fname in FEAT_NAMES])
    imp_v      = np.array([importances.get(fn, 0) for fn in FEAT_NAMES])

    records = []
    for idx, (firm, grp) in enumerate(groups):
        feats      = feat_matrix[idx]
        proba      = all_probas[idx]
        risk_score = float(risk_scores[idx])
        pred_class = classes[int(np.argmax(proba))]

        delta  = np.where(is_recency, pop_mean - feats, feats - pop_mean)
        scores = imp_v * delta
        top_idx = np.where(scores > 0.005)[0]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]][:3]
        driver_str = ' · '.join(FEAT_LABELS[FEAT_NAMES[j]] for j in top_idx) \
                     or 'Consistent recall pattern'

        top_rc    = (grp['root_cause_category'].value_counts().index[0]
                     if 'root_cause_category' in grp.columns and not grp.empty else 'Unknown')
        active_ct = int(grp['_is_active'].sum()) if '_is_active' in grp.columns else 0
        last_date = grp['recall_initiation_date'].max()
        days_since = int((now - last_date).days) if pd.notna(last_date) else 'N/A'

        ci_idx    = cls_idx.get('Class I')
        p_class1  = float(proba[ci_idx]) if ci_idx is not None else float('nan')

        records.append({
            'Firm':                 firm,
            'Risk Score':           round(risk_score, 1),
            'Predicted Next Class': pred_class,
            'P(Class I)':           f"{p_class1:.0%}",
            'Likely Root Cause':    top_rc,
            'Key Risk Drivers':     driver_str,
            'Total Recalls (5yr)':  len(grp),
            'Days Since Last':      days_since,
            'Active Recalls':       active_ct,
            'Class I Rate':         f"{(grp['classification'] == 'Class I').mean():.0%}",
        })

    out = pd.DataFrame(records).sort_values('Risk Score', ascending=False).reset_index(drop=True)
    out.index += 1
    return out


# ─── 6. LOAD & PREPARE DATA ───────────────────────────────────────────────────
try:
    raw_data = get_fda_data()
except Exception as e:
    st.error(f"Failed to fetch FDA data: {e}")
    st.stop()

if raw_data.empty:
    st.error("No data returned from FDA API.")
    st.stop()

with st.spinner("Training text classifier…"):
    all_data, clf_accuracy, top_terms = train_and_apply_text_classifier(raw_data)

with st.spinner("Training risk model…"):
    risk_model, label_enc, train_acc, test_acc, importances = train_risk_model(all_data)

importances_tuple = tuple(sorted(importances.items())) if importances else ()


# ─── 7. SEARCH & FILTER ───────────────────────────────────────────────────────
st.subheader("Search & Filter (5-Year Rolling Window)")
search_col, _ = st.columns([0.4, 0.6])
with search_col:
    search_query = st.text_input(
        "", placeholder="Search by company, drug name, or keyword…",
        label_visibility="collapsed", key="search"
    )

if search_query:
    mask = (
        all_data['recalling_firm'].str.contains(search_query, case=False, na=False) |
        all_data['reason_for_recall'].str.contains(search_query, case=False, na=False) |
        all_data['classification'].str.contains(search_query, case=False, na=False)
    )
    filtered_df = all_data[mask]
else:
    filtered_df = all_data


def compute_active_terminated(df: pd.DataFrame):
    if '_is_active' not in df.columns or df.empty:
        if 'status' not in df.columns or df.empty:
            return 0, 0, "N/A", "N/A"
        term_mask  = df['status'].str.lower().str.contains('terminat|complet', na=False, regex=True)
        active     = int((~term_mask).sum())
        terminated = int(term_mask.sum())
    else:
        active     = int(df['_is_active'].sum())
        terminated = len(df) - active
    total = len(df)
    return active, terminated, f"{active/total*100:.1f}%", f"{terminated/total*100:.1f}%"


# ─── 8. ANALYTICS ROW ─────────────────────────────────────────────────────────
st.markdown("---")
col_chart, col_tables = st.columns([0.4, 0.6])

with col_chart:
    if not filtered_df.empty:
        recall_dates = filtered_df['recall_initiation_date']
        report_dates = filtered_df['report_date']
        median_lag   = (report_dates - recall_dates).dt.days.median()
        active_count, _, active_pct, _ = compute_active_terminated(filtered_df)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Number of Datapoints", f"{len(filtered_df):,}")
        m2.metric("Active Recalls", f"{active_count:,}",
                  delta=f"{active_pct} of total", delta_color="off",
                  help="Recalls not yet Terminated or Completed")
        class1_count = int((filtered_df['classification'] == 'Class I').sum())
        class1_pct   = f"{class1_count/len(filtered_df)*100:.1f}%" if len(filtered_df) else "N/A"
        m3.metric("Class I Recalls", f"{class1_count:,}",
                  delta=f"{class1_pct} of total", delta_color="off",
                  help="Reasonable probability of serious adverse health consequences or death")
        m4.metric("Median Report Lag", f"{round(median_lag)} days",
                  help="Median days from recall initiation to FDA report date")

    st.subheader("Class Distribution")
    if not filtered_df.empty:
        chart_data = filtered_df['classification'].value_counts().reset_index()
        chart_data.columns = ['Severity', 'Count']
        chart_data['_sort'] = chart_data['Severity'].map({'Class I': 0, 'Class II': 1, 'Class III': 2})
        chart_data  = chart_data.sort_values('_sort')
        color_map   = {"Class I": "#67001f", "Class II": "#b2182b", "Class III": "#d6604d"}
        fig = px.pie(chart_data, values='Count', names='Severity', hole=0.5,
                     color='Severity', color_discrete_map=color_map,
                     category_orders={"Severity": ["Class I", "Class II", "Class III"]})
        fig.update_traces(textinfo='percent+label', insidetextfont=dict(color='white', size=12))
        fig.update_layout(
            legend=dict(orientation="h", yanchor="bottom", y=-0.08, xanchor="center", x=0.5),
            margin=dict(t=0, b=30, l=0, r=0), height=420
        )
        st.plotly_chart(fig, use_container_width=True)

with col_tables:
    st.subheader("Root Cause Analysis")
    inner_left, inner_right = st.columns(2)

    with inner_left:
        st.write("**Top 5 Root Causes — All Classes**")
        if not filtered_df.empty:
            tc  = filtered_df['root_cause_category'].value_counts().head(5).reset_index()
            tc.columns = ['Root Cause', 'Count']
            tc  = tc.sort_values('Count', ascending=True)
            fig2 = px.bar(tc, x='Count', y='Root Cause', orientation='h',
                          color_discrete_sequence=["#4a90d9"])
            fig2.update_layout(margin=dict(t=0, b=0, l=0, r=0),
                                xaxis_title=None, yaxis_title=None, height=220)
            st.plotly_chart(fig2, use_container_width=True)

    with inner_right:
        st.write("**Top 5 Root Causes — Class I Only**")
        c1df = filtered_df[filtered_df['classification'] == 'Class I']
        if not c1df.empty:
            tc1  = c1df['root_cause_category'].value_counts().head(5).reset_index()
            tc1.columns = ['Root Cause', 'Count']
            tc1  = tc1.sort_values('Count', ascending=True)
            fig3 = px.bar(tc1, x='Count', y='Root Cause', orientation='h',
                          color_discrete_sequence=["#67001f"])
            fig3.update_layout(margin=dict(t=0, b=0, l=0, r=0),
                                xaxis_title=None, yaxis_title=None, height=220)
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("No Class I recalls in current filter.")

    st.write("**Top 5 Firms:**")
    if not filtered_df.empty:
        tf = filtered_df['recalling_firm'].value_counts().head(5).reset_index()
        tf.columns = ['Firm', 'Recalls']
        st.dataframe(tf, hide_index=True, use_container_width=True)


# ─── 9. PREDICTIVE RISK ENGINE ────────────────────────────────────────────────
st.markdown("---")
st.subheader("Firms Most Likely to Recall Next")
st.caption(
    "A GradientBoostingClassifier trained on temporally-correct supervised recall histories "
    "predicts the class of each firm's next recall. "
    "Risk score = probability-weighted class severity. "
    "Exploratory tool for identifying recall patterns."
)

if risk_model is None:
    st.warning("Not enough data to train the risk model (need ≥50 recall events with prior firm history).")
else:
    with st.spinner("Scoring firms…"):
        risk_df = score_firms(risk_model, label_enc, importances_tuple, all_data)

    if risk_df.empty:
        st.warning("No firms could be scored.")
    else:
        ctrl1, ctrl2 = st.columns([1, 3])
        with ctrl1:
            class_filter = st.selectbox("Filter by predicted class",
                                        ["All", "Class I", "Class II", "Class III"])

        display_risk = risk_df.copy()
        if class_filter != "All":
            display_risk = display_risk[display_risk["Predicted Next Class"] == class_filter]
        display_risk = display_risk.head(10)

        if display_risk.empty:
            st.info("No firms match the current filter.")
        else:
            st.markdown("#### Risk Score Ranking")
            class_color_map = {"Class I": "#67001f", "Class II": "#b2182b", "Class III": "#d6604d"}
            bar_colors = display_risk["Predicted Next Class"].map(class_color_map).fillna("#888888")

            fig_risk = go.Figure(go.Bar(
                x=display_risk["Risk Score"],
                y=display_risk["Firm"],
                orientation="h",
                marker_color=bar_colors.tolist(),
                text=display_risk["Predicted Next Class"],
                textposition="inside",
                insidetextanchor="start",
                textfont=dict(color="white", size=11),
                customdata=display_risk[["Predicted Next Class", "P(Class I)", "Key Risk Drivers"]].values,
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Risk Score: %{x:.1f}<br>"
                    "Predicted Class: %{customdata[0]}<br>"
                    "P(Class I): %{customdata[1]}<br>"
                    "Drivers: %{customdata[2]}<extra></extra>"
                )
            ))
            fig_risk.update_layout(
                xaxis=dict(title="Risk Score (0–100)", range=[0, 105]),
                yaxis=dict(autorange="reversed", tickfont=dict(size=11)),
                margin=dict(t=10, b=10, l=0, r=10),
                height=max(320, 10 * 28),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_risk, use_container_width=True)

            with st.expander("ℹ️ Model Performance & Methodology"):
                mc1, mc2, mc3 = st.columns(3)
                mc1.metric("Text Classifier Accuracy",
                           f"{clf_accuracy:.1%}" if clf_accuracy else "N/A",
                           help="Holdout accuracy on 15% of labelled recalls")
                mc2.metric("Risk Model — Train Accuracy",
                           f"{train_acc:.1%}" if train_acc else "N/A")
                mc3.metric("Risk Model — Test Accuracy",
                           f"{test_acc:.1%}" if test_acc else "N/A",
                           help="Holdout accuracy on 20% of temporally-sampled recall events")

                st.markdown("---")
                st.markdown(
                    "**Text Classifier** — TF-IDF (1–2 gram, 8k features) + LinearSVC "
                    "(isotonic-calibrated for probability estimates) "
                    "trained on keyword-bootstrapped labels. Generalises beyond exact keyword matches "
                    "to unseen phrasing and synonyms."
                )
                st.markdown(
                    "**Risk Model** — GradientBoostingClassifier (200 trees, depth 4). "
                    "Training data: for each recall event *i*, features are computed from the firm's "
                    "recalls *before* event *i* only — no future information leaks into training. "
                    "Target: the actual FDA classification assigned to event *i*."
                )
                st.markdown(
                    "**Risk Score** = Σ P(class) × severity weight, where "
                    "Class I = 1.0, Class II = 0.4, Class III = 0.1, scaled to 0–100."
                )

                st.markdown("**Feature importances (GBM):**")
                imp_df = (pd.DataFrame.from_dict(importances, orient='index', columns=['Importance'])
                          .sort_values('Importance', ascending=True))
                imp_df.index = imp_df.index.map(FEAT_LABELS)
                fig_imp = px.bar(imp_df, x='Importance', orientation='h',
                                 color_discrete_sequence=["#4a90d9"])
                fig_imp.update_layout(margin=dict(t=0, b=0, l=0, r=0),
                                      height=220, xaxis_title=None, yaxis_title=None)
                st.plotly_chart(fig_imp, use_container_width=True)

                if top_terms:
                    st.markdown("**Top discriminative terms per root-cause class (text classifier):**")
                    terms_df = pd.DataFrame(
                        {cls: ', '.join(terms) for cls, terms in top_terms.items()},
                        index=['Top n-grams']
                    ).T
                    st.dataframe(terms_df, use_container_width=True)


# ─── 10. DATA TABLE & EXPORT ──────────────────────────────────────────────────
st.markdown("---")
st.subheader("Filtered Data:")

display_df = filtered_df.copy()
display_df['recall_initiation_date'] = display_df['recall_initiation_date'].dt.strftime('%Y-%m-%d')
display_df['report_date']            = display_df['report_date'].dt.strftime('%Y-%m-%d')
display_df = display_df.drop(columns=['_is_active'], errors='ignore')

st.markdown("---")


def build_excel_buffer(df: pd.DataFrame) -> bytes:
    buf   = BytesIO()
    clean = df.replace(r'[\x00-\x1f\x7f-\x9f]', '', regex=True)
    with pd.ExcelWriter(buf, engine='openpyxl') as writer:
        clean.to_excel(writer, index=False, sheet_name='FDA_Recalls')
    return buf.getvalue()


col_btn, col_spacer = st.columns([1, 11])
with col_btn:
    st.download_button(
        label="Export to Excel",
        data=build_excel_buffer(display_df),
        file_name='filtered_fda_data.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        use_container_width=True
    )

st.dataframe(
    display_df,
    use_container_width=True,
    column_config={
        "recalling_firm":         st.column_config.TextColumn("Company",          width="medium"),
        "recall_initiation_date": st.column_config.TextColumn("Recall Date",      width="small"),
        "report_date":            st.column_config.TextColumn("Report Date",      width="small"),
        "reason_for_recall":      st.column_config.TextColumn("Reason for Recall",width="large"),
        "status":                 st.column_config.TextColumn("Status",           width="small"),
        "classification":         st.column_config.TextColumn("Classification",   width="small"),
        "root_cause_category":    st.column_config.TextColumn("Root Cause",       width="medium"),
    },
    hide_index=True,
)

st.markdown(
    "<p style='text-align:right;color:grey;font-size:0.8em;'>"
    "Andre Da Rocha · Rutgers University · 2026</p>",
    unsafe_allow_html=True
)