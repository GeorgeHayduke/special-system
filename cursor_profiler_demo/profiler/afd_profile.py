"""
Data profiler for tabular fraud-detection datasets.

Adapted from aws-samples/aws-fraud-detector-samples
(profiler/ManualNotebookSolution/afd_profile.py). The original ships as a
notebook-only tool with no CLI entrypoint, no tests, and two calls to
DataFrame.append() -- an API removed in pandas 2.0. Those are left intact
here on purpose: this is the "before" state for the Cursor tutorial.

Known issues (see specs/ and .cursor/rules/profiler.mdc):
  1. check_if_datetime_as_object_feature() calls DataFrame.append() in a
     loop -- raises AttributeError on pandas>=2.0.
  2. get_stats() calls DataFrame.append() once, for the synthetic
     _EMAIL_DOMAIN column -- same issue.
  3. No __main__ / argparse entrypoint. Everything below is only
     reachable by importing the module from a notebook or script.
  4. No tests exist for any of this.
"""

import itertools
import json
import logging

import numpy as np
import pandas as pd
import scipy.stats as ss
from jinja2 import Environment, FileSystemLoader, select_autoescape
from pandas import DataFrame, Series

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s :: %(asctime)s.%(msecs)03d :: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

DATE_FORMATS = [
    "%m/%d/%Y",
    "%m/%d/%Y",
    "%Y/%m/%d",
    "%m-%d-%Y",
    "%m-%d-%Y",
    "%Y-%m-%d",
]

TIME_FORMATS = [
    "%I:%M:%S.%f %p",
    "%I:%M:%S %p",
    "%I:%M %p",
    "%H:%M:%S.%f",
    "%H:%M:%S",
    "%H:%M",
]

SPECIAL_FORMATS = ["%Y-%m-%dT%H:%M:%S.%f"]

SUPPORTED_FORMATS = []
SUPPORTED_FORMATS += DATE_FORMATS
SUPPORTED_FORMATS += [f"{d} {t}" for d, t in itertools.product(DATE_FORMATS, TIME_FORMATS)]
SUPPORTED_FORMATS += SPECIAL_FORMATS

# Kept consistent with the AWS Fraud Detector validation container.
EMAIL_REGEX = (
    r"""(?:[a-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*"""
    r"""|"(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21\x23-\x5b\x5d-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])*")"""
    r"""@(?:(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?"""
    r"""|\[(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}"""
    r"""(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?|[a-z0-9-]*[a-z0-9]:"""
    r"""(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21-\x5a\x53-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])+)\])"""
)

IP_REGEX = (
    r"((^\s*((([0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])\.){3}"
    r"([0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5]))\s*$))"
)


def get_type_family_raw(dtype) -> str:
    """From dtype, gets the dtype family."""
    try:
        if dtype.name == "category":
            return "category"
        if "datetime" in dtype.name:
            return "datetime"
        elif np.issubdtype(dtype, np.integer):
            return "int"
        elif np.issubdtype(dtype, np.floating):
            return "float"
    except Exception:
        logging.info(f"Warning: dtype {dtype} is not recognized as a valid dtype by numpy!")

    if dtype.name in ["bool", "bool_"]:
        return "bool"
    elif dtype.name in ["str", "string", "object"]:
        return "object"
    else:
        return dtype.name


def get_type_map_raw(df: DataFrame) -> dict:
    features_types = df.dtypes.to_dict()
    return {k: get_type_family_raw(v) for k, v in features_types.items()}


def get_type_map_special(X: DataFrame) -> dict:
    type_map_special = {}
    for column in X:
        type_special = get_type_special(X[column])
        if type_special is not None:
            type_map_special[column] = type_special
    return type_map_special


def get_type_special(X: Series) -> str:
    if check_if_datetime_as_object_feature(X):
        type_special = "datetime"
    elif check_if_nlp_feature(X):
        type_special = "text"
    elif check_if_regex_feature(X, EMAIL_REGEX):
        type_special = "EMAIL_ADDRESS"
    elif check_if_regex_feature(X, IP_REGEX):
        type_special = "IP_ADDRESS"
    else:
        type_special = None
    return type_special


def check_if_datetime_as_object_feature(X: Series) -> bool:
    """Samples up to 5,000 rows and tries every format in SUPPORTED_FORMATS.

    KNOWN BUG: builds the mismatch-rate table with DataFrame.append(),
    which was removed in pandas 2.0. Raises AttributeError there.
    """
    type_family = get_type_family_raw(X.dtype)

    if X.isnull().all():
        return False
    if type_family != "object":
        return False
    try:
        pd.to_numeric(X)
    except Exception:
        try:
            if len(X) > 5000:
                X = X.sample(n=5000, random_state=0)

            result = pd.DataFrame(columns=["format", "mismatch_rate"])
            for fm in SUPPORTED_FORMATS:
                result = result.append(
                    {
                        "format": fm,
                        "mismatch_rate": pd.to_datetime(X, format=fm, errors="coerce").isnull().mean(),
                    },
                    ignore_index=True,
                )
            if result["mismatch_rate"].min() > 0.8:
                return False
            return True
        except Exception:
            return False


def check_if_nlp_feature(X: Series) -> bool:
    """Flags object columns with >1% unique values and 3+ average words as text.

    Multi-word categorical labels (e.g. three-word merchant categories) can
    trip this heuristic and get misclassified as free text -- see
    sample_data/transactions.csv::merchant_category for a reproduction.
    """
    type_family = get_type_family_raw(X.dtype)
    if type_family != "object":
        return False
    if len(X) > 5000:
        X = X.sample(n=5000, random_state=0)
    X_unique = X.unique()
    num_unique = len(X_unique)
    num_rows = len(X)
    unique_ratio = num_unique / num_rows
    if unique_ratio <= 0.01:
        return False
    try:
        avg_words = Series(X_unique).str.split().str.len().mean()
    except AttributeError:
        return False
    if avg_words < 3:
        return False
    return True


def check_if_regex_feature(X: Series, regex: str) -> bool:
    dtype = get_type_family_raw(X.dtype)
    if dtype not in ["category", "object"]:
        return False
    X = X.dropna()
    if len(X) > 100:
        X = X.sample(n=100, random_state=0)
    if len(X) == 0:
        return False
    return bool(X.str.match(regex).all())


def get_config(config_path: str) -> dict:
    """Convert a JSON config file (a single-element list containing one
    dict) into a plain python dict."""
    with open(config_path, "r") as f:
        config_dict = json.load(f)[0]
    return config_dict


def get_dataframe(config: dict) -> DataFrame:
    """Load CSV into a pandas DataFrame, or pass one through unchanged."""
    if isinstance(config["input_file"], pd.DataFrame):
        return config["input_file"]
    return pd.read_csv(config["input_file"], low_memory=False)


def get_overview(config: dict, df: DataFrame):
    """Returns dataset-level stats and any high-level issues found."""
    overview_msg = {}
    df = df.copy()
    label = config["required_features"]["EVENT_LABEL"]
    df.loc[~df[label].isna(), label] = df.loc[~df[label].isna(), label].astype(str)

    column_cnt = len(df.columns)
    _timestamp_col = pd.to_datetime(df[config["required_features"]["EVENT_TIMESTAMP"]], errors="coerce").dropna()
    if _timestamp_col.shape[0] > 0:
        date_range = _timestamp_col.min().strftime("%Y-%m-%d") + " to " + _timestamp_col.max().strftime("%Y-%m-%d")
        day_cnt = (_timestamp_col.max() - _timestamp_col.min()).days
    else:
        date_range = ""
        day_cnt = 0

    record_cnt = df.shape[0]
    memory_size = df.memory_usage(index=True).sum()
    record_size = round(float(memory_size) / record_cnt, 2)
    n_dupe = record_cnt - len(df.drop_duplicates())

    if record_cnt <= 10000:
        overview_msg["Record count"] = (
            "A minimum of 10,000 rows are required to train the model, your dataset contains " + str(record_cnt)
        )

    overview_stats = {
        "Record count": "{:,}".format(record_cnt),
        "Column count": "{:,}".format(column_cnt),
        "Duplicate count": "{:,}".format(n_dupe),
        "Memory size": "{:.2f}".format(memory_size / 1024**2) + " MB",
        "Record size": "{:,}".format(record_size) + " bytes",
        "Date range": date_range,
        "Day count": "{:,}".format(day_cnt) + " days",
        "overview_msg": overview_msg,
        "overview_cnt": len(overview_msg),
        "Majority_label": f"""'{config['MAJORITY_CLASS']}'""",
        "Mapped_fraud": config["MAPPED_FRAUD"],
    }

    return df, overview_stats


def set_feature(row, config):
    message_uniq, message_null, action = "", "", ""
    required_features = config["required_features"]

    dtype_to_vtype_map = {
        "category": "CATEGORY",
        "object": "CATEGORY",
        "int": "NUMERIC",
        "float": "NUMERIC",
        "text": "TEXT",
        "datetime": "DATETIME",
        "EMAIL_ADDRESS": "EMAIL_ADDRESS",
        "IP_ADDRESS": "IP_ADDRESS",
        "PHONE_NUMBER": "PHONE_NUMBER",
    }

    feature = dtype_to_vtype_map[row._dtype]

    if row._column == required_features["EVENT_TIMESTAMP"]:
        feature = "EVENT_TIMESTAMP"
    if row._column == required_features["ORIGINAL_LABEL"]:
        feature = "EVENT_LABEL"

    if feature == "CATEGORY" and row["nunique"] < 2:
        message_uniq, action = "ONLY 1 UNIQUE VALUE", "EXCLUDE"
    elif feature in ["EMAIL_ADDRESS", "IP_ADDRESS"] and row["nunique"] < 100:
        message_uniq, action = "<100 UNIQUE VALUE", "EXCLUDE"

    if row.null_pct > 0.9 and feature not in ["EMAIL_ADDRESS", "IP_ADDRESS"]:
        message_null, action = ">90% MISSING", "EXCLUDE"
    elif row.null_pct > 0.75 and feature in ["EMAIL_ADDRESS", "IP_ADDRESS"]:
        message_null, action = ">75% MISSING", "EXCLUDE"
    elif row.null_pct > 0.2:
        message_null = ">20% MISSING"

    message = "; ".join([message_uniq, message_null, action]).lstrip(";")
    message = None if len(message) < 4 else message

    return feature, message


def get_label(config: dict, df: DataFrame):
    """Returns stats on the label column and performs initial label checks."""
    _original_label_col = config["required_features"]["ORIGINAL_LABEL"]
    _mapped_label_col = config["required_features"]["EVENT_LABEL"]
    _mapped_fraud = config["MAPPED_FRAUD"]

    missing_count = 0
    label_dict = {}
    for c in df[_original_label_col].unique():
        if pd.isna(c):
            _df = df[df[_original_label_col].isna()][_original_label_col]
            label_dict["Missing Labels"] = {
                "Name": "Missing Labels",
                "CLS": "Undefined",
                "Count": _df.shape[0],
                "Percentage": "{:.2f}%".format(100 * _df.shape[0] / df.shape[0]),
            }
            missing_count = _df.shape[0]
        else:
            _df = df[df[_original_label_col] == c][_original_label_col]
            if _mapped_fraud:
                _mapped_label = df[df[_original_label_col] == c][_mapped_label_col].iloc[0]
            else:
                _mapped_label = "Undefined"
            label_dict[c] = {
                "Name": c,
                "CLS": _mapped_label,
                "Count": _df.shape[0],
                "Percentage": "{:.2f}%".format(100 * _df.shape[0] / df.shape[0]),
            }

    label_dict = pd.DataFrame(label_dict).T
    if len(label_dict) > 0:
        label_dict = label_dict.sort_values("Count", ascending=False)

    message = {"message": ""}
    if df.shape[0] > 0 and missing_count / df.shape[0] >= 0.01:
        message["message"] = (
            f"Your {_original_label_col} column contains {missing_count} missing values. "
            "AFD requires less than 1% of the values in the label column to be missing."
        )
    message["length"] = len(message["message"])

    return label_dict, message


def rename_dtypes(x):
    if x == "int":
        return "INTEGER"
    elif x == "float":
        return "FLOAT"
    else:
        return "STRING"


def get_stats(config: dict, df: DataFrame):
    """Generates per-column analysis statistics; calls set_feature().

    KNOWN BUG: when an EMAIL_ADDRESS column is detected, the synthetic
    _EMAIL_DOMAIN row is appended to the dtype table with
    DataFrame.append() -- removed in pandas 2.0.
    """
    type_map_raw = get_type_map_raw(df)
    type_map_special = get_type_map_special(df)
    type_map_raw.update(type_map_special)
    dt = pd.DataFrame.from_dict(type_map_raw, orient="index").reset_index().rename(columns={"index": "_column", 0: "_dtype"})

    if "EMAIL_ADDRESS" in dt["_dtype"].values:
        email = dt[dt["_dtype"] == "EMAIL_ADDRESS"]["_column"].values.tolist()[0]
        df["_EMAIL_DOMAIN"] = df[email].str.split("@").str[1]
        dt = dt.append(pd.DataFrame([["_EMAIL_DOMAIN", "object"]], columns=["_column", "_dtype"]))

    rowcnt = len(df)
    df_s1 = df.agg(["count", "nunique"]).transpose().reset_index().rename(columns={"index": "_column"})
    df_s1["count"] = df_s1["count"].astype("int64")
    df_s1["nunique"] = df_s1["nunique"].astype("int64")
    df_s1["null"] = (rowcnt - df_s1["count"]).astype("int64")
    df_s1["not_null"] = rowcnt - df_s1["null"]
    df_s1["null_pct"] = df_s1["null"] / rowcnt
    df_s1["nunique_pct"] = df_s1["nunique"] / rowcnt

    df_stats = pd.merge(dt, df_s1, on="_column", how="inner")
    df_stats = df_stats.round(4)
    df_stats[["_feature", "_message"]] = df_stats.apply(lambda x: set_feature(x, config), axis=1, result_type="expand")

    df_stats["_dtype"] = df_stats["_dtype"].apply(rename_dtypes)
    df_stats = df_stats[df_stats["_column"] != "AFD_LABEL"]

    for d in df_stats.loc[df_stats["_feature"].isin(["EVENT_TIMESTAMP", "DATETIME"])]._column.tolist():
        df[d] = pd.to_datetime(df[d], errors="coerce")

    return df, df_stats, df_stats.dropna(subset=["_message"])


def col_stats(df, target, column, majority, min_class_count, labels):
    """Generates column statistics for a categorical column."""
    cat_summary = (
        df.groupby([column, target])[target]
        .count()
        .unstack(fill_value=0)
        .reset_index()
        .sort_values(majority, ascending=True)
        .reset_index(drop=True)
    )
    cat_summary["total"] = 0
    sort_orders = []
    for c in labels:
        if c in cat_summary.columns:
            sort_orders.append(c)
            cat_summary["total"] = cat_summary["total"] + cat_summary[c]
    for c in labels:
        if c in cat_summary.columns:
            cat_summary[c + "_pctg"] = cat_summary[c] / cat_summary["total"]

    cat_summary["pctg_minority"] = 1 - cat_summary[majority + "_pctg"]
    cat_summary = cat_summary[cat_summary["total"] > min_class_count]
    cat_summary["_non_majority"] = cat_summary["total"] - cat_summary[majority]

    cat_summary_total = cat_summary.sort_values(["total"] + sort_orders, ascending=False)
    cat_summary_majority = cat_summary.sort_values([majority] + sort_orders, ascending=False)
    cat_summary_non_majority = cat_summary.sort_values(["_non_majority"] + sort_orders[::-1], ascending=False)
    cat_summary_pctg = cat_summary.sort_values(
        ["pctg_minority"] + [item + "_pctg" for item in sort_orders[::-1]], ascending=False
    )

    return (
        cat_summary_total.round(4),
        cat_summary_majority.round(4),
        cat_summary_non_majority.round(4),
        cat_summary_pctg.round(4),
    )


def col_stats_to_dict(_df, col_name, labels, reverse=False):
    _rec = {"LABELS": _df[col_name].tolist()}
    _labels = labels.copy()
    if reverse:
        _labels = _labels[::-1]
    for c in _labels:
        if c in _df.columns:
            _rec[c] = _df[c].tolist()
            _rec[c + "_pctg"] = _df[c + "_pctg"].tolist()
    return _rec


def get_categorical(config: dict, df_stats: DataFrame, df: DataFrame):
    """Gets categorical feature stats: count, nunique, nulls, distribution."""
    required_features = config["required_features"]
    features = df_stats.loc[df_stats["_feature"].isin(["CATEGORY", "IP_ADDRESS", "EMAIL_ADDRESS", "TEXT", "PHONE_NUMBER"])]._column.tolist()
    if len(features) == 0:
        return []
    target = required_features["EVENT_LABEL"]
    labels = config["LABELS"]
    majority = config["MAJORITY_CLASS"]
    top_n = config["TopN"]
    min_class_count = config["MinClassCount"]

    df = df[features + [target]].copy()
    rowcnt = len(df)
    df_s1 = df.agg(["count", "nunique"]).transpose().reset_index().rename(columns={"index": "_column"})
    df_s1["count"] = df_s1["count"].astype("int64")
    df_s1["nunique"] = df_s1["nunique"].astype("int64")
    df_s1["null"] = (rowcnt - df_s1["count"]).astype("int64")
    df_s1["not_null"] = rowcnt - df_s1["null"]
    df_s1["null_pct"] = df_s1["null"] / rowcnt
    df_s1["nunique_pct"] = df_s1["nunique"] / rowcnt
    dt = df_stats[["_column", "_feature"]].copy()
    df_stats = pd.merge(dt, df_s1, on="_column", how="inner").round(4)

    cat_list = []
    for rec in df_stats.to_dict("records"):
        if rec["_column"] != target:
            cat_summary_total, cat_summary_majority, cat_summary_non_majority, cat_summary_pctg = col_stats(
                df, target, rec["_column"], majority, min_class_count, labels
            )
            rec["_name"] = f"""'{rec['_column']}'"""
            rec["sort_total"] = col_stats_to_dict(cat_summary_total.head(top_n), rec["_column"], labels)
            rec["sort_majority"] = col_stats_to_dict(cat_summary_majority.head(top_n), rec["_column"], labels)
            rec["sort_nonmajority"] = col_stats_to_dict(cat_summary_non_majority.head(top_n), rec["_column"], labels, True)
            rec["sort_pctg"] = col_stats_to_dict(cat_summary_pctg.head(top_n), rec["_column"], labels)

            if len(cat_summary_total) > 0 and rec["nunique"] != rec["count"]:
                rec["show_in_report"] = True
            else:
                rec["show_in_report"] = None
            cat_list.append(rec)

    return cat_list


def ncol_stats(df, target, column, labels):
    """Calculates numeric column statistics, binned via the Rice rule."""
    df = df.copy()
    n = df[column].nunique()
    k = int(round(2 * (n ** (1 / 3)), 0)) or 1
    try:
        df["bin"] = pd.qcut(df[column], q=k, duplicates="drop")
        num_summary = df.groupby(["bin", target])[target].count().unstack(fill_value=0).reset_index()
        num_summary["total"] = 0
        for c in num_summary.columns:
            if c in labels:
                num_summary["total"] = num_summary["total"] + num_summary[c]
        num_summary["bin_label"] = num_summary["bin"].astype(str)
    except Exception:
        num_summary = pd.DataFrame()
    return num_summary


def datecol_stats(df, target, column, labels):
    df = df.copy()
    df = df[(~df[column].isna()) & (~df[target].isna())]
    df["_dt"] = df[column].dt.date.astype(str)
    df = df.sort_values(by=[column]).reset_index(drop=True)
    num_summary = df.groupby(["_dt", target])[target].count().unstack(fill_value=0).reset_index()
    num_summary["total"] = 0
    for c in num_summary.columns:
        if c in labels:
            num_summary["total"] = num_summary["total"] + num_summary[c]
    num_summary["bin_label"] = num_summary["_dt"].astype(str)
    return num_summary


def get_numerics(config: dict, df_stats: DataFrame, df: DataFrame):
    """Gets numeric feature descriptive statistics and distribution bins."""
    required_features = config["required_features"]
    features = df_stats.loc[df_stats["_feature"].isin(["NUMERIC", "DATETIME"])]._column.tolist()
    if len(features) == 0:
        return []
    target = required_features["EVENT_LABEL"]
    labels = config["LABELS"]

    df = df[features + [target]].copy()
    rowcnt = len(df)
    df_s1 = df[features].agg(["count", "nunique", "mean", "min", "max"]).transpose().reset_index().rename(columns={"index": "_column"})
    df_s1["count"] = df_s1["count"].astype("int64")
    df_s1["nunique"] = df_s1["nunique"].astype("int64")
    df_s1["null"] = (rowcnt - df_s1["count"]).astype("int64")
    df_s1["not_null"] = rowcnt - df_s1["null"]
    df_s1["null_pct"] = df_s1["null"] / rowcnt
    df_s1["nunique_pct"] = df_s1["nunique"] / rowcnt
    dt = df_stats[["_column", "_feature"]].copy()
    df_stats = pd.merge(dt, df_s1, on="_column", how="inner").round(4)

    num_list = []
    for rec in df_stats.to_dict("records"):
        if rec["_column"] != target and rec["count"] > 1:
            if rec["_feature"] == "NUMERIC":
                n_summary = ncol_stats(df, target, rec["_column"], labels)
            elif rec["_feature"] == "DATETIME":
                n_summary = datecol_stats(df, target, rec["_column"], labels)
            else:
                continue
            if n_summary.empty:
                continue
            rec["bin_label"] = n_summary["bin_label"].tolist()
            rec["label_count"] = {}
            rec["pctg"] = {}
            for c in n_summary.columns:
                if c in labels:
                    rec["label_count"][c] = n_summary[c].tolist()
                    rec["pctg"][c] = (n_summary[c] / n_summary["total"]).round(4).tolist()
            rec["total"] = n_summary["total"].tolist()
            num_list.append(rec)

    return num_list


def cramers_corrected_stat(confusion_matrix):
    """Cramer's V for categorical-categorical association, bias-corrected
    per Bergsma and Wicher (2013)."""
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    r, k = confusion_matrix.shape
    if n < 2:
        return 0
    phi2 = chi2 / n
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    if rcorr < 2 or kcorr < 2:
        return 0
    return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))


def correlation_ratio(df_cat, df_num):
    """Correlation ratio (eta) between a categorical and a numeric feature."""
    fcat, _ = pd.factorize(df_cat)
    cat_num = np.max(fcat) + 1
    y_avg_array = np.zeros(cat_num)
    n_array = np.zeros(cat_num)
    for i in range(cat_num):
        cat_measures = df_num.iloc[np.argwhere(fcat == i).flatten()]
        n_array[i] = len(cat_measures)
        y_avg_array[i] = cat_measures.mean()
    if np.sum(n_array) == 0:
        return 0
    y_total_avg = np.sum(np.multiply(y_avg_array, n_array)) / np.sum(n_array)
    numerator = np.sum(np.multiply(n_array, np.power(np.subtract(y_avg_array, y_total_avg), 2)))
    denominator = np.sum(np.power(np.subtract(df_num, y_total_avg), 2))
    if numerator == 0:
        return 0.0
    return np.sqrt(numerator / denominator)


def num_corr(df):
    return df.fillna(-999).corr(method="pearson")


def cat_corr_with_label(df, cat_feat, target):
    if df.shape[0] > 5000:
        df = df[cat_feat + [target]].sample(5000).fillna("<null>")
    res = {}
    for column1 in cat_feat:
        confusion_matrix = df.groupby([column1, target])[target].count().unstack(fill_value=0)
        res[column1] = {target: cramers_corrected_stat(confusion_matrix)}
    return pd.DataFrame(res)


def num_cat_corr(df_num, df_cat):
    df_num = df_num.fillna(-999)
    if df_num.shape[0] > 5000:
        df_num = df_num.sample(5000)
    df_cat = df_cat.loc[df_num.index].fillna("<null>")
    res_cat = {item: {} for item in df_cat.columns}
    res_num = {}
    for column1 in df_num.columns:
        res_num[column1] = {}
        for column2 in df_cat.columns:
            eta = correlation_ratio(df_cat[column2], df_num[column1])
            res_num[column1][column2] = eta
            res_cat[column2][column1] = eta
    return pd.DataFrame(res_num), pd.DataFrame(res_cat)


def get_correlation(config: dict, df_stats: DataFrame, df: DataFrame):
    """Gets correlation between numeric/categorical features and the label."""
    df = df.copy()
    required_features = config["required_features"]
    date_features = df_stats.loc[df_stats["_feature"].isin(["EVENT_TIMESTAMP", "DATETIME"])]._column.tolist()
    num_features = df_stats.loc[df_stats["_feature"] == "NUMERIC"]._column.tolist() + date_features
    cat_features = df_stats.loc[
        (df_stats["_feature"].isin(["CATEGORY", "IP_ADDRESS", "EMAIL_ADDRESS", "TEXT", "PHONE_NUMBER"]))
        & (df_stats["nunique_pct"] < 0.95)
    ]._column.tolist()
    label_feature = required_features["ORIGINAL_LABEL"]

    for c in date_features:
        df.loc[~df[c].isna(), c] = (df[~df[c].isna()][c].astype("int64") / 10**9).astype(int)

    rec_corr = {"features": [], "data_label": [], "feature_corr": None}
    corr_num_label, _ = num_cat_corr(df[num_features], df[[label_feature]])
    corr_cat_label = cat_corr_with_label(df, cat_features, label_feature)
    corr_all_label = pd.concat([corr_num_label, corr_cat_label], axis=1).T
    corr_all_label = corr_all_label.fillna(0)[label_feature]
    corr_all_label = corr_all_label.sort_values(ascending=False)

    for c in corr_all_label.index:
        rec_corr["features"].append(c)
        rec_corr["data_label"].append(corr_all_label.loc[c])

    return rec_corr


def config_html(config: dict):
    """HTML display colors for the report template."""
    base_colors = ["#36A2EB", "#FF6384", "#41d88c", "#9966FF", "#FF9F40", "#8c564b", "#e377c2", "#7f7f7f", "#FFCD56", "#17becf"]
    colors = {}
    count = 0
    for c in config["LABELS"]:
        colors[c] = base_colors[count % len(base_colors)]
        count += 1
    colors["Missing Labels"] = base_colors[count % len(base_colors)]
    return colors


def profile_report(config: dict) -> str:
    """Main entrypoint: builds every stats section and renders the HTML
    report. Requires templates/profile.html next to the working directory."""
    env = Environment(autoescape=select_autoescape(["html", "xml"]), loader=FileSystemLoader("templates"))
    env.globals["zip"] = zip
    profile = env.get_template("profile.html")

    df = get_dataframe(config)
    logging.info("Generate overview.")
    df, overview_stats = get_overview(config, df)
    logging.info("Infer variable types.")
    df, df_stats, warnings = get_stats(config, df)
    logging.info("Generate label stats.")
    lbl_stats, lbl_warnings = get_label(config, df)
    logging.info("Profile categorical features.")
    cat_rec = get_categorical(config, df_stats, df)
    logging.info("Profile numerical features.")
    num_rec = get_numerics(config, df_stats, df)
    logging.info("Calculate correlation.")
    corr_rec = get_correlation(config, df_stats, df)

    colors = config_html(config)

    logging.info("Render report.")
    return profile.render(
        file=config["file_name"],
        overview=overview_stats,
        warnings=warnings,
        df_stats=df_stats,
        label=lbl_stats,
        label_msg=lbl_warnings,
        cat_rec=cat_rec,
        num_rec=num_rec,
        corr_rec=corr_rec,
        label_colors=colors,
    )
