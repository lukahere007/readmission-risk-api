"""Model pipelines. Each takes raw encounter columns (data.INPUT_COLUMNS) end to end."""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler

from .data import CATEGORICAL_FEATURES, NUMERIC_FEATURES, engineer_features

SEED = 42

# Rare levels (e.g. most of the ~70 medical specialties) are pooled into one
# "infrequent" bucket so the encoders do not memorise tiny groups.
MIN_CATEGORY_COUNT = 100


def _features() -> FunctionTransformer:
    return FunctionTransformer(engineer_features, validate=False)


def logistic_pipeline() -> Pipeline:
    pre = ColumnTransformer(
        [
            ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                              ("scale", StandardScaler())]), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                  min_frequency=MIN_CATEGORY_COUNT), CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline([
        ("features", _features()),
        ("pre", pre),
        ("clf", LogisticRegression(C=0.1, max_iter=2000)),
    ])


def gbm_pipeline(**params) -> Pipeline:
    pre = ColumnTransformer(
        [
            ("num", "passthrough", NUMERIC_FEATURES),
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                                   encoded_missing_value=-1,
                                   min_frequency=MIN_CATEGORY_COUNT), CATEGORICAL_FEATURES),
        ],
        verbose_feature_names_out=False,
    )
    n_num = len(NUMERIC_FEATURES)
    cat_mask = [False] * n_num + [True] * len(CATEGORICAL_FEATURES)
    defaults = dict(
        learning_rate=0.05, max_iter=400, max_leaf_nodes=15, min_samples_leaf=100,
        l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=30, random_state=SEED,
    )
    defaults.update(params)
    return Pipeline([
        ("features", _features()),
        ("pre", pre),
        ("clf", HistGradientBoostingClassifier(categorical_features=cat_mask, **defaults)),
    ])
