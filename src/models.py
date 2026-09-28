from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from xgboost import XGBRegressor

def make_models(
    linear_params,
    xgb_params,
):
    """
    Create fresh model objects using the hyperparameters
    selected during the validation period.
    """

    models = {
        "ols": LinearRegression(),

        "ridge": Ridge(
            alpha=linear_params[
                "ridge"
            ]["alpha"]
        ),

        "elastic_net": ElasticNet(
            alpha=linear_params[
                "elastic_net"
            ]["alpha"],
            l1_ratio=linear_params[
                "elastic_net"
            ]["l1_ratio"],
            max_iter=100_000,
        ),

        "xgboost": XGBRegressor(
            objective="reg:squarederror",

            max_depth=xgb_params[
                "max_depth"
            ],

            learning_rate=xgb_params[
                "learning_rate"
            ],

            n_estimators=xgb_params[
                "n_estimators"
            ],

            subsample=xgb_params[
                "subsample"
            ],

            colsample_bytree=xgb_params[
                "colsample_bytree"
            ],

            min_child_weight=xgb_params[
                "min_child_weight"
            ],

            reg_lambda=xgb_params[
                "reg_lambda"
            ],

            reg_alpha=xgb_params[
                "reg_alpha"
            ],

            random_state=42,
            n_jobs=-1,
        ),
    }

    return models