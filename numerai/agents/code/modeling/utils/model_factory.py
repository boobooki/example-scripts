from __future__ import annotations

def build_model(
    model_type: str,
    model_params: dict,
    model_config: dict | None = None,
    *,
    feature_cols: list[str] | None = None,
):
    model_config = model_config or {}
    if model_type == "LGBMRegressor":
        from agents.code.modeling.models.lgbm_regressor import LGBMRegressor
        model = LGBMRegressor(feature_cols=feature_cols, **model_params)
    elif model_type == "NumeraiMLP":
        from agents.code.modeling.models.nn_mlp import NumeraiMLP
        model = NumeraiMLP(feature_cols=feature_cols, **model_params)
    elif model_type == "NumeraiResNet":
        from agents.code.modeling.models.nn_resnet import NumeraiResNet
        model = NumeraiResNet(feature_cols=feature_cols, **model_params)
    elif model_type == "NumeraiFTTransformer":
        from agents.code.modeling.models.nn_ft_transformer import NumeraiFTTransformer
        model = NumeraiFTTransformer(feature_cols=feature_cols, **model_params)
    elif model_type == "NumeraiTabM":
        from agents.code.modeling.models.nn_tabm import NumeraiTabM
        model = NumeraiTabM(feature_cols=feature_cols, **model_params)
    elif model_type == "NumeraiRealMLP":
        from agents.code.modeling.models.nn_realmlp import NumeraiRealMLP
        model = NumeraiRealMLP(feature_cols=feature_cols, **model_params)
    else:
        raise ValueError(
            "Unsupported model type: "
            f"{model_type}. Supported types: LGBMRegressor, NumeraiMLP, NumeraiResNet, "
            "NumeraiFTTransformer, NumeraiTabM, NumeraiRealMLP"
        )

    target_transform = model_config.get("target_transform")
    if target_transform:
        from agents.code.modeling.utils.target_transforms import TargetTransformWrapper

        model = TargetTransformWrapper(model, target_transform)
    return model
