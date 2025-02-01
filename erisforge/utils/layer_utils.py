import json
import logging
import os.path
from typing import (
    Dict,
)

from transformers import (
    AutoModelForCausalLM,
)


def get_layers_names_by_model(
    model_name: str | None = None,
) -> Dict[str, str | Dict[str, str]]:
    """
    Get the layers names by model.
    :param model_name: The model name, if None, return all models.
    :return: model_dict, a dictionary with the model name as key and the layers names as value. If model_name is not None, return the layers names for the model.
    """
    path = (
        os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        + "/assets/llm_models.json"
    )
    model_dict = json.load(open(path))
    if model_name:
        if model_name not in model_dict:
            raise ValueError(f"Model {model_name} not found in the model dictionary.")
        return model_dict[model_name]
    return model_dict


def identify_model(custom_model: AutoModelForCausalLM) -> str:
    """
    Identify the model based on the layer names.
    :param custom_model: the model to identify.
    :return: model_name
    """
    param_names = set(name for name, _ in custom_model.named_parameters())

    for model_name, layers in get_layers_names_by_model().items():
        self_attention_layer = layers["self_attention"]
        mlp_layer = layers["mlp"]
        if any(self_attention_layer in param for param in param_names) and any(
            mlp_layer in param for param in param_names
        ):
            logging.info(f"Compatible layers naming with {model_name} family model.")
            return model_name

    raise ValueError("No matching model found based on layer names.")
