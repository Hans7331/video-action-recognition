"""Convert Flax checkpoints from original paper to PyTorch"""
import argparse
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

from flax.training.checkpoints import restore_checkpoint


def transform_attention(current: np.ndarray):
    if np.ndim(current) == 2:
        return transform_attention_bias(current)

    elif np.ndim(current) == 3:
        return transform_attention_kernel(current)

    else:
        raise Exception(f"Invalid number of dimesions: {np.ndim(current)}")


def transform_attention_bias(current: np.ndarray):
    return current.flatten()


def transform_attention_kernel(current: np.ndarray):
    print(np.shape(current))
    return np.reshape(current, (current.shape[0], current.shape[1] * current.shape[2])).T


def transform_attention_output_weight(current: np.ndarray):
    return np.reshape(current, (current.shape[0] * current.shape[1], current.shape[2])).T


def transform_state_encoder_block(state_dict, i):
    state = state_dict["optimizer"]["target"]["TemporalTransformer"][f"encoderblock_{i}"]

    q = torch.tensor(transform_attention(state["MultiHeadDotProductAttention_0"]["query"]["kernel"]))
    k = torch.tensor(transform_attention(state["MultiHeadDotProductAttention_0"]["key"]["kernel"]))
    v = torch.tensor(transform_attention(state["MultiHeadDotProductAttention_0"]["value"]["kernel"]))
    proj_weight = torch.cat((q, k, v), 0)
    q_bias = torch.tensor(transform_attention(state["MultiHeadDotProductAttention_0"]["query"]["bias"]))
    k_bias = torch.tensor(transform_attention(state["MultiHeadDotProductAttention_0"]["key"]["bias"]))
    v_bias = torch.tensor(transform_attention(state["MultiHeadDotProductAttention_0"]["value"]["bias"]))
    proj_bias = torch.cat((q_bias, k_bias, v_bias), 0)
    out_w = transform_attention_output_weight(state["MultiHeadDotProductAttention_0"]["out"]["kernel"])
    out_b = state["MultiHeadDotProductAttention_0"]["out"]["bias"]

    new_state = OrderedDict()
    prefix = f"temporal_transformer.model_layers.{i}."
    new_state = {
        prefix + "0.weight": state["LayerNorm_0"]["scale"],
        prefix + "0.bias": state["LayerNorm_0"]["bias"],
        prefix + "1.make_qkv.weight": proj_weight,
        prefix + "1.make_qkv.bias": proj_bias,
        prefix + "1.get_output.0.weight": out_w,
        prefix + "1.get_output.0.bias": out_b,
        prefix + "2.weight": state["LayerNorm_1"]["scale"],
        prefix + "2.bias": state["LayerNorm_1"]["bias"],
        prefix + "3.mlp.0.weight": np.transpose(state["MlpBlock_0"]["Dense_0"]["kernel"]),
        prefix + "3.mlp.0.bias": state["MlpBlock_0"]["Dense_0"]["bias"],
        # random intitialization for the hidden layer in the mlp
        prefix + "3.mlp.3.weight": np.transpose(state["MlpBlock_0"]["Dense_1"]["kernel"]),
        prefix + "3.mlp.3.bias": state["MlpBlock_0"]["Dense_1"]["bias"],
    }

    return new_state


def transform_state(state_dict, transformer_layers=12, classification_head=False):
    new_state = OrderedDict()

    new_state["temporal_transformer.layer_norm.weight"] = state_dict["optimizer"]["target"]["TemporalTransformer"]["encoder_norm"]["scale"]
    new_state["temporal_transformer.layer_norm.bias"] = state_dict["optimizer"]["target"]["TemporalTransformer"]["encoder_norm"]["bias"]
    

    new_state["tube.projection.weight_temp"] = np.transpose(
        state_dict["optimizer"]["target"]["embedding"]["kernel"], (4, 3, 0, 1, 2)
    )
    new_state["tube.projection.bias_temp"] = state_dict["optimizer"]["target"]["embedding"]["bias"]

    new_state["temporal_token"] = state_dict["optimizer"]["target"]["cls_TemporalTransformer"]
    pos = torch.tensor((state_dict["optimizer"]["target"]["TemporalTransformer"]["posembed_input"][
        "pos_embedding"]))
    new_state["pos_embed_temp"] = pos.unsqueeze(1)

    for i in range(transformer_layers):
        new_state.update(transform_state_encoder_block(state_dict, i))

    if classification_head:
        new_state = {"vivit." + k: v for k, v in new_state.items()}
        new_state["classifier.weight"] = np.transpose(state_dict["optimizer"]["target"]["output_projection"]["kernel"])
        new_state["classifier.bias"] = np.transpose(state_dict["optimizer"]["target"]["output_projection"]["bias"])

    return {k: torch.tensor(v) for k, v in new_state.items()}


def get_n_layers(state_dict):
    print(state_dict['optimizer']['target']['TemporalTransformer']['posembed_input'].keys())
    return sum([1 if "encoderblock_" in k else 0 for k in state_dict["optimizer"]["target"]["TemporalTransformer"].keys()])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--flax_model", type=str, help="Path to flax model")
    parser.add_argument("--output_model_name", type=str, help="Name of the outputed file")
    parser.add_argument("--classification_head", action="store_true", help="Add classification head weights")

    args = parser.parse_args()

    state_dict = restore_checkpoint(args.flax_model, None)

    n_layers = get_n_layers(state_dict)
    new_state = transform_state(state_dict, n_layers, classification_head=args.classification_head)

    out_path = Path(args.flax_model).parent.absolute()

    if ".pt" in args.output_model_name:
        out_path = out_path / args.output_model_name

    else:
        out_path = out_path / (args.output_model_name + ".pt")

    torch.save(new_state, out_path)
