import torch.nn as nn
from executorch.examples.models.llama2.model import (
    ModelArgs as LlamaModelArgs,
    Transformer as LlamaTransformer,
)
from executorch.extension.gguf_util.load_gguf import GGUFModelArgs, GGUFWeights


def _create_pt_model(
    gguf_model_args: GGUFModelArgs, gguf_weights: GGUFWeights
) -> nn.Module:
    llama_model_args = LlamaModelArgs(
        dim=gguf_model_args.embedding_length,
        n_layers=gguf_model_args.block_count,
        n_heads=gguf_model_args.attenion.head_count,
        n_kv_heads=gguf_model_args.attenion.head_count_kv,
        vocab_size=gguf_model_args.vocab_size,
        norm_eps=gguf_model_args.attenion.layer_norm_rms_epsilon,
        # feed_forward_length=gguf_model_args.feed_forward_length,
        # rope_freq_base=gguf_model_args.rope.freq_base,
    )
    pt_model = LlamaTransformer(llama_model_args)
    return pt_model


def _create_pte_program(pt_model: nn.Module) -> bytes:
    # TODO
    return ""


def convert_to_pte(gguf_model_args: GGUFModelArgs, gguf_weights: GGUFWeights) -> None:
    """Convert a GGUF model into an ExecuTorch program.

    Args:
        model_args: The arguments for the GGUF model.
        weights: The weights of the GGUF model.
    """
    pt_model = _create_pt_model(gguf_model_args, gguf_weights)
    print("Converting to PTE")
    pte_program = _create_pte_program(pt_model)
    return pte_program
