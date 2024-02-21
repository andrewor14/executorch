from executorch.extension.gguf_util.load_gguf import GGUFModelArgs, GGUFWeights


def convert_to_pte(model_args: GGUFModelArgs, weights: GGUFWeights) -> None:
    """Convert a GGUF model into an ExecuTorch program.

    Args:
        model_args: The arguments for the GGUF model.
        weights: The weights of the GGUF model.
    """
    if model_args.arch == "llama":
        from executorch.extension.gguf_util.converters.llama_converter import (
            convert_to_pte as llama_convert_to_pte,
        )

        return llama_convert_to_pte(model_args, weights)
    else:
        raise NotImplementedError("Unsupported architecture.")
