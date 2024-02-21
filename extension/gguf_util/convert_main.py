# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse

from executorch.extension.gguf_util.converter import convert_to_pte
from executorch.extension.gguf_util.load_gguf import load_file


def save_pte_program(_, pte_file) -> None:
    # TODO
    print(f"Saving PTE program to {pte_file}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gguf_file",
        type=str,
        help="The GGUF file to load.",
    )
    parser.add_argument(
        "--pte_file",
        type=str,
        help="The path to save the PTE file.",
    )
    args = parser.parse_args()
    gguf_model_args, gguf_weights = load_file(args.gguf_file)
    pte_program = convert_to_pte(gguf_model_args, gguf_weights)
    save_pte_program(pte_program, args.pte_file)


if __name__ == "__main__":
    main()
