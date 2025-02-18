# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import torch
from executorch.exir.pass_base import ExportPass, PassResult
from torch._decomp import get_decompositions
from torch.fx.experimental.proxy_tensor import make_fx


class ConvertScaledDotProductAttention(ExportPass):
    """
    Decompose from scaled_dot_product_attention to multiple nodes.
    """

    def call(self, graph_module: torch.fx.GraphModule):
        graph = graph_module.graph
        for node in graph.nodes:
            if node.target == torch.ops.aten.scaled_dot_product_attention.default:
                input_tensors = (
                    node.args[0].meta["val"],
                    node.args[1].meta["val"],
                    node.args[2].meta["val"],
                )
                # refer to pytorch/test/test_decomp.py
                decomposed_module = make_fx(
                    node.target,
                    decomposition_table=get_decompositions(
                        [
                            torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default,
                        ]
                    ),
                    tracing_mode="fake",
                    _allow_non_fake_inputs=True,
                )(*input_tensors)
                with graph.inserting_before(node):
                    name_to_input_tensor_map = {
                        "arg0_1": node.args[0],
                        "arg1_1": node.args[1],
                        "arg2_1": node.args[2],
                    }
                    decomposed_node_to_subgraph_node = {}
                    last_decomposed_node = None
                    # Create a mapping from input nodes in decomposed module to original nodes.
                    # In decomposed module, there are only input tensors for placeholder op.
                    for decomposed_node in decomposed_module.graph.nodes:
                        if decomposed_node.op == "placeholder":
                            decomposed_node_to_subgraph_node[
                                decomposed_node
                            ] = name_to_input_tensor_map[decomposed_node.name]

                        if decomposed_node.op == "output":
                            last_decomposed_node = decomposed_node.args[0]

                    # Copy node from decompose graph module
                    for decomposed_node in decomposed_module.graph.nodes:
                        if decomposed_node.op == "placeholder":
                            continue

                        if (
                            decomposed_node.op == "output"
                            and last_decomposed_node is not None
                        ):
                            for user in node.users.copy():
                                user.replace_input_with(
                                    node,
                                    decomposed_node_to_subgraph_node[
                                        last_decomposed_node
                                    ],
                                )
                            continue

                        subgraph_node = graph.node_copy(
                            decomposed_node,
                            arg_transform=lambda x: decomposed_node_to_subgraph_node[  # noqa: B023
                                x
                            ],
                        )
                        subgraph_node.meta["source_fn_stack"] = [
                            (subgraph_node, subgraph_node.target)
                        ]
                        decomposed_node_to_subgraph_node[
                            decomposed_node
                        ] = subgraph_node

                    graph.erase_node(node)

        graph.eliminate_dead_code()
        graph_module.recompile()
        return PassResult(graph_module, True)
