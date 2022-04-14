from torch import nn
from typing import List, Dict, Any


def LinearBlock(
    in_features: int,
    out_features: int,
    batchnorm: bool, 
    activation: nn.Module,
    dropout: float,
):
    block = list()
    block.append(nn.Linear(in_features, out_features))

    if batchnorm:
        block.append(nn.BatchNorm1d(out_features))

    if activation is not None:
        block.append(activation)

    assert dropout is None or isinstance(dropout, float)
    if dropout is not None and dropout > 0:
        block.append(nn.Dropout(p=dropout))
    return block


class LinearSequential(nn.Sequential):
    def __init__(
        self,
        features: List[int],
        batchnorm: bool,
        activation: nn.Module,
        dropout: float,
        last_block: bool = True,
    ):
        flags = [True] * max(len(features) - 1, 1)
        flags[-1] = last_block

        sequential = list()
        for in_features, out_features, flag in zip(features[:-1], features[1:], flags):
            if flag:
                sequential += LinearBlock(
                    in_features=in_features,
                    out_features=out_features,
                    batchnorm=batchnorm,
                    activation=activation,
                    dropout=dropout,
                )
            else:
                sequential += [nn.Linear(
                    in_features=in_features,
                    out_features=out_features,
                )]
        super().__init__(*sequential)
