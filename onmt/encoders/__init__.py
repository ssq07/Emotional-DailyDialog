"""Module defining encoders."""
from onmt.encoders.encoder import EncoderBase
from onmt.encoders.transformer import TransformerEncoder
from onmt.encoders.ea_transformer import EA_TransformerEncoder
from onmt.encoders.ggnn_encoder import GGNNEncoder
from onmt.encoders.rnn_encoder import RNNEncoder
from onmt.encoders.cnn_encoder import CNNEncoder
from onmt.encoders.mean_encoder import MeanEncoder


str2enc = {"ggnn": GGNNEncoder, "rnn": RNNEncoder, "brnn": RNNEncoder,
           "cnn": CNNEncoder, "transformer": TransformerEncoder,
           "mean": MeanEncoder, "ea_trans":EA_TransformerEncoder}

__all__ = ["EncoderBase", "TransformerEncoder", "EA_TransformerEncoder", "RNNEncoder", "CNNEncoder",
           "MeanEncoder", "str2enc"]
