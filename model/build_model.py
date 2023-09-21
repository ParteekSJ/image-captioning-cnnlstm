import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from model.decoder import LSTMModel
from model.encoder import CNNModel
from model.encoder_decoder import CNNLSTMModel


def get_model(cfg, vocabulary):
    if cfg.MODEL.NAME == "cnnlstm":
        encoder = CNNModel(embedding_size=cfg.MODEL.EMBEDDING_SIZE)
        decoder = LSTMModel(
            embedding_size=cfg.MODEL.EMBEDDING_SIZE,
            hidden_layer_size=cfg.MODEL.HIDDEN_SIZE,
            vocabulary_size=len(vocabulary),
            num_layers=cfg.MODEL.NUM_LSTM_LAYERS,
        )
        model = CNNLSTMModel(encoder, decoder)

    return model


def is_parallel(model):
    return type(model) in (
        nn.parallel.DataParallel,
        nn.parallel.DistributedDataParallel,
    )


def de_parallel(model):
    return model.module if is_parallel(model) else model


def parallel_model(model, device, rank, local_rank):
    # DDP mode
    ddp_mode = device.type != "cpu" and rank != -1
    if ddp_mode:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )
    return model
