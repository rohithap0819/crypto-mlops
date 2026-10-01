import torch

from src.models.sequence_models import (
    MultiTaskCNNLSTM,
    MultiTaskGRU,
    MultiTaskLSTM,
)


INPUT_SIZE = 61
SEQUENCE_LENGTH = 60
BATCH_SIZE = 8
NUM_SYMBOLS = 5
NUM_CLASSES = 3


def create_test_input():

    x = torch.randn(
        BATCH_SIZE,
        SEQUENCE_LENGTH,
        INPUT_SIZE,
    )

    symbol_id = torch.tensor(
        [0, 1, 2, 3, 4, 0, 1, 2],
        dtype=torch.long,
    )

    return x, symbol_id


def check_model(model):

    x, symbol_id = (
        create_test_input()
    )

    regression, classification = (
        model(
            x,
            symbol_id,
        )
    )

    assert regression.shape == (
        BATCH_SIZE,
    )

    assert classification.shape == (
        BATCH_SIZE,
        NUM_CLASSES,
    )


def test_gru():

    model = MultiTaskGRU(
        input_size=INPUT_SIZE
    )

    check_model(model)


def test_lstm():

    model = MultiTaskLSTM(
        input_size=INPUT_SIZE
    )

    check_model(model)


def test_cnn_lstm():

    model = MultiTaskCNNLSTM(
        input_size=INPUT_SIZE
    )

    check_model(model)