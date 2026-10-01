from src.models.train_sequence_models import (
    create_model,
    select_evenly,
)


def test_select_evenly():

    indices = list(
        range(100)
    )

    selected = select_evenly(
        indices,
        10,
    )

    assert len(selected) == 10
    assert selected[0] == 0
    assert selected[-1] == 99


def test_create_gru():

    model = create_model(
        "GRU"
    )

    assert model is not None


def test_create_lstm():

    model = create_model(
        "LSTM"
    )

    assert model is not None


def test_create_cnn_lstm():

    model = create_model(
        "CNNLSTM"
    )

    assert model is not None