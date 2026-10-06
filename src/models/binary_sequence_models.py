import torch
import torch.nn as nn


class BinaryGRU(nn.Module):
    """
    GRU for binary 5-minute direction prediction.

    Output:
        One logit.
        > 0 -> UP
        < 0 -> DOWN
    """

    def __init__(
        self,
        input_size,
        hidden_size=64,
        num_layers=1,
        dropout=0.2,
        num_symbols=5,
        symbol_embedding_dim=4,
    ):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=(
                dropout
                if num_layers > 1
                else 0.0
            ),
        )

        self.symbol_embedding = nn.Embedding(
            num_symbols,
            symbol_embedding_dim,
        )

        combined_size = (
            hidden_size
            + symbol_embedding_dim
        )

        self.dropout = nn.Dropout(
            dropout
        )

        self.classification_head = nn.Sequential(
            nn.Linear(
                combined_size,
                32,
            ),
            nn.ReLU(),
            nn.Linear(
                32,
                1,
            ),
        )

    def forward(
        self,
        x,
        symbol_id,
    ):
        _, hidden = self.gru(x)

        sequence_representation = (
            hidden[-1]
        )

        symbol_representation = (
            self.symbol_embedding(
                symbol_id
            )
        )

        combined = torch.cat(
            [
                sequence_representation,
                symbol_representation,
            ],
            dim=1,
        )

        combined = self.dropout(
            combined
        )

        output = (
            self.classification_head(
                combined
            ).squeeze(-1)
        )

        return output


class BinaryLSTM(nn.Module):
    """
    LSTM for binary 5-minute direction prediction.
    """

    def __init__(
        self,
        input_size,
        hidden_size=64,
        num_layers=1,
        dropout=0.2,
        num_symbols=5,
        symbol_embedding_dim=4,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=(
                dropout
                if num_layers > 1
                else 0.0
            ),
        )

        self.symbol_embedding = nn.Embedding(
            num_symbols,
            symbol_embedding_dim,
        )

        combined_size = (
            hidden_size
            + symbol_embedding_dim
        )

        self.dropout = nn.Dropout(
            dropout
        )

        self.classification_head = nn.Sequential(
            nn.Linear(
                combined_size,
                32,
            ),
            nn.ReLU(),
            nn.Linear(
                32,
                1,
            ),
        )

    def forward(
        self,
        x,
        symbol_id,
    ):
        _, (hidden, _) = self.lstm(x)

        sequence_representation = (
            hidden[-1]
        )

        symbol_representation = (
            self.symbol_embedding(
                symbol_id
            )
        )

        combined = torch.cat(
            [
                sequence_representation,
                symbol_representation,
            ],
            dim=1,
        )

        combined = self.dropout(
            combined
        )

        output = (
            self.classification_head(
                combined
            ).squeeze(-1)
        )

        return output


class BinaryCNNLSTM(nn.Module):
    """
    CNN-LSTM for binary 5-minute direction prediction.
    """

    def __init__(
        self,
        input_size,
        cnn_channels=64,
        hidden_size=64,
        dropout=0.2,
        num_symbols=5,
        symbol_embedding_dim=4,
    ):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(
                in_channels=input_size,
                out_channels=cnn_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=cnn_channels,
                out_channels=cnn_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
        )

        self.lstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )

        self.symbol_embedding = nn.Embedding(
            num_symbols,
            symbol_embedding_dim,
        )

        combined_size = (
            hidden_size
            + symbol_embedding_dim
        )

        self.dropout = nn.Dropout(
            dropout
        )

        self.classification_head = nn.Sequential(
            nn.Linear(
                combined_size,
                32,
            ),
            nn.ReLU(),
            nn.Linear(
                32,
                1,
            ),
        )

    def forward(
        self,
        x,
        symbol_id,
    ):
        # x:
        # [batch, sequence, features]

        x = x.transpose(
            1,
            2,
        )

        x = self.cnn(x)

        x = x.transpose(
            1,
            2,
        )

        _, (hidden, _) = self.lstm(
            x
        )

        sequence_representation = (
            hidden[-1]
        )

        symbol_representation = (
            self.symbol_embedding(
                symbol_id
            )
        )

        combined = torch.cat(
            [
                sequence_representation,
                symbol_representation,
            ],
            dim=1,
        )

        combined = self.dropout(
            combined
        )

        output = (
            self.classification_head(
                combined
            ).squeeze(-1)
        )

        return output