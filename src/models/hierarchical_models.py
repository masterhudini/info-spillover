"""
Hierarchical Modeling Pipeline for Cryptocurrency Sentiment Analysis
Level 1: Subreddit-Level LSTM/Transformer Models
Level 2: Cross-Subreddit Graph Neural Networks

Based on academic methodology for financial spillover analysis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch_geometric
from torch_geometric.nn import GCNConv, GATConv, GatedGraphConv
from torch_geometric.data import Data, Batch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import networkx as nx
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, f1_score
import logging
from pathlib import Path
import pickle
import json

# Transformers for attention mechanism
from transformers import AutoModel, AutoConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SubredditTimeSeriesDataset(Dataset):
    """
    Dataset for individual subreddit time series
    Implements sliding window approach for temporal modeling
    """

    def __init__(self, df: pd.DataFrame, subreddit: str,
                 sequence_length: int = 24, prediction_horizon: int = 1,
                 feature_columns: List[str] = None, target_columns: List[str] = None):

        self.subreddit = subreddit
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon

        # Filter data for specific subreddit
        self.data = df[df['subreddit'] == subreddit].copy()
        self.data = self.data.sort_values('created_utc').reset_index(drop=True)

        # Define features and targets
        if feature_columns is None:
            self.feature_columns = [col for col in df.columns if any(
                pattern in col for pattern in ['sentiment_', 'emotion_', 'ratio_', 'count_', 'net_spillover', 'pagerank']
            )]
        else:
            self.feature_columns = feature_columns

        if target_columns is None:
            self.target_columns = ['next_sentiment', 'sentiment_change', 'next_return', 'return_direction']
        else:
            self.target_columns = target_columns

        # Remove missing target columns
        self.target_columns = [col for col in self.target_columns if col in self.data.columns]

        # Prepare sequences
        self._prepare_sequences()

    def _prepare_sequences(self):
        """Prepare sliding window sequences"""

        # Extract feature and target arrays
        features = self.data[self.feature_columns].values
        targets = self.data[self.target_columns].values

        # Handle missing values
        features = np.nan_to_num(features, nan=0.0)
        targets = np.nan_to_num(targets, nan=0.0)

        self.sequences = []
        self.targets = []

        for i in range(len(features) - self.sequence_length - self.prediction_horizon + 1):
            # Input sequence
            seq = features[i:i + self.sequence_length]

            # Target (next prediction_horizon steps)
            target = targets[i + self.sequence_length:i + self.sequence_length + self.prediction_horizon]

            self.sequences.append(seq)
            self.targets.append(target)

        self.sequences = np.array(self.sequences, dtype=np.float32)
        self.targets = np.array(self.targets, dtype=np.float32)

        logger.info(f"Created {len(self.sequences)} sequences for {self.subreddit}")
        logger.info(f"Sequence shape: {self.sequences.shape}, Target shape: {self.targets.shape}")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.sequences[idx]), torch.FloatTensor(self.targets[idx])


class SubredditLSTM(nn.Module):
    """
    LSTM model for individual subreddit sentiment prediction
    Based on Hochreiter & Schmidhuber (1997) with modern enhancements
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2,
                 output_dim: int = 4, dropout: float = 0.2, attention: bool = True):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.attention = attention

        # LSTM layers
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )

        # Attention mechanism
        if self.attention:
            self.attention_layer = nn.MultiheadAttention(
                hidden_dim, num_heads=8, dropout=dropout, batch_first=True
            )

        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier initialization"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if len(param.shape) >= 2:
                    nn.init.xavier_uniform_(param)
                else:
                    nn.init.uniform_(param, -0.1, 0.1)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

    def forward(self, x):
        """
        Forward pass
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_dim)
        Returns:
            Output tensor of shape (batch_size, prediction_horizon, output_dim)
        """
        batch_size = x.size(0)

        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)

        # Apply attention if enabled
        if self.attention:
            # Self-attention over sequence
            attended, attention_weights = self.attention_layer(lstm_out, lstm_out, lstm_out)
            # Use the last attended output
            final_output = attended[:, -1, :]
        else:
            # Use last LSTM output
            final_output = lstm_out[:, -1, :]

        # Apply dropout and output layer
        output = self.dropout(final_output)
        output = self.output_layer(output)

        # Reshape for prediction horizon
        if self.output_dim > 1:
            output = output.view(batch_size, -1, self.output_dim // 1)  # Assume single step prediction

        return output


class SubredditTransformer(nn.Module):
    """
    Transformer model for subreddit sentiment prediction
    Based on Vaswani et al. (2017) "Attention Is All You Need"
    """

    def __init__(self, input_dim: int, d_model: int = 128, nhead: int = 8,
                 num_encoder_layers: int = 3, dim_feedforward: int = 512,
                 output_dim: int = 4, dropout: float = 0.1, max_seq_len: int = 1000):
        super().__init__()

        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)

        # Positional encoding
        self.pos_encoding = self._create_positional_encoding(max_seq_len, d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

        # Output projection
        self.output_projection = nn.Linear(d_model, output_dim)
        self.dropout = nn.Dropout(dropout)

    def _create_positional_encoding(self, max_len: int, d_model: int):
        """Create sinusoidal positional encoding"""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           (-np.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return nn.Parameter(pe.unsqueeze(0), requires_grad=False)

    def forward(self, x):
        """
        Forward pass
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_dim)
        """
        batch_size, seq_len, _ = x.shape

        # Input projection
        x = self.input_projection(x)

        # Add positional encoding
        if seq_len <= self.max_seq_len:
            x = x + self.pos_encoding[:, :seq_len, :]

        # Transformer encoder
        x = self.transformer_encoder(x)

        # Use last token for prediction (or could use pooling)
        x = x[:, -1, :]  # Take last time step

        # Output projection
        output = self.dropout(x)
        output = self.output_projection(output)

        return output


class SpilloverGNN(nn.Module):
    """
    Graph Neural Network for cross-subreddit spillover modeling
    Based on Li et al. (2015) Gated Graph Neural Networks
    """

    def __init__(self, node_features: int, edge_features: int = 1,
                 hidden_dim: int = 64, num_layers: int = 3,
                 output_dim: int = 1, gnn_type: str = 'GAT'):
        super().__init__()

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.gnn_type = gnn_type

        # Input projection
        self.input_projection = nn.Linear(node_features, hidden_dim)

        # Edge feature processing
        if edge_features > 0:
            self.edge_mlp = nn.Sequential(
                nn.Linear(edge_features, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )

        # GNN layers
        self.gnn_layers = nn.ModuleList()

        if gnn_type == 'GAT':
            # Graph Attention Network
            for i in range(num_layers):
                if i == 0:
                    self.gnn_layers.append(
                        GATConv(hidden_dim, hidden_dim // 8, heads=8, dropout=0.1)
                    )
                else:
                    self.gnn_layers.append(
                        GATConv(hidden_dim, hidden_dim // 8, heads=8, dropout=0.1)
                    )
        elif gnn_type == 'GCN':
            # Graph Convolutional Network
            for i in range(num_layers):
                self.gnn_layers.append(
                    GCNConv(hidden_dim, hidden_dim)
                )
        elif gnn_type == 'GGNN':
            # Gated Graph Neural Network
            self.gnn_layers.append(
                GatedGraphConv(hidden_dim, num_layers)
            )

        # Output layers
        self.norm_layers = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.dropout = nn.Dropout(0.1)
        self.output_projection = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        """
        Forward pass
        Args:
            x: Node features (num_nodes, node_features)
            edge_index: Edge indices (2, num_edges)
            edge_attr: Edge attributes (num_edges, edge_features)
            batch: Batch assignment for batched processing
        """

        # Input projection
        h = self.input_projection(x)
        h = F.relu(h)

        # Process edge attributes if available
        if edge_attr is not None and hasattr(self, 'edge_mlp'):
            edge_attr = self.edge_mlp(edge_attr)

        # GNN layers
        if self.gnn_type == 'GGNN':
            h = self.gnn_layers[0](h, edge_index)
        else:
            for i, gnn_layer in enumerate(self.gnn_layers):
                h_new = gnn_layer(h, edge_index, edge_attr=edge_attr)
                h_new = self.norm_layers[i](h_new)
                h_new = F.relu(h_new)
                h_new = self.dropout(h_new)

                # Residual connection
                if h.shape == h_new.shape:
                    h = h + h_new
                else:
                    h = h_new

        # Output projection
        output = self.output_projection(h)

        return output


class HierarchicalSentimentModel(pl.LightningModule):
    """
    Hierarchical model combining individual subreddit models with GNN
    PyTorch Lightning implementation for scalable training
    """

    def __init__(self, subreddit_models: Dict[str, nn.Module],
                 gnn_model: SpilloverGNN,
                 feature_dim: int,
                 learning_rate: float = 1e-3,
                 weight_decay: float = 1e-4):
        super().__init__()

        self.save_hyperparameters(ignore=['subreddit_models', 'gnn_model'])

        # Individual subreddit models (Level 1)
        self.subreddit_models = nn.ModuleDict(subreddit_models)

        # Graph neural network (Level 2)
        self.gnn_model = gnn_model

        # Fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),  # Combine L1 and L2 features
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim, 4)  # Final predictions
        )

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # Metrics storage
        self.train_metrics = {'mse': [], 'mae': [], 'accuracy': []}
        self.val_metrics = {'mse': [], 'mae': [], 'accuracy': []}

    def forward(self, subreddit_data: Dict[str, torch.Tensor],
                graph_data: Data) -> torch.Tensor:
        """
        Hierarchical forward pass
        Level 1: Individual subreddit predictions
        Level 2: Graph-level spillover modeling
        """

        # Level 1: Subreddit-level predictions
        subreddit_embeddings = {}
        subreddit_predictions = {}

        for subreddit, data in subreddit_data.items():
            if subreddit in self.subreddit_models:
                # Get individual prediction and embedding
                pred = self.subreddit_models[subreddit](data)
                subreddit_predictions[subreddit] = pred

                # Extract embedding (before final layer)
                if hasattr(self.subreddit_models[subreddit], 'lstm'):
                    # For LSTM model
                    lstm_out, _ = self.subreddit_models[subreddit].lstm(data)
                    embedding = lstm_out[:, -1, :]  # Last hidden state
                else:
                    # For Transformer model
                    embedding = self.subreddit_models[subreddit].transformer_encoder(
                        self.subreddit_models[subreddit].input_projection(data)
                    )[:, -1, :]

                subreddit_embeddings[subreddit] = embedding

        # Level 2: Graph neural network
        if graph_data is not None:
            # Stack node embeddings
            node_embeddings = torch.stack([
                subreddit_embeddings[subreddit]
                for subreddit in graph_data.subreddit_names
                if subreddit in subreddit_embeddings
            ])

            # GNN forward pass
            gnn_output = self.gnn_model(
                node_embeddings,
                graph_data.edge_index,
                graph_data.edge_attr if hasattr(graph_data, 'edge_attr') else None
            )

            # Combine Level 1 and Level 2 predictions
            combined_features = torch.cat([
                torch.stack([subreddit_predictions[sr] for sr in subreddit_predictions]),
                gnn_output
            ], dim=-1)

            final_predictions = self.fusion_layer(combined_features)
        else:
            # Fallback: use only Level 1 predictions
            final_predictions = torch.stack([
                pred for pred in subreddit_predictions.values()
            ])

        return final_predictions

    def training_step(self, batch, batch_idx):
        subreddit_data, graph_data, targets = batch

        predictions = self(subreddit_data, graph_data)

        # Multi-task loss
        mse_loss = F.mse_loss(predictions[:, :2], targets[:, :2])  # Sentiment targets
        if predictions.shape[-1] > 2 and targets.shape[-1] > 2:
            ce_loss = F.cross_entropy(predictions[:, 2:], targets[:, 2:].long())  # Direction targets
            total_loss = mse_loss + ce_loss
        else:
            total_loss = mse_loss

        # Log metrics
        self.log('train_loss', total_loss, on_step=True, on_epoch=True)
        self.log('train_mse', mse_loss, on_step=True, on_epoch=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        subreddit_data, graph_data, targets = batch

        predictions = self(subreddit_data, graph_data)

        # Multi-task loss
        mse_loss = F.mse_loss(predictions[:, :2], targets[:, :2])
        if predictions.shape[-1] > 2 and targets.shape[-1] > 2:
            ce_loss = F.cross_entropy(predictions[:, 2:], targets[:, 2:].long())
            total_loss = mse_loss + ce_loss

            # Accuracy for direction prediction
            direction_preds = torch.argmax(predictions[:, 2:], dim=-1)
            accuracy = accuracy_score(
                targets[:, 2:].cpu().numpy(),
                direction_preds.cpu().numpy()
            )
            self.log('val_accuracy', accuracy, on_epoch=True)
        else:
            total_loss = mse_loss

        self.log('val_loss', total_loss, on_epoch=True)
        self.log('val_mse', mse_loss, on_epoch=True)

        return total_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }


class HierarchicalModelBuilder:
    """
    Builder class for constructing hierarchical sentiment models
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.scalers = {}
        self.subreddit_models = {}
        self.gnn_model = None

    def build_subreddit_models(self, train_data: Dict[str, SubredditTimeSeriesDataset]) -> Dict[str, nn.Module]:
        """Build individual subreddit models"""
        logger.info("Building subreddit-level models...")

        subreddit_models = {}

        for subreddit, dataset in train_data.items():
            input_dim = dataset.sequences.shape[-1]  # Feature dimension

            model_type = self.config.get('subreddit_model_type', 'lstm')

            if model_type == 'lstm':
                model = SubredditLSTM(
                    input_dim=input_dim,
                    hidden_dim=self.config.get('hidden_dim', 128),
                    num_layers=self.config.get('num_layers', 2),
                    output_dim=dataset.targets.shape[-1],
                    dropout=self.config.get('dropout', 0.2),
                    attention=self.config.get('attention', True)
                )
            elif model_type == 'transformer':
                model = SubredditTransformer(
                    input_dim=input_dim,
                    d_model=self.config.get('d_model', 128),
                    nhead=self.config.get('nhead', 8),
                    num_encoder_layers=self.config.get('num_encoder_layers', 3),
                    output_dim=dataset.targets.shape[-1],
                    dropout=self.config.get('dropout', 0.1)
                )
            else:
                raise ValueError(f"Unknown model type: {model_type}")

            subreddit_models[subreddit] = model
            logger.info(f"Built {model_type} model for {subreddit}")

        self.subreddit_models = subreddit_models
        return subreddit_models

    def build_gnn_model(self, node_features: int) -> SpilloverGNN:
        """Build graph neural network model"""
        logger.info("Building GNN model...")

        gnn_model = SpilloverGNN(
            node_features=node_features,
            edge_features=self.config.get('edge_features', 1),
            hidden_dim=self.config.get('gnn_hidden_dim', 64),
            num_layers=self.config.get('gnn_num_layers', 3),
            output_dim=self.config.get('gnn_output_dim', 4),
            gnn_type=self.config.get('gnn_type', 'GAT')
        )

        self.gnn_model = gnn_model
        return gnn_model

    def build_hierarchical_model(self, train_data: Dict[str, SubredditTimeSeriesDataset],
                               node_features: int) -> HierarchicalSentimentModel:
        """Build complete hierarchical model"""
        logger.info("Building hierarchical sentiment model...")

        # Build components
        subreddit_models = self.build_subreddit_models(train_data)
        gnn_model = self.build_gnn_model(node_features)

        # Create hierarchical model
        hierarchical_model = HierarchicalSentimentModel(
            subreddit_models=subreddit_models,
            gnn_model=gnn_model,
            feature_dim=self.config.get('hidden_dim', 128),
            learning_rate=self.config.get('learning_rate', 1e-3),
            weight_decay=self.config.get('weight_decay', 1e-4)
        )

        return hierarchical_model


class HierarchicalDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for hierarchical data
    """

    def __init__(self, processed_data: pd.DataFrame,
                 network: nx.DiGraph,
                 config: Dict[str, Any]):
        super().__init__()

        self.processed_data = processed_data
        self.network = network
        self.config = config

        self.train_datasets = {}
        self.val_datasets = {}
        self.test_datasets = {}

    def setup(self, stage: str = None):
        """Setup train/val/test datasets"""

        # Time-based split
        data_sorted = self.processed_data.sort_values('created_utc')

        train_end = int(0.7 * len(data_sorted))
        val_end = int(0.85 * len(data_sorted))

        train_data = data_sorted.iloc[:train_end]
        val_data = data_sorted.iloc[train_end:val_end]
        test_data = data_sorted.iloc[val_end:]

        # Feature columns
        feature_columns = [col for col in self.processed_data.columns
                          if any(pattern in col for pattern in
                          ['sentiment_', 'emotion_', 'ratio_', 'count_', 'net_spillover', 'pagerank'])]

        target_columns = ['next_sentiment', 'sentiment_change', 'next_return', 'return_direction']
        target_columns = [col for col in target_columns if col in self.processed_data.columns]

        # Create datasets for each subreddit
        for subreddit in self.processed_data['subreddit'].unique():

            # Training dataset
            self.train_datasets[subreddit] = SubredditTimeSeriesDataset(
                df=train_data,
                subreddit=subreddit,
                sequence_length=self.config.get('sequence_length', 24),
                prediction_horizon=self.config.get('prediction_horizon', 1),
                feature_columns=feature_columns,
                target_columns=target_columns
            )

            # Validation dataset
            self.val_datasets[subreddit] = SubredditTimeSeriesDataset(
                df=val_data,
                subreddit=subreddit,
                sequence_length=self.config.get('sequence_length', 24),
                prediction_horizon=self.config.get('prediction_horizon', 1),
                feature_columns=feature_columns,
                target_columns=target_columns
            )

            # Test dataset
            self.test_datasets[subreddit] = SubredditTimeSeriesDataset(
                df=test_data,
                subreddit=subreddit,
                sequence_length=self.config.get('sequence_length', 24),
                prediction_horizon=self.config.get('prediction_horizon', 1),
                feature_columns=feature_columns,
                target_columns=target_columns
            )

    def train_dataloader(self):
        # For now, return the first subreddit's dataloader
        # In practice, you'd implement custom batching across subreddits
        first_subreddit = list(self.train_datasets.keys())[0]
        return DataLoader(
            self.train_datasets[first_subreddit],
            batch_size=self.config.get('batch_size', 32),
            shuffle=True,
            num_workers=self.config.get('num_workers', 4)
        )

    def val_dataloader(self):
        first_subreddit = list(self.val_datasets.keys())[0]
        return DataLoader(
            self.val_datasets[first_subreddit],
            batch_size=self.config.get('batch_size', 32),
            shuffle=False,
            num_workers=self.config.get('num_workers', 4)
        )

    def test_dataloader(self):
        first_subreddit = list(self.test_datasets.keys())[0]
        return DataLoader(
            self.test_datasets[first_subreddit],
            batch_size=self.config.get('batch_size', 32),
            shuffle=False,
            num_workers=self.config.get('num_workers', 4)
        )


def main():
    """Example training pipeline"""

    # Configuration
    config = {
        'subreddit_model_type': 'lstm',
        'hidden_dim': 128,
        'num_layers': 2,
        'dropout': 0.2,
        'attention': True,
        'gnn_hidden_dim': 64,
        'gnn_num_layers': 3,
        'gnn_type': 'GAT',
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'batch_size': 32,
        'sequence_length': 24,
        'prediction_horizon': 1,
        'max_epochs': 100
    }

    # Load processed data (placeholder)
    # In practice, this would come from the hierarchical data processor
    logger.info("Loading processed data...")
    # processed_data = pd.read_parquet("data/processed/hierarchical/hierarchical_features.parquet")
    # network = nx.read_gml("data/processed/hierarchical/causality_network.gml")

    print("Hierarchical modeling pipeline ready!")
    print("To train models, provide processed data from hierarchical_data_processor.py")


if __name__ == "__main__":
    main()