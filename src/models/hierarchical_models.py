"""
Hierarchical Modeling Pipeline for Cryptocurrency Sentiment Analysis
Level 1: Subreddit-Level LSTM/Transformer Models
Level 2: Cross-Subreddit Graph Neural Networks

Based on academic methodology for financial spillover analysis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
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
import random

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
        # Use correct timestamp column
        timestamp_col = 'created_utc' if 'created_utc' in self.data.columns else 'post_created_utc'
        self.data = self.data.sort_values(timestamp_col).reset_index(drop=True)

        # Define features and targets - only include numeric columns
        if feature_columns is None:
            candidate_columns = [col for col in df.columns if any(
                pattern in col for pattern in ['sentiment_', 'emotion_', 'ratio_', 'count_', 'net_spillover', 'pagerank']
            )]
            # Filter to only numeric columns
            self.feature_columns = [col for col in candidate_columns
                                  if col in self.data.columns and
                                  self.data[col].dtype in ['int64', 'float64', 'int32', 'float32']]

            # Fallback: if no features found with patterns, use all numeric columns except timestamp and identifier columns
            if not self.feature_columns:
                logger.warning(f"No features found with standard patterns for {subreddit}, using numeric columns as fallback")
                exclude_cols = ['post_created_utc', 'created_utc', 'subreddit', 'id', 'post_id', 'comment_id']
                self.feature_columns = [col for col in self.data.columns
                                      if (self.data[col].dtype in ['int64', 'float64', 'int32', 'float32'] and
                                          col not in exclude_cols)]

            # Final validation: ensure we have at least one feature
            if not self.feature_columns:
                raise ValueError(f"No valid features found for subreddit {subreddit}. "
                               f"Available columns: {list(self.data.columns)}")

        else:
            self.feature_columns = feature_columns

        if target_columns is None:
            candidate_targets = ['next_sentiment', 'sentiment_change', 'next_return', 'return_direction']
            # Filter to only numeric target columns that exist
            self.target_columns = [col for col in candidate_targets
                                 if col in self.data.columns and
                                 self.data[col].dtype in ['int64', 'float64', 'int32', 'float32']]
        else:
            self.target_columns = target_columns

        # Remove missing target columns (additional safety)
        self.target_columns = [col for col in self.target_columns if col in self.data.columns]

        # Prepare sequences
        self._prepare_sequences()

    def _prepare_sequences(self):
        """Prepare sliding window sequences"""

        # Validate that we have features and targets
        if not self.feature_columns:
            raise ValueError(f"No feature columns available for {self.subreddit}")

        if not self.target_columns:
            logger.warning(f"No target columns found for {self.subreddit}, using compound_sentiment as fallback")
            if 'compound_sentiment' in self.data.columns:
                self.target_columns = ['compound_sentiment']
            else:
                # Create a simple target from first feature column
                target_col = self.feature_columns[0]
                self.data[f'next_{target_col}'] = self.data[target_col].shift(-1)
                self.target_columns = [f'next_{target_col}']

        # Extract feature and target arrays - force numeric conversion
        features = self.data[self.feature_columns].apply(pd.to_numeric, errors='coerce').values
        targets = self.data[self.target_columns].apply(pd.to_numeric, errors='coerce').values

        # Validate shapes
        if features.shape[1] == 0:
            raise ValueError(f"Feature matrix has 0 columns for {self.subreddit}. "
                           f"Feature columns: {self.feature_columns}")

        # Handle missing values
        features = np.nan_to_num(features, nan=0.0)
        targets = np.nan_to_num(targets, nan=0.0)

        self.sequences = []
        self.targets = []

        # Ensure we have enough data for at least one sequence
        min_required_length = self.sequence_length + self.prediction_horizon
        if len(features) < min_required_length:
            logger.warning(f"Insufficient data for {self.subreddit}: {len(features)} < {min_required_length}")
            # Create minimal sequences with available data
            self.sequences = np.array([features], dtype=np.float32)
            self.targets = np.array([targets], dtype=np.float32)
        else:
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
        logger.info(f"Feature columns ({len(self.feature_columns)}): {self.feature_columns[:5]}{'...' if len(self.feature_columns) > 5 else ''}")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.sequences[idx]), torch.FloatTensor(self.targets[idx])


class HierarchicalBatchSampler(Sampler):
    """
    Custom batch sampler for hierarchical model
    Samples from multiple subreddit datasets to create balanced batches
    """

    def __init__(self, dataset: 'HierarchicalDataset',
                 batch_size: int = 32, shuffle: bool = True):
        self.dataset = dataset
        self.datasets = dataset.datasets
        self.subreddits = list(self.datasets.keys())
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Calculate total samples and samples per subreddit
        self.total_samples = sum(len(ds) for ds in self.datasets.values())

        # Create mapping from (subreddit, local_idx) to global_idx
        self.global_index_map = {}
        for global_idx, (subreddit, local_idx) in enumerate(dataset.index_map):
            self.global_index_map[(subreddit, local_idx)] = global_idx
        self.samples_per_subreddit = {sr: len(ds) for sr, ds in self.datasets.items()}

        # Calculate how many samples from each subreddit per batch
        self.samples_per_batch = max(1, batch_size // len(self.subreddits))

    def __iter__(self):
        if self.shuffle:
            # Create shuffled indices for each subreddit
            subreddit_indices = {}
            for subreddit in self.subreddits:
                indices = list(range(len(self.datasets[subreddit])))
                random.shuffle(indices)
                subreddit_indices[subreddit] = indices
        else:
            subreddit_indices = {sr: list(range(len(ds)))
                               for sr, ds in self.datasets.items()}

        # Track current position for each subreddit
        positions = {sr: 0 for sr in self.subreddits}

        while any(positions[sr] < len(subreddit_indices[sr]) for sr in self.subreddits):
            batch = []

            for subreddit in self.subreddits:
                if positions[subreddit] < len(subreddit_indices[subreddit]):
                    # Take samples_per_batch samples from this subreddit
                    end_pos = min(positions[subreddit] + self.samples_per_batch,
                                 len(subreddit_indices[subreddit]))

                    for i in range(positions[subreddit], end_pos):
                        local_idx = subreddit_indices[subreddit][i]
                        global_idx = self.global_index_map[(subreddit, local_idx)]
                        batch.append(global_idx)

                    positions[subreddit] = end_pos

            if batch:
                yield batch

    def __len__(self):
        return self.total_samples // self.batch_size


class HierarchicalDataset(Dataset):
    """
    Wrapper dataset that combines multiple subreddit datasets
    """

    def __init__(self, datasets: Dict[str, SubredditTimeSeriesDataset]):
        self.datasets = datasets
        self.subreddits = list(datasets.keys())
        # Create index mapping
        self.index_map = []
        for subreddit, dataset in datasets.items():
            for idx in range(len(dataset)):
                self.index_map.append((subreddit, idx))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        subreddit, dataset_idx = self.index_map[idx]
        data = self.datasets[subreddit][dataset_idx]
        return (subreddit, data)


class HierarchicalCollator:
    """
    Collator class for hierarchical batches that has access to datasets
    """

    def __init__(self, datasets: Dict[str, SubredditTimeSeriesDataset],
                 network: nx.DiGraph = None):
        self.datasets = datasets
        self.network = network
        self.subreddits = list(datasets.keys())

        # Validate graph structure on initialization
        if not self.validate_graph_structure():
            logger.warning("Graph structure validation failed, graph operations may not work correctly")

    def __call__(self, batch_items):
        """
        Custom collate function for hierarchical batches
        """
        subreddit_data = {}
        all_targets = []

        # Group by subreddit and extract data
        subreddit_sequences = {}
        subreddit_targets = {}

        for subreddit, data_item in batch_items:
            if subreddit not in subreddit_sequences:
                subreddit_sequences[subreddit] = []
                subreddit_targets[subreddit] = []

            # data_item is (sequence, target)
            sequence, target = data_item
            subreddit_sequences[subreddit].append(sequence)
            subreddit_targets[subreddit].append(target)

        # Stack sequences for each subreddit
        for subreddit in subreddit_sequences:
            if subreddit_sequences[subreddit]:
                subreddit_data[subreddit] = torch.stack(subreddit_sequences[subreddit])
                all_targets.extend(subreddit_targets[subreddit])

        # Create graph data if network is available - pass available subreddits
        available_subreddits = list(subreddit_data.keys()) if subreddit_data else None
        graph_data = self._create_graph_data(available_subreddits) if self.network else None

        # Concatenate all targets (handle variable lengths)
        if all_targets:
            # Flatten all targets to handle variable sequence lengths
            flattened_targets = []
            for target in all_targets:
                if target.dim() > 1:
                    # If target has multiple dimensions, flatten to 1D
                    flattened_targets.append(target.view(-1))
                else:
                    flattened_targets.append(target)
            targets_tensor = torch.cat(flattened_targets, dim=0)
        else:
            targets_tensor = torch.empty(0)

        return subreddit_data, graph_data, targets_tensor

    def _create_graph_data(self, available_subreddits=None):
        """Create graph representation from network with proper node indexing

        Args:
            available_subreddits: List of subreddits that have actual data/embeddings
                                If None, uses all subreddits from datasets
        """
        if not self.network:
            return None

        # Use available subreddits if provided, otherwise use all subreddits
        if available_subreddits is None:
            available_subreddits = self.subreddits

        # Filter to only include subreddits that exist in both datasets and network
        network_nodes = set(self.network.nodes())
        valid_subreddits = [sr for sr in available_subreddits if sr in network_nodes]

        if not valid_subreddits:
            logger.warning("No valid subreddits found in both datasets and network, creating fallback graph")
            return self.create_fallback_graph(available_subreddits if available_subreddits else [])

        logger.info(f"Creating graph data for {len(valid_subreddits)} valid subreddits: {valid_subreddits}")

        # Create a simple object to hold graph data
        class GraphData:
            def __init__(self):
                pass

        graph_data = GraphData()
        graph_data.subreddit_names = valid_subreddits
        graph_data.num_nodes = len(valid_subreddits)

        # Create edge_index for PyTorch Geometric format with ONLY valid subreddits
        edge_list = []
        edge_weights = []
        subreddit_to_idx = {sr: i for i, sr in enumerate(valid_subreddits)}

        logger.info(f"Subreddit to index mapping: {subreddit_to_idx}")

        for u, v, data in self.network.edges(data=True):
            # Only include edges where both nodes are in valid_subreddits
            if u in subreddit_to_idx and v in subreddit_to_idx:
                i, j = subreddit_to_idx[u], subreddit_to_idx[v]
                edge_list.append([i, j])

                # Handle complex edge data - extract scalar weight
                weight = data.get('weight', 1.0)
                if hasattr(weight, '__len__') and not isinstance(weight, str):
                    # If weight is array-like, take the mean or first element
                    if hasattr(weight, 'mean'):
                        weight = float(weight.mean())
                    else:
                        weight = float(weight[0]) if len(weight) > 0 else 1.0
                edge_weights.append(float(weight))

        logger.info(f"Created {len(edge_list)} edges for graph")

        # Validate edge indices
        if edge_list:
            max_index = max(max(edge) for edge in edge_list)
            if max_index >= len(valid_subreddits):
                raise ValueError(f"Edge index {max_index} >= num_nodes {len(valid_subreddits)}")

            graph_data.edge_index = torch.tensor(edge_list, dtype=torch.long).t()
            graph_data.edge_attr = torch.tensor(edge_weights, dtype=torch.float)
        else:
            # No edges from network, try fallback graph if we have multiple nodes
            if len(valid_subreddits) > 1:
                logger.warning("No valid edges found in network graph, creating fallback connections")
                fallback_graph = self.create_fallback_graph(valid_subreddits)
                graph_data.edge_index = fallback_graph.edge_index
                graph_data.edge_attr = fallback_graph.edge_attr
            else:
                # Single node or no nodes, create empty tensors
                graph_data.edge_index = torch.empty((2, 0), dtype=torch.long)
                graph_data.edge_attr = torch.empty(0, dtype=torch.float)
                logger.warning("No valid edges found and insufficient nodes for fallback graph")

        return graph_data

    def _create_empty_graph_data(self):
        """Create empty graph data structure"""
        class GraphData:
            def __init__(self):
                pass

        graph_data = GraphData()
        graph_data.subreddit_names = []
        graph_data.num_nodes = 0
        graph_data.edge_index = torch.empty((2, 0), dtype=torch.long)
        graph_data.edge_attr = torch.empty(0, dtype=torch.float)
        return graph_data

    def validate_graph_structure(self):
        """Validate the graph structure against available datasets"""
        if not self.network:
            logger.info("No network provided, skipping graph validation")
            return True

        network_nodes = set(self.network.nodes())
        dataset_subreddits = set(self.subreddits)

        # Check for missing subreddits
        missing_in_network = dataset_subreddits - network_nodes
        missing_in_datasets = network_nodes - dataset_subreddits

        if missing_in_network:
            logger.warning(f"Subreddits in datasets but not in network: {missing_in_network}")

        if missing_in_datasets:
            logger.warning(f"Subreddits in network but not in datasets: {missing_in_datasets}")

        # Check connectivity
        valid_subreddits = dataset_subreddits & network_nodes
        subgraph = self.network.subgraph(valid_subreddits)

        if len(subgraph.nodes()) == 0:
            logger.error("No valid subreddits found in both datasets and network")
            return False

        if len(subgraph.edges()) == 0:
            logger.warning("No edges found between valid subreddits")

        logger.info(f"Graph validation: {len(subgraph.nodes())} nodes, {len(subgraph.edges())} edges")
        return True

    def create_fallback_graph(self, available_subreddits):
        """Create a minimal graph structure when network is unavailable or invalid

        Args:
            available_subreddits: List of subreddits that have data
        """
        if len(available_subreddits) <= 1:
            return self._create_empty_graph_data()

        logger.info(f"Creating fallback fully connected graph for {len(available_subreddits)} subreddits")

        class GraphData:
            def __init__(self):
                pass

        graph_data = GraphData()
        graph_data.subreddit_names = available_subreddits
        graph_data.num_nodes = len(available_subreddits)

        # Create a fully connected graph with uniform weights
        edge_list = []
        edge_weights = []

        for i in range(len(available_subreddits)):
            for j in range(len(available_subreddits)):
                if i != j:  # No self-loops
                    edge_list.append([i, j])
                    edge_weights.append(1.0)

        if edge_list:
            graph_data.edge_index = torch.tensor(edge_list, dtype=torch.long).t()
            graph_data.edge_attr = torch.tensor(edge_weights, dtype=torch.float)
        else:
            graph_data.edge_index = torch.empty((2, 0), dtype=torch.long)
            graph_data.edge_attr = torch.empty(0, dtype=torch.float)

        return graph_data


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
        Forward pass with robust edge index validation
        Args:
            x: Node features (num_nodes, node_features)
            edge_index: Edge indices (2, num_edges)
            edge_attr: Edge attributes (num_edges, edge_features)
            batch: Batch assignment for batched processing
        """

        # Validate inputs
        if x.size(0) == 0:
            logger.warning("Empty node features tensor, returning zeros")
            return torch.zeros(0, self.hidden_dim, device=x.device)

        # Validate edge indices are within bounds
        if edge_index.numel() > 0:
            max_edge_idx = edge_index.max().item()
            if max_edge_idx >= x.size(0):
                logger.error(f"Edge index {max_edge_idx} >= num_nodes {x.size(0)}")
                # Create empty edge_index if all edges are invalid
                edge_index = torch.empty((2, 0), dtype=torch.long, device=edge_index.device)
                if edge_attr is not None:
                    edge_attr = torch.empty(0, edge_attr.size(-1), device=edge_attr.device)

            # Check for negative indices
            if (edge_index < 0).any():
                logger.warning("Found negative edge indices, filtering them out")
                valid_mask = (edge_index >= 0).all(dim=0)
                edge_index = edge_index[:, valid_mask]
                if edge_attr is not None and edge_attr.numel() > 0:
                    edge_attr = edge_attr[valid_mask]

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
                # Check if the layer supports edge attributes
                try:
                    if edge_attr is not None and hasattr(self, 'edge_mlp'):
                        h_new = gnn_layer(h, edge_index, edge_attr=edge_attr)
                    else:
                        h_new = gnn_layer(h, edge_index)
                except (TypeError, RuntimeError, IndexError) as e:
                    # Layer doesn't support edge_attr or has index issues, try without it
                    logger.warning(f"GNN layer {i} failed with edge_attr, trying without: {e}")
                    try:
                        h_new = gnn_layer(h, edge_index)
                    except (RuntimeError, IndexError) as e2:
                        logger.error(f"GNN layer {i} failed completely: {e2}")
                        # Skip this layer and continue with previous h
                        continue

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

        # Level 2: Graph neural network with robust error handling
        if graph_data is not None and graph_data.num_nodes > 0:
            # Get subreddits that have both embeddings and are in graph
            available_subreddits = [
                subreddit for subreddit in graph_data.subreddit_names
                if subreddit in subreddit_embeddings
            ]

            if len(available_subreddits) > 0:
                logger.info(f"Processing GNN with {len(available_subreddits)} nodes: {available_subreddits}")

                # Stack node embeddings in the same order as graph_data.subreddit_names
                node_embeddings = torch.stack([
                    subreddit_embeddings[subreddit]
                    for subreddit in available_subreddits
                ])

                # Validate node embeddings shape
                if node_embeddings.size(0) != len(available_subreddits):
                    raise ValueError(f"Node embeddings size {node_embeddings.size(0)} != expected {len(available_subreddits)}")

                # Validate edge indices are within bounds
                if graph_data.edge_index.numel() > 0:
                    max_edge_idx = graph_data.edge_index.max().item()
                    if max_edge_idx >= node_embeddings.size(0):
                        logger.warning(f"Edge index {max_edge_idx} >= num_nodes {node_embeddings.size(0)}, filtering edges")
                        # Filter out invalid edges
                        valid_edges_mask = (graph_data.edge_index < node_embeddings.size(0)).all(dim=0)
                        graph_data.edge_index = graph_data.edge_index[:, valid_edges_mask]
                        if hasattr(graph_data, 'edge_attr') and graph_data.edge_attr.numel() > 0:
                            graph_data.edge_attr = graph_data.edge_attr[valid_edges_mask]
                        logger.info(f"Filtered to {graph_data.edge_index.size(1)} valid edges")

                try:
                    # GNN forward pass
                    gnn_output = self.gnn_model(
                        node_embeddings,
                        graph_data.edge_index,
                        graph_data.edge_attr if hasattr(graph_data, 'edge_attr') else None
                    )

                    # Combine Level 1 and Level 2 predictions
                    # Ensure compatible shapes for concatenation
                    l1_preds = torch.stack([subreddit_predictions[sr] for sr in available_subreddits])

                    if l1_preds.shape[0] == gnn_output.shape[0]:
                        combined_features = torch.cat([l1_preds, gnn_output], dim=-1)
                        final_predictions = self.fusion_layer(combined_features)
                    else:
                        logger.warning(f"Shape mismatch: L1 preds {l1_preds.shape}, GNN output {gnn_output.shape}")
                        # Fallback to L1 only
                        final_predictions = l1_preds

                except Exception as e:
                    logger.error(f"GNN forward pass failed: {e}")
                    # Fallback to Level 1 predictions
                    final_predictions = torch.stack([
                        pred for pred in subreddit_predictions.values()
                    ])
            else:
                logger.warning("No subreddits with embeddings found in graph, using L1 predictions only")
                # Fallback: use only Level 1 predictions
                final_predictions = torch.stack([
                    pred for pred in subreddit_predictions.values()
                ])
        else:
            logger.info("No graph data available or empty graph, using L1 predictions only")
            # Fallback: use only Level 1 predictions
            final_predictions = torch.stack([
                pred for pred in subreddit_predictions.values()
            ])

        return final_predictions

    def training_step(self, batch, batch_idx):
        """Training step with proper hierarchical batch structure"""
        subreddit_data, graph_data, targets = batch
        predictions = self(subreddit_data, graph_data)

        # Multi-task loss - handle different tensor shapes
        if targets.dim() == 1:
            # Flatten predictions to match targets if needed
            if predictions.dim() > 1:
                predictions = predictions.view(-1)

            # Handle size mismatch by taking minimum size
            min_size = min(predictions.size(0), targets.size(0))
            if predictions.size(0) != targets.size(0):
                logger.warning(f"Size mismatch: predictions {predictions.size(0)} vs targets {targets.size(0)}, using first {min_size}")
                predictions = predictions[:min_size]
                targets = targets[:min_size]

            mse_loss = F.mse_loss(predictions, targets)
            total_loss = mse_loss
        else:
            # 2D case
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
        """Validation step with proper hierarchical batch structure"""
        subreddit_data, graph_data, targets = batch
        predictions = self(subreddit_data, graph_data)

        # Multi-task loss - handle different tensor shapes
        if targets.dim() == 1:
            # Flatten predictions to match targets if needed
            if predictions.dim() > 1:
                predictions = predictions.view(-1)

            # Handle size mismatch by taking minimum size
            min_size = min(predictions.size(0), targets.size(0))
            if predictions.size(0) != targets.size(0):
                logger.warning(f"Size mismatch: predictions {predictions.size(0)} vs targets {targets.size(0)}, using first {min_size}")
                predictions = predictions[:min_size]
                targets = targets[:min_size]

            mse_loss = F.mse_loss(predictions, targets)
            total_loss = mse_loss
        else:
            # 2D case
            mse_loss = F.mse_loss(predictions[:, :2], targets[:, :2])
            if predictions.shape[-1] > 2 and targets.shape[-1] > 2:
                ce_loss = F.cross_entropy(predictions[:, 2:], targets[:, 2:].long())
                total_loss = mse_loss + ce_loss
            else:
                total_loss = mse_loss

            # Accuracy for direction prediction
            direction_preds = torch.argmax(predictions[:, 2:], dim=-1)
            accuracy = accuracy_score(
                targets[:, 2:].cpu().numpy(),
                direction_preds.cpu().numpy()
            )
            self.log('val_accuracy', accuracy, on_epoch=True)

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
            optimizer, mode='min', factor=0.5, patience=5
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
            edge_features=self.config.get('edge_features', 0),  # Disable edge features for now
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

        # Time-based split - use correct timestamp column
        timestamp_col = 'created_utc' if 'created_utc' in self.processed_data.columns else 'post_created_utc'
        data_sorted = self.processed_data.sort_values(timestamp_col)

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
        """Create hierarchical DataLoader for training"""
        if not self.train_datasets:
            raise ValueError("No training datasets available. Run setup() first.")

        # Create hierarchical dataset wrapper
        hierarchical_dataset = HierarchicalDataset(self.train_datasets)

        # Create custom batch sampler
        batch_sampler = HierarchicalBatchSampler(
            dataset=hierarchical_dataset,
            batch_size=self.config.get('batch_size', 32),
            shuffle=True
        )

        # Create custom collator
        collator = HierarchicalCollator(
            datasets=self.train_datasets,
            network=self.network
        )

        # Create DataLoader with custom sampler and collator
        return DataLoader(
            dataset=hierarchical_dataset,
            batch_sampler=batch_sampler,
            collate_fn=collator,
            num_workers=self.config.get('num_workers', 0)  # Set to 0 for custom sampler
        )

    def val_dataloader(self):
        """Create hierarchical DataLoader for validation"""
        if not self.val_datasets:
            raise ValueError("No validation datasets available. Run setup() first.")

        # Create hierarchical dataset wrapper
        hierarchical_dataset = HierarchicalDataset(self.val_datasets)

        # Create custom batch sampler
        batch_sampler = HierarchicalBatchSampler(
            dataset=hierarchical_dataset,
            batch_size=self.config.get('batch_size', 32),
            shuffle=False
        )

        # Create custom collator
        collator = HierarchicalCollator(
            datasets=self.val_datasets,
            network=self.network
        )

        # Create DataLoader with custom sampler and collator
        return DataLoader(
            dataset=hierarchical_dataset,
            batch_sampler=batch_sampler,
            collate_fn=collator,
            num_workers=self.config.get('num_workers', 0)  # Set to 0 for custom sampler
        )

    def test_dataloader(self):
        """Create hierarchical DataLoader for testing"""
        if not self.test_datasets:
            raise ValueError("No test datasets available. Run setup() first.")

        # Create hierarchical dataset wrapper
        hierarchical_dataset = HierarchicalDataset(self.test_datasets)

        # Create custom batch sampler
        batch_sampler = HierarchicalBatchSampler(
            dataset=hierarchical_dataset,
            batch_size=self.config.get('batch_size', 32),
            shuffle=False
        )

        # Create custom collator
        collator = HierarchicalCollator(
            datasets=self.test_datasets,
            network=self.network
        )

        # Create DataLoader with custom sampler and collator
        return DataLoader(
            dataset=hierarchical_dataset,
            batch_sampler=batch_sampler,
            collate_fn=collator,
            num_workers=self.config.get('num_workers', 0)  # Set to 0 for custom sampler
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