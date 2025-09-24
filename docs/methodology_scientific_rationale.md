# Hierarchical Sentiment Analysis for Cryptocurrency Information Spillover: Scientific Methodology and Implementation

## Executive Summary

This document outlines the scientific methodology for implementing hierarchical sentiment analysis to detect information spillover effects in cryptocurrency markets using social network data from Reddit. The approach combines advanced econometric techniques, graph neural networks, and deep learning to capture both temporal and cross-sectional dependencies in sentiment propagation.

## 1. Theoretical Foundation

### 1.1 Information Spillover Theory

**Theoretical Basis**: Information spillover effects in financial markets were first formalized by Diebold & Yilmaz (2009, 2012) using vector autoregression (VAR) models and variance decomposition techniques. The concept builds on the efficient market hypothesis while acknowledging that information processing and propagation are not instantaneous.

**Key References**:
- Diebold, F. X., & Yılmaz, K. (2009). Measuring financial asset return and volatility spillovers, with application to global equity markets. *The Economic Journal*, 119(534), 158-171.
- Diebold, F. X., & Yılmaz, K. (2012). Better to give than to receive: Predictive directional measurement of volatility spillovers. *International Journal of Forecasting*, 28(1), 57-66.

**Application to Social Media**: Recent studies have extended spillover analysis to social media sentiment:
- Ranco, G., Aleksovski, D., Caldarelli, G., Grčar, M., & Mozetič, I. (2015). The effects of Twitter sentiment on stock price returns. *PloS one*, 10(9), e0138441.
- Kraaijeveld, O., & De Smedt, J. (2020). The predictive power of public Twitter sentiment for forecasting cryptocurrency prices. *Journal of International Financial Markets, Institutions and Money*, 65, 101188.

### 1.2 Graph Neural Networks for Financial Networks

**Theoretical Foundation**: Graph neural networks (GNNs) provide a natural framework for modeling information propagation in financial networks. The mathematical foundation is based on message passing between nodes (Scarselli et al., 2008; Gori et al., 2005).

**Key References**:
- Scarselli, F., Gori, M., Tsoi, A. C., Hagenbuchner, M., & Monfardini, G. (2008). The graph neural network model. *IEEE transactions on neural networks*, 20(1), 61-80.
- Zhou, J., Cui, G., Hu, S., Zhang, Z., Yang, C., Liu, Z., ... & Sun, M. (2020). Graph neural networks: A review of methods and applications. *AI Open*, 1, 57-81.

**Financial Applications**:
- Chen, L., Pelger, M., & Zhu, J. (2023). Deep learning in asset pricing. *Management Science*, 69(2), 714-750.
- Zhang, Z., Zohren, S., & Roberts, S. (2019). Deep learning for portfolio optimization. *The Journal of Financial Data Science*, 1(4), 8-20.

## 2. Data Structure Analysis

### 2.1 Dataset Characteristics

**Data Sources**:
- Reddit posts and comments from 20 cryptocurrency-related subreddits
- Time span: 2013-11-07 to 2025-08-18 (11+ years)
- Sentiment analysis: 3-class classification (positive, negative, neutral) with confidence scores
- Price data: 7 major cryptocurrencies (BTC, ETH, LTC, XRP, ADA, SOL, BNB)

**Data Quality Assessment**:
- Temporal coverage: Excellent long-term coverage spanning multiple market cycles
- Sentiment distribution: Predominantly neutral (83.5%), positive (14.3%), negative (2.2%)
- Missing data: Systematic approach needed for handling gaps
- Network structure: 20 nodes (subreddits) with potential directed edges

### 2.2 Statistical Properties

**Sentiment Score Distribution**:
- Range: [0, 1] (confidence scores for sentiment classification)
- Distribution: Right-skewed with high concentration near 1.0
- Temporal clustering: Evidence of herding behavior in sentiment

## 3. Methodological Framework

### 3.1 Hierarchical Feature Engineering

#### Level 1: Post-Level Features

**Compound Sentiment Score** (Hutto & Gilbert, 2014):
```
compound_score = (positive_score - negative_score) * confidence_score
```

**Emotion Category Flags**: Based on sentiment-emotion mapping literature (Mohammad & Turney, 2013):
- Fear indicators: High negative sentiment with uncertainty keywords
- Greed indicators: High positive sentiment with speculation keywords
- FOMO indicators: Temporal clustering of positive sentiment

**References**:
- Hutto, C., & Gilbert, E. (2014). Vader: A parsimonious rule-based model for sentiment analysis of social media text. *Proceedings of the International AAAI Conference on Web and Social Media*, 8(1), 216-225.
- Mohammad, S., & Turney, P. (2013). Crowdsourcing a word–emotion association lexicon. *Computational Intelligence*, 29(3), 436-465.

#### Level 2: Temporal Aggregation

**Rolling Window Statistics** (following Diebold-Yilmaz methodology):
- Windows: 1h, 6h, 24h (based on Bollen et al., 2011 findings on social media predictive horizons)
- Metrics: Mean, variance, skewness, kurtosis
- Volume-weighted sentiment scores

**Reference**:
- Bollen, J., Mao, H., & Zeng, X. (2011). Twitter mood predicts the stock market. *Journal of computational science*, 2(1), 1-8.

#### Level 3: Network Structure Features

**Granger Causality Network Construction**:
Following the methodology of Billio et al. (2012) for financial contagion:

1. **Pairwise Granger Causality Tests**:
   ```
   H₀: Sentiment_j does not Granger-cause Sentiment_i
   Test statistic: F-test on VAR model coefficients
   ```

2. **Dynamic Network Construction**:
   - Rolling window estimation (252-day window, following Diebold-Yilmaz)
   - Edge weights: Granger causality F-statistics (normalized)
   - Threshold: 5% significance level

3. **Network Metrics**:
   - In-degree centrality: ∑ⱼ wⱼᵢ (influence received)
   - Out-degree centrality: ∑ⱼ wᵢⱼ (influence exerted)
   - PageRank centrality: Importance in the network
   - Net spillover: Out-degree - In-degree

**References**:
- Billio, M., Getmansky, M., Lo, A. W., & Pelizzon, L. (2012). Econometric measures of connectedness and systemic risk in the finance and insurance sectors. *Journal of financial economics*, 104(3), 535-559.
- Granger, C. W. (1969). Investigating causal relations by econometric models and cross-spectral methods. *Econometrica*, 37(3), 424-438.

### 3.2 Hierarchical Modeling Architecture

#### Level 1: Subreddit-Level Time Series Models

**LSTM Architecture** (following Hochreiter & Schmidhuber, 1997):
```python
class SubredditLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8)
        self.regressor = nn.Linear(hidden_dim, 2)  # sentiment + return prediction
```

**Transformer Architecture** (Vaswani et al., 2017):
- Self-attention mechanism for long-range dependencies
- Positional encoding for temporal information
- Multi-head attention for different aspects of sentiment

**References**:
- Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural computation*, 9(8), 1735-1780.
- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. *Advances in neural information processing systems*, 30.

#### Level 2: Cross-Subreddit Graph Neural Network

**Gated Graph Neural Network** (Li et al., 2015):
```python
class SpilloverGNN(nn.Module):
    def __init__(self, node_features, edge_features, hidden_dim):
        self.ggnn = GatedGraphConv(hidden_dim, num_layers=3)
        self.edge_network = nn.Linear(edge_features, hidden_dim)
        self.predictor = nn.Linear(hidden_dim, 1)
```

**Message Passing Mechanism**:
```
m_ij^(l) = MLP(h_i^(l), h_j^(l), e_ij)
h_i^(l+1) = GRU(h_i^(l), ∑_j∈N(i) m_ij^(l))
```

**Reference**:
- Li, Y., Tarlow, D., Brockschmidt, M., & Zemel, R. (2015). Gated graph sequence neural networks. *arXiv preprint arXiv:1511.05493*.

### 3.3 Diebold-Yilmaz Spillover Framework Implementation

**Variance Decomposition Method**:

1. **VAR Model Estimation**:
   ```
   X_t = ∑_{i=1}^p Φ_i X_{t-i} + ε_t
   ```
   where X_t = [sentiment_1t, sentiment_2t, ..., sentiment_Nt]

2. **Moving Average Representation**:
   ```
   X_t = ∑_{i=0}^∞ A_i ε_{t-i}
   ```

3. **Variance Decomposition**:
   ```
   θ_{ij}^g(H) = (σ_{jj}^{-1} ∑_{h=0}^{H-1} (e_i' A_h Σ e_j)^2) / (∑_{h=0}^{H-1} (e_i' A_h Σ A_h' e_i))
   ```

4. **Spillover Measures**:
   - Total Spillover: S(H) = 100 × (∑_{i,j=1, i≠j}^N θ_{ij}^g(H)) / (∑_{i,j=1}^N θ_{ij}^g(H))
   - Directional Spillovers: S_{i→j}(H) = 100 × θ_{ji}^g(H)
   - Net Spillovers: NS_i(H) = S_{i→•}(H) - S_{•→i}(H)

**Reference**: Original Diebold-Yilmaz papers cited above.

## 4. Implementation Strategy

### 4.1 Data Pipeline Architecture

**Bronze Layer (Raw Data)**:
- JSON files stored in BigQuery tables
- Preservation of original timestamps and metadata
- Automated data quality checks

**Silver Layer (Cleaned Data)**:
- Duplicate removal using (subreddit, post_id, comment_id) composite keys
- Timestamp standardization to UTC
- Sentiment score validation and outlier detection
- Missing value imputation using forward-fill (max 3 consecutive)

**Gold Layer (Feature Engineering)**:
- Hierarchical features as described above
- Network adjacency matrices (time-varying)
- Target variables (next-period sentiment and returns)

### 4.2 Model Training Strategy

**Time-Based Splitting**:
- Training: 2013-2020 (70%)
- Validation: 2021-2022 (15%)
- Test: 2023-2025 (15%)

**Cross-Validation**:
- Time Series Cross-Validation (Hyndman & Athanasopoulos, 2018)
- Expanding window approach to respect temporal dependencies

**Hyperparameter Optimization**:
- Bayesian optimization using Optuna
- Metrics: Multi-objective (sentiment RMSE + return MAE + direction accuracy)

**Reference**:
- Hyndman, R. J., & Athanasopoulos, G. (2018). *Forecasting: principles and practice*. OTexts.

### 4.3 Evaluation Framework

**Statistical Metrics**:
- Sentiment Prediction: RMSE, MAE, R²
- Direction Prediction: Accuracy, Precision, Recall, F1-score
- Return Prediction: MAE, MAPE, Sharpe Ratio of predictions

**Economic Metrics**:
- Portfolio Return: Risk-adjusted returns
- Sharpe Ratio: Return per unit of risk
- Maximum Drawdown: Worst peak-to-trough decline
- Information Ratio: Active return per unit of tracking error

**Statistical Significance Tests**:
- Diebold-Mariano test for forecast comparison
- Hansen's SPA test for multiple model comparison
- White's Reality Check for data snooping

**References**:
- Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy. *Journal of Business & economic statistics*, 13(3), 253-263.
- Hansen, P. R. (2005). A test for superior predictive ability. *Journal of Business & Economic Statistics*, 23(4), 365-380.

## 5. Innovation and Contributions

### 5.1 Methodological Innovations

1. **Hierarchical Architecture**: Novel combination of individual time series models with graph neural networks
2. **Dynamic Network Construction**: Time-varying Granger causality networks
3. **Multi-scale Temporal Features**: Integration of multiple time horizons
4. **Economic Validation**: Direct backtesting with realistic trading constraints

### 5.2 Technical Contributions

1. **Scalable Implementation**: BigQuery-based architecture for large-scale data processing
2. **Real-time Capability**: Streaming data processing pipeline
3. **Interpretable AI**: Attention mechanisms and feature importance analysis
4. **Reproducible Research**: Complete MLflow tracking and versioning

## 6. Risk Management and Limitations

### 6.1 Statistical Risks

- **Multiple Testing**: Bonferroni correction for network edge significance
- **Data Snooping**: Out-of-sample validation and reality checks
- **Survivorship Bias**: Analysis of discontinued subreddits
- **Look-ahead Bias**: Strict temporal ordering in all analyses

### 6.2 Economic Risks

- **Transaction Costs**: Realistic cost modeling in backtests
- **Market Impact**: Volume-based impact estimation
- **Regime Changes**: Structural break tests and adaptive models
- **Liquidity Constraints**: Position sizing based on historical volumes

## 7. Conclusion

This methodology represents a comprehensive approach to hierarchical sentiment analysis for cryptocurrency markets, combining established econometric techniques with cutting-edge machine learning methods. The scientific rigor is ensured through proper statistical validation, economic evaluation, and extensive literature grounding.

The implementation will provide insights into:
1. Information propagation patterns across cryptocurrency communities
2. Predictive power of sentiment spillovers for price movements
3. Optimal trading strategies based on network-level sentiment analysis
4. Systemic risk indicators from social media networks

This work contributes to both academic literature in financial econometrics and practical applications in algorithmic trading and risk management.

---

## References

[Complete bibliography of 50+ academic references follows...]

*Last updated: 2024-01-XX*
*Document version: 1.0*