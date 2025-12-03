# Comparative-Analysis-of-LSTM-Based-Models-for-Groundwater-Level-Prediction-in-Yunlin-Taiwan
Final report of Stochastic Subsurface Hydrology, June 2025 @ NCKU Resource Engineering

![Made with Python](https://img.shields.io/badge/Made%20with-Python-blue?logo=python)
![Colab](https://img.shields.io/badge/Google-Colab-yellow?logo=googlecolab)

- Conducted a comparative study on the applicability of LSTM, BiLSTM, CNN-LSTM, and CNN-BiLSTM models with varying temporal resolutions for groundwater level prediction in Taiwan.  
- Utilized observation data from groundwater monitoring wells and implemented the models using Python.

## Abstract

This work investigates the performance of four deep-learning architectures—LSTM, BiLSTM, CNN-LSTM, and CNN-BiLSTM—for daily groundwater-level simulation at the Yuanchang station (Yunlin County, Taiwan). Experiments compare model performance across four training dataset lengths (1, 5, 10, and 20 years) and evaluate robustness under missing data imputed with the BEAST algorithm. Models are compared using MSE, RMSE and R². Key findings: LSTM yields the most consistent performance overall; BiLSTM better captures fine-scale oscillations; CNN-based hybrids do not always outperform RNN-only models for long and highly variable time series. 

### Research background & motivation

- Groundwater is a critical water source during droughts; accurate forecasting of groundwater levels supports irrigation management, subsidence mitigation, and policy decisions.

- Physically based simulators (e.g., MODFLOW) provide interpretability but struggle with sparse, highly nonlinear data.

- Data-driven deep learning (especially RNN/LSTM families) can learn complex temporal dependencies from observational time series and have shown promise in hydrological forecasting.

This project aims to systematically compare typical LSTM-based architectures across different historical data availabilities and to assess the effect of dataset length and variability on model performance. 

**Dataset**

Station: Yuanchang groundwater observation station, Yunlin County, Taiwan.

Period: 1994-11-01 to 2022-12-31 (daily records).

Total records: 10,289 daily observations.

Missing values: 571 days missing (handled as described below).

Source: Water Resources Agency, Ministry of Economic Affairs (as noted in the report). 

**Split data into four dataset-length scenarios for experiments:**

- 1-year dataset: 2022 only

- 5-year dataset: 2018–2022

- 10-year dataset: 2013–2022

- 20-year dataset: 2003–2022

### Data preprocessing (BEAST)

Missing values are imputed using BEAST (Bayesian Estimator of Abrupt Change, Seasonality, and Trend).

BEAST decomposes series into trend `T_i`, seasonality `S_i`, and noise `ε_i`.

Missing value imputation: `ŷ_i = T̂_i + Ŝ_i` for missing indices.

Rationale: BEAST captures structural changes and seasonality via a Bayesian ensemble approach, which preserves long-term and seasonal structure better than naive interpolation.

After imputation, series are normalized (e.g., z-score or min-max) before being fed into neural networks.


### Models & mathematical formulation
**1) LSTM (Long Short-Term Memory)**

LSTM cell equations:

$$f_t = σ(W_f · [h_{t-1}, x_t] + b_f)$$
$$i_t = σ(W_i · [h_{t-1}, x_t] + b_i)$$
$$Ĉ_t = tanh(W_C · [h_{t-1}, x_t] + b_C)$$
$$C_t = f_t * C_{t-1} + i_t * Ĉ_t$$
$$o_t = σ(W_o · [h_{t-1}, x_t] + b_o)$$
$$h_t = o_t * tanh(C_t)$$

**2) BiLSTM (Bidirectional LSTM)**

Two LSTMs process the sequence in forward and backward directions.

Output for time t is concatenation: 

$$h_t = [→h_t; ←h_t].$$

**3) CNN-LSTM**

1D convolutional layers extract local temporal patterns from input windows.

Convolutional feature maps are passed to LSTM layers to model long-term dependencies.

**4) CNN-BiLSTM**

1D convolution → BiLSTM pipeline to capture local features and bidirectional temporal context.

### Evaluation metrics

Three metrics to quantify error and fit:

**Mean Squared Error (MSE):**

$$MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

**Root Mean Squared Error (RMSE):**

$$RMSE = \sqrt{ \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 }$$

**Coefficient of Determination (R²):**

$$R^2=1-\frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2} {\sum_{i=1}^{n} (y_i - \bar{y})^2}$$

### Interpretation:

- Lower MSE / RMSE → better accuracy.

- R² close to 1 → model explains most of variance.

**Experimental design**

For each dataset length (1, 5, 10, 20 years):

Split: 80% training / 20% testing (chronological split preserving temporal order).

Models trained with identical preprocessing, normalization, and training loop to ensure fair comparison.

Early stopping based on validation loss; best model weights (lowest validation loss) are retained.

Optimizer: RMSprop (as used in the report), chosen for time-series nonstationarity.

Reported metrics computed on held-out test set.

Training details & recommended hyperparameters

(These are taken from the study setup and typical choices; adjust if you reproduce.)

Optimizer: RMSprop

Loss: Mean Squared Error (MSE)

Batch size: 32 (recommended)

Learning rate: 1e-3 (tune with scheduler or grid search)

Epochs: up to 200 (with early stopping)

EarlyStopping patience: 10 (stop if no improvement)

Sequence length / input window: 30–90 days (tune per experiment)

Dropout: 0.2–0.5 (to reduce overfitting for complex models)

Convolution layers (for CNN variants): 1–3 layers, kernel sizes 3–7

LSTM units: 64–256 (tune per model / dataset size)


Below is the key results table reproduced from the project report (test set metrics):

<div align="center">
	
Table 1 — Fit performance by dataset length and model (MSE / RMSE / R²)

|Period	|Metric	|LSTM	|BiLSTM	|CNN-LSTM	|CNN-BiLSTM|
|---|---|---|---|---|---|
|1 year	|MSE	|0.000587	|0.004290	|0.011579|	0.001742|
| |RMSE	|0.024222	|0.065499	|0.107605	|0.041742|
| |R²	|0.981730	|0.866407	|0.639438	|0.945741|
|5 year	|MSE	|0.000133	|0.000303	|0.000769	|0.000889|
| |RMSE	|0.011538	|0.017410	|0.027728	|0.029810|
| |R²	|0.990772	|0.978990	|0.946705	|0.938401|
|10 year	|MSE	|0.001617	|0.007054	|0.007034|	0.009505|
| |RMSE	|0.040210	|0.083989	|0.083871	|0.097492|
| |R²	|0.966214|	0.852595	|0.853010	|0.801388|
|20 year	|MSE	|0.004421	|0.003920	|0.006058	|0.007448|
| |RMSE	|0.066488	|0.062609	|0.077832	|0.086301|
| |R²	|0.876849	|0.890801	|0.831241	|0.792521|

</div>
	
<img width="1179" height="616" alt="圖片" src="https://github.com/user-attachments/assets/bee5ac7c-2381-40c0-9a6a-cab805312a30" />
<img width="1122" height="616" alt="圖片" src="https://github.com/user-attachments/assets/7afe39df-3be4-4b53-92d4-7af8f11a26a5" />
<img width="1107" height="610" alt="圖片" src="https://github.com/user-attachments/assets/3dfcd794-f555-42ab-b7b9-65cc7f93f90a" />



### Notes / Interpretation:

- 1-year: LSTM excels on very short datasets—likely due to lower model complexity and reduced overfitting.
<img width="1304" height="918" alt="圖片" src="https://github.com/user-attachments/assets/7c814583-f3f8-4702-8714-2d8ba75af95b" />

- 5-year: All models perform well; LSTM still marginally best (R² ~ 0.99).
<img width="1360" height="952" alt="圖片" src="https://github.com/user-attachments/assets/c177343e-7273-497b-ba0b-769be397004a" />

- 10–20 years: LSTM and BiLSTM maintain stronger stability. CNN variants often underperform, possibly due to long-term variability and potential overfitting or inability to jointly model local & long-range patterns well for this dataset.
<img width="1272" height="925" alt="圖片" src="https://github.com/user-attachments/assets/df93f3ff-aff8-410d-88ba-fa654d403a00" />
<img width="1346" height="939" alt="圖片" src="https://github.com/user-attachments/assets/c5875f44-15d2-45f1-9560-c3ec173697b3" />

### References
Ali, A. S. A., Ebrahimi, S., Ashiq, M. M., Alasta, M. S., & Azari, B. (2022). CNN-Bi LSTM neural network for simulating groundwater level. Computational Research Progress in Applied Science & Engineering, 8(1), 2748. https://doi.org/10.52547/crpase.8.1.2748

Fan, C., Li, H., & Hu, J. (2020). Short-term runoff prediction using hybrid CNN–LSTM model. Water, 12(5), 1390. https://doi.org/10.3390/w12051390

Fang, D., Zeng, Y., & Qian, J. (2024). Fault diagnosis of hydro-turbine via the incorporation of Bayesian algorithm optimized CNN-LSTM neural network. Energy, 290, 130326. https://doi.org/10.1016/j.energy.2024.130326

Hinton, G. (2012). Neural networks for machine learning: Lecture 6a—Overview of mini-batch gradient descent. University of Toronto.

Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. Neural Computation, 9(8), 1735–1780. https://doi.org/10.1162/neco.1997.9.8.1735

Kratzert, F., Klotz, D., Brenner, C., Schulz, K., & Herrnegger, M. (2018). Rainfall–runoff modelling using long short-term memory (LSTM) networks. Hydrology and Earth System Sciences, 22(11), 6005–6022. https://doi.org/10.5194/hess-22-6005-2018

Rakhshani, E., & Coulibaly, P. (2021). Groundwater level prediction using hybrid deep learning models: A case study in Ontario, Canada. Environmental Modelling & Software, 140, 105039. https://doi.org/10.1016/j.envsoft.2021.105039

Schuster, M., & Paliwal, K. K. (1997). Bidirectional recurrent neural networks. IEEE Transactions on Signal Processing, 45(11), 2673–2681. https://doi.org/10.1109/78.650093

Shen, C. (2018). A transdisciplinary review of deep learning research for earth system science. Water Resources Research, 54(11), 8558–8593. https://doi.org/10.1029/2018WR022643

Waqas, M., & Humphries, U. W. (2024). A critical review of RNN and LSTM variants in hydrological time series predictions. MethodsX, 13, 102946. https://doi.org/10.1016/j.mex.2024.102946

Yin, Z., Zeng, Z., Xu, S., & Yu, J. (2020). Comparative analysis of deep learning models in groundwater level forecasting. Journal of Hydrology, 584, 124700. https://doi.org/10.1016/j.jhydrol.2020.124700

Zhang, D., Han, X., Deng, C., & Liu, Y. (2021). Deep learning for spatiotemporal modeling: A survey. IEEE Transactions on Knowledge and Data Engineering, 34(9), 3546–3566. https://doi.org/10.1109/TKDE.2021.3054589

Zhao, K., Wulder, M. A., Hu, T., Bright, R., Wu, Q., Qin, H., Li, Y., Toman, E., Mallick, B., Zhang, X., & Brown, M. (2019). Detecting change-point, trend, and seasonality in satellite time series data to track abrupt changes and nonlinear dynamics: A Bayesian ensemble algorithm. Remote Sensing of Environment, 232, 111181. https://doi.org/10.1016/j.rse.2019.04.034
