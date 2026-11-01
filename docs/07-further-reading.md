# Further reading

This is a set of books, papers, benchmarks and tools about time series forecasting,
probabilistic evaluation, anomaly detection and time series foundation models (TSFMs).
The entries are grouped by topic. Each entry has a short note on what it covers and how
it relates to this project.

Papers are given by title and authors, so you can find them on arXiv or in a search
engine.

- [Forecasting fundamentals](#forecasting-fundamentals)
- [Probabilistic forecasts and their evaluation](#probabilistic-forecasts-and-their-evaluation)
- [Calibrated intervals: conformal prediction](#calibrated-intervals-conformal-prediction)
- [Deep learning for time series](#deep-learning-for-time-series)
- [Time series foundation models](#time-series-foundation-models)
- [Benchmarks and competitions](#benchmarks-and-competitions)
- [Anomaly detection in time series](#anomaly-detection-in-time-series)
- [Libraries and tools](#libraries-and-tools)
- [Journals, conferences and blogs](#journals-conferences-and-blogs)

## Forecasting fundamentals

| Reference | What it covers |
|---|---|
| *Forecasting: Principles and Practice* (3rd edition), Rob J. Hyndman and George Athanasopoulos. Free online at [otexts.com/fpp3](https://otexts.com/fpp3/). A Python edition, written with Nixtla, is also available. | The standard introduction: time series patterns, decomposition, exponential smoothing (ETS), ARIMA, prediction intervals, forecasts with several seasonal cycles, and time series cross-validation. The rolling-origin evaluation in this project is the cross-validation method of this book. |
| *Time Series Analysis: Forecasting and Control*, George Box, Gwilym Jenkins, Gregory Reinsel and Greta Ljung | The classical reference for ARIMA models and the statistical theory of time series. |
| *Forecasting: theory and practice*, Fotios Petropoulos and many co-authors (International Journal of Forecasting, 2022) | A long review of almost all areas of forecasting, written by more than 80 specialists. It is a good map of the field. |

Classical methods are the baselines against which TSFMs are measured. A seasonal naive
forecast ("same hour last week") is the simplest; ETS, ARIMA and MSTL are stronger.

## Probabilistic forecasts and their evaluation

| Reference | What it covers |
|---|---|
| *Probabilistic forecasts, calibration and sharpness*, Tilmann Gneiting, Fadoua Balabdaoui and Adrian E. Raftery (2007) | The principle "maximize the sharpness of the forecast, subject to calibration". This is the reason why [06-comparing-models.md](06-comparing-models.md) reads the width of a range together with its coverage. |
| *Strictly proper scoring rules, prediction, and estimation*, Tilmann Gneiting and Adrian E. Raftery (2007) | Why some scores, such as the pinball loss and the CRPS, reward a model only when it gives its true beliefs. The theory behind WQL. |
| *Another look at measures of forecast accuracy*, Rob J. Hyndman and Anne B. Koehler (2006) | The problems of MAPE (for example, with values near zero) and the MASE, a scaled error that compares a forecast with a naive baseline. |
| *Comparing predictive accuracy*, Francis X. Diebold and Roberto S. Mariano (1995) | The Diebold–Mariano test: whether the difference in error between two forecasts is statistically significant, or can be chance. |
| *Quantile regression*, Roger Koenker (book, 2005) | The theory of quantiles and of the pinball (check) loss. |

## Calibrated intervals: conformal prediction

Conformal prediction makes prediction intervals with a guaranteed coverage, from the
errors of a model on past data. It is one way to give a model with only 10%-90%
quantiles, such as TimesFM 3, calibrated 1% and 99% lines.

| Reference | What it covers |
|---|---|
| *A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification*, Anastasios N. Angelopoulos and Stephen Bates | A practical introduction, with code examples. |
| *Conformal prediction interval for dynamic time-series* (EnbPI), Chen Xu and Yao Xie (2021), and *Adaptive Conformal Inference Under Distribution Shift*, Isaac Gibbs and Emmanuel Candès (2021) | Conformal methods for data that is not independent, and for data that changes in time, such as electricity demand. |
| *Algorithmic Learning in a Random World*, Vladimir Vovk, Alexander Gammerman and Glenn Shafer (book) | The original theory of conformal prediction. |

## Deep learning for time series

These models came before the foundation models, and introduced most of their ideas.

| Reference | What it covers |
|---|---|
| *DeepAR: Probabilistic forecasting with autoregressive recurrent networks*, David Salinas, Valentin Flunkert, Jan Gasthaus and Tim Januschowski | One neural network trained on many related series, with a probabilistic output. |
| *N-BEATS: Neural basis expansion analysis for interpretable time series forecasting*, Boris N. Oreshkin and co-authors | A deep model, made only of fully connected layers, that won against statistical methods on the M4 data. |
| *Temporal Fusion Transformers for interpretable multi-horizon time series forecasting*, Bryan Lim and co-authors | Attention, with past-only and known-future covariates, and quantile outputs. |
| *A Time Series is Worth 64 Words: Long-term Forecasting with Transformers* (PatchTST), Yuqi Nie and co-authors | Patching: the series is cut into short pieces that act as tokens. Most TSFMs, Timestral and TimesFM included, use patches. |
| *Neural forecasting: Introduction and literature overview*, Konstantinos Benidis and co-authors | A review of neural forecasting methods. |

## Time series foundation models

Useful questions for each model: how it changes numbers into tokens, what data it was
trained on, how it gives its uncertainty (quantiles or samples), and which covariates it
accepts.

| Reference | What it covers |
|---|---|
| *Chronos: Learning the Language of Time Series*, Abdul Fatir Ansari and co-authors (Amazon, 2024) | Values are scaled and put into bins, and a language-model architecture predicts the next bin. Also introduces training on synthetic series. |
| *Chronos-2* technical report (Amazon, 2025) | The model used in this project: covariates, multivariate forecasts with group attention, and 21 quantile levels. |
| *A decoder-only foundation model for time-series forecasting*, Abhimanyu Das, Weihao Kong, Rajat Sen and Yichen Zhou (Google, ICML 2024) | TimesFM: a decoder-only transformer on patches, trained on a large corpus that includes Google Trends and Wikipedia page views. The basis of TimesFM 3. |
| *Unified Training of Universal Time Series Forecasting Transformers*, Gerald Woo and co-authors (Salesforce, 2024) | Moirai: one model for any frequency and any number of variates. |
| *Lag-Llama: Towards Foundation Models for Probabilistic Time Series Forecasting*, Kashif Rasul and co-authors (2023) | An early open probabilistic TSFM, with lagged values as features. |
| *This Time is Different: An Observability Perspective on Time Series Foundation Models*, Datadog (2025) | Toto, a TSFM trained on observability metrics, and the BOOM benchmark. The reference for the use of TSFMs in software monitoring. |
| Timestral announcement, Mistral AI ([mistral.ai/news/timestral](https://mistral.ai/news/timestral/)) | The default model of this project. See also [02-capabilities.md](02-capabilities.md). |
| TiRex (NX-AI) and Sundial (Tsinghua University) | Two other open models: TiRex uses an xLSTM instead of a transformer; Sundial generates samples, so it can give any quantile. |

## Benchmarks and competitions

| Reference | What it covers |
|---|---|
| GIFT-Eval (Salesforce) | A benchmark for general forecasting models over many domains and frequencies. Many TSFM papers report WQL and MASE on it. |
| fev-bench (Amazon / AutoGluon) | A benchmark that includes tasks with covariates, with tools to make the results reproducible. |
| BOOM (Datadog) | A benchmark of real observability metrics, for the use case in the README. |
| The M4 and M5 competitions, Spyros Makridakis and co-authors (results in the International Journal of Forecasting) | Large open forecasting competitions. The results papers describe which methods won and why. M5 includes probabilistic forecasts. |
| Monash Time Series Forecasting Archive | A large collection of public data sets, used to train and to test many models. |

When you read a benchmark table, check whether the test data was also in the training
data of a model (data leakage), which metric is used, and whether the baselines were
tuned.

## Anomaly detection in time series

| Reference | What it covers |
|---|---|
| *Current Time Series Anomaly Detection Benchmarks are Flawed and are Creating the Illusion of Progress*, Renjie Wu and Eamonn Keogh (2021) | Many public anomaly benchmarks have trivial or wrong labels. Important context for any claim about anomaly detection results. |
| TSB-UAD and TSB-AD, John Paparrizos and co-authors | Newer benchmarks for time series anomaly detection, made to correct these problems. |
| The matrix profile, Eamonn Keogh's group (University of California, Riverside) | A fast method to find unusual and repeated patterns in a series, without a forecast. A useful contrast to the forecast-based method of this project. |
| *Anomaly Detection: A Survey*, Varun Chandola, Arindam Banerjee and Vipin Kumar (2009) | A general survey of anomaly types (point, contextual, collective) and methods. See also [04-anomaly-detection.md](04-anomaly-detection.md). |
| *Site Reliability Engineering* and *The Site Reliability Workbook*, Google (free online) | The chapters on monitoring and alerting: alert fatigue, symptoms against causes, and service level objectives. The operational context for forecast-based alerts. |

## Libraries and tools

| Library | What it does |
|---|---|
| [statsforecast](https://github.com/Nixtla/statsforecast), [neuralforecast](https://github.com/Nixtla/neuralforecast) (Nixtla) | Fast classical models (ETS, ARIMA, MSTL, seasonal naive) and neural models, with one interface. |
| [AutoGluon-TimeSeries](https://github.com/autogluon/autogluon) | Classical models, deep models and Chronos, with automatic ensembles. |
| [GluonTS](https://github.com/awslabs/gluonts) | Probabilistic deep learning models and evaluation tools, used for the evaluation in many papers. |
| [Darts](https://github.com/unit8co/darts) and [sktime](https://github.com/sktime/sktime) | General toolkits for forecasting, with backtesting helpers. |
| [STUMPY](https://github.com/TDAmeritrade/stumpy) | Matrix profile computations. |
| [chronos-forecasting](https://github.com/amazon-science/chronos-forecasting), [timesfm](https://github.com/google-research/timesfm), [toto](https://github.com/DataDog/toto) | The libraries of the models in [06-comparing-models.md](06-comparing-models.md). |

## Journals, conferences and blogs

- **International Journal of Forecasting**, and the **International Symposium on
  Forecasting** of the International Institute of Forecasters.
- **NeurIPS, ICML and ICLR**: most TSFM papers appear at these conferences first, or on
  arXiv.
- **Rob J. Hyndman's blog** ([robjhyndman.com/hyndsight](https://robjhyndman.com/hyndsight/)):
  forecasting practice and evaluation.
- **The research blogs of Nixtla, AutoGluon and Datadog**: comparisons of TSFMs with
  classical methods, and new model releases.
