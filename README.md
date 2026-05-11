# Barlow Twins SASRec


Non-contrastive learning (NCL) methods have recently advanced many areas of machine learning—often outperforming their contrastive counterparts.  Among these, Barlow Twins excels at producing embeddings whose dimensions capture distinct, non-redundant features while remaining robust to perturbations.  Despite these benefits, NCL remains almost entirely unexplored within recommender systems.  In this paper, we bridge that gap by adapting Barlow Twins to sequential recommendation and examining its effects on model behavior.  Specifically, we build on the Transformer-based next-item predictor, augmenting its standard cross-entropy objective with a Barlow Twins redundancy-reduction term.  Our resulting Barlow Twins for Sequential Recommendation (BT-SR) model not only outperforms strong baselines in accuracy, but also mitigates popularity bias and produces more confident, well-calibrated recommendations.

## Requirements Installation

We recommend to use python 3.10 to replicate our environment.
To install all the necessary packages, simply run

```bash
pip install -r requirements.txt
```

## Data

For all datasets except Amazon Beauty (to ensure comparable performance for *Table 4* from the paper), we excluded unpopular items with fewer than 5 interactions and removed users with fewer than 20 interaction records. An example of the preprocessing can be found in `notebooks/Example_preprocessing.ipynb`. Preprocessed datasets can also be downloaded directly: [Kindle Store](https://disk.yandex.ru/d/Nlg1Lw3zYanosA), [Yelp](https://disk.yandex.ru/d/qdJZPjGt14H01w), [Gowalla](https://disk.yandex.ru/d/UnlGkcKD14uPNQ), [Amazon Beauty](https://disk.yandex.ru/d/3IriR7a-Ahvd3w). MovieLens-1M is donwloading authomaticly.


## Experiments Reproduction

When running the code for the experiments, you can pass +project_name={PNAME} +task_name{TNAME} options, in which case the intermediate validation metrics and the final test metrics will be reported to a ClearML server and could be later viewed in a web interface, otherwise only the final test metrics will be printed to the terminal.

To reproduce the best results from the paper (in terms of NDCG@10) for each model ($SCE$, $BCE$, $gBCE$, $CE^-$, $CE$), you should run the following command
```bash
python train.py --config-path={CONFIG_PATH} --config-name={CONFIG_NAME} data_path={DATA_PATH}
```
For example, to reproduce the best results of the $BT-SR$ model on the Yelp dataset with temporal train/test splitting, you should run
```bash
python train.py --config-path=configs/temporal/yelp --config-name='ce_bt' data_path=data/yelp.csv
```

Below are commands to reproduce our best results of $BT-SR$ model on 5 datasets included in the paper:
```bash
python train.py --config-path=configs/temporal/ml1m --config-name='ce_bt' data_path='None'
```

```bash
python train.py --config-path=configs/temporal/yelp --config-name='ce_bt' data_path=data/yelp.csv
```

```bash
python train.py --config-path=configs/temporal/beauty --config-name='sce_bt' data_path=data/beauty.csv
```

```bash
python train.py --config-path=configs/temporal/kindle_store --config-name='sce_bt' data_path=data/kindle.csv
```

```bash
python train.py --config-path=configs/temporal/gowalla --config-name='ce_bt' data_path=data/gowalla.csv
```

## Ablation Study
Figure 1 presents the complete results of our ablation study on the parameters α and λ from Equations 
\[
C_{ij} = \frac{1}{B} \sum_{b=1}^B 
\frac{Z^A_{b,i} \; Z^B_{b,j}}
{\sqrt{\sum_{b'=1}^B (Z^A_{b',i})^2} \; \sqrt{\sum_{b'=1}^B (Z^B_{b',j})^2}},
\]
and
\[
\mathcal{L}_{BT} = \sum_{i=1}^D (1 - C_{ii})^2
\;+\; \lambda \sum_{i=1}^D \sum_{\substack{j=1 \\ j \neq i}}^D C_{ij}^2.
\]
Specifically, we set α = 0 to entirely remove the Barlow Twins term from the training loss and λ = 0 to eliminate the decorrelation term.

For sensitivity analysis, we first fix α to its optimal value and vary λ to assess its impact on invariance. Similarly, we fix λ and adjust α to analyze sensitivity with respect to decorrelation. The results demonstrate that both invariance and decorrelation play crucial roles in learning generalizable user representations for sequential recommendation (SR).

We observe dataset-dependent optimal configurations for α and λ. More interestingly, tuning these parameters allows control over recommender behavior, such as the trade-off between the quality of long-tail and short-head recommendations. For instance, on Gowalla, the maximum HR@1 is achieved with α = 0.5, while HR@10 peaks at α = 0.1. Additionally, increasing α significantly improves Cov@K on YELP and Gowalla. The popularity bucket metrics further (Figure 3) reveal that α can be adjusted to balance recommendation quality across different popularity segments.
