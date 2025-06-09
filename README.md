# Barlow Twins SASRec


Non-contrastive learning (NCL) methods have recently advanced many areas of machine learning—often outperforming their contrastive counterparts.  Among these, Barlow Twins excels at producing embeddings whose dimensions capture distinct, non-redundant features while remaining robust to perturbations.  Despite these benefits, NCL remains almost entirely unexplored within recommender systems.  In this paper, we bridge that gap by adapting Barlow Twins to sequential recommendation and examining its effects on model behavior.  Specifically, we build on the Transformer-based next-item predictor, augmenting its standard cross-entropy objective with a Barlow Twins redundancy-reduction term.  Our resulting Barlow Twins for Sequential Recommendation (BT-SR) model not only outperforms strong baselines in accuracy, but also mitigates popularity bias and produces more confident, well-calibrated recommendations.

## Requirements Installation

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
For example, to reproduce the best results of the $CE$ model on the Yelp dataset with temporal train/test splitting, you should run
```bash
python train.py --config-path=configs/temporal/yelp --config-name='ce_bt' data_path=data/yelp.csv
```
For the $BT$ model, there are both configs for the best NDCG@10 performance (sce_max_ndcg.yaml) and for the same performance as the second-best model but with reduced memory consumption (ce_bt.yaml) or (sce_bt.yaml).

To reproduce the result for non-optimal configurations (other points on the corresponding figure) and to reproduce more accurate results for optimal configurations (using several random seeds), you should perform the grid search on relevant hyperparameters for each model and modify the configs accordingly. The grid used is shown below:
```json
{
    "ce": 
        {"trainer_params.seed": [1235, 37, 2451, 12, 3425],
         "dataloader.batch_size": [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]},
    "bce": 
        {"trainer_params.seed": [1235, 37, 2451, 12, 3425],
         "dataloader.batch_size": [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096],
         "dataloader.n_neg_samples": [1, 4, 16, 64, 256, 1024, 4096]},
    "dross(CE^-)": 
        {"trainer_params.seed": [1235, 37, 2451, 12, 3425],
         "dataloader.batch_size": [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096],
         "dataloader.n_neg_samples": [1, 4, 16, 64, 256, 1024, 4096]},
    "gbce": 
        {"trainer_params.seed": [1235, 37, 2451, 12, 3425],
         "dataloader.batch_size": [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096],
         "dataloader.n_neg_samples": [1, 4, 16, 64, 256, 1024, 4096],
         "model_params.gbce_t": [0.75, 0.9]},
    "sce": 
        {"trainer_params.seed": [1235, 37, 2451, 12, 3425],
         "dataloader.batch_size": [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096],
         "model_params.n_buckets": "int((dataloader.batch_size * interactions_per_user) ** 0.5 * 2.)",
         "model_params.bucket_size_x": "int((dataloader.batch_size * interactions_per_user) ** 0.5 * 2.)",
         "model_params.bucket_size_y": [64, 256, 512, 1024, 4096]},
}
``` 


