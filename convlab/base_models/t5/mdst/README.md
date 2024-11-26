## Official Code for EMNLP2023 Paper "Building Multi-domain Dialog State Trackers from Single-domain Dialogs"

paper link: https://aclanthology.org/2023.emnlp-main.946/

- `get_sgd_groups.ipynb`: get the combination of SGD for experiments.
- `split_data.py`: split the data into single domain/multi-domain dialogs. Split single dialogs to train:valid:test == 8:1:1 randomly. Split multi-domain dialogs to train:valid:test == 4:1:5 randomly. Also simplify state by removing empty slot-value pairs. Run 
    ```
    python split_data.py -d multiwoz21 sgd
    ```
    to generate 
    - `((train|validation|test)_|^)(single|multi)_domain.json`, `full_state.json` (all non-empty slots in the data), and `multi_domain_slot_pairs.json` (cross-domain slot pairs from true multi-domain dialog) under `data/${dataset_name}(group${group_idx}/)`
    - data statistics `data_stat.md` & `original_data_stat.md` under `data/${dataset_name}`.
- `run_qadst_(mwoz|sgd).sh`: Train QADST model on single domain dialogs. Input: `slot question + dialog`, output: `value`. Infer the value for potential related slots in different domains. Generate 
    - Trained model under corresponding `output_dir`.
    - `${output_dir}/(test|eval_cross_domain)_generated_preditions.json`: single domain test set, and single domain training set but slots from all domains.
    - Under qadst_dir: `qa_slot_pairs.json` (cross-domain slots filtered by QADST f1) and `(train|validation)_single_domain.json` (add cross-domain predictions to `(train|validation)_single_domain.json`) from `evaluate_qa.py`.
- `run_coqr_canard.sh`: Train CoQR models on CANARD dataset (context, question, rewrite) with 3 different settings:
    - `origin`: (context, question) -> rewrite. Aim to remove anaphora and ellipsis.
    - `reverse_SDI`: (context, rewrite) -> question. Label the substitution, deletion, and insertion span of gold rewrite to ease the learning and control the model behavior.
- `../../bert/train_bio.sh`: Train a BIO tagger for extract value from a uttearance on `Taskmaster` datasets. This tagger is used in `evaluate_qr.py` as a filter of generated utterances.
- `run_coqr_syn.sh`: Use trained CoQR model for data augmentation. `aug_type` must be 4 (`DataProcessor.AUG_TYPE_CONCAT2REL`).
- `run_dst.sh`: Train DST model using different data.
- `create_data.py`: create data for different tasks, models, and data augmentation methods. (see `run_data_aug.sh`)
    - create data for single domain QADST (used in `run_qadst_(mwoz|sgd).sh`):
    - create data for conversational query rewrite (used in `run_coqr_canard.sh`):
    - create data for data augmentation (used in `run_dst.sh`, see `data_processor.py` for `data_aug_type`, generate sub-directories under `data/${dataset_name}/(group${group_idx}/)`.):
- `data_processor.py`: process data for training and inference of different types of models and data_augmentation methods.
- `inference.py`: support iteractive inference (turn by turn), generate predictions for test dialogs.
- `evaluate.py`: evaluate the result generated by `inference.py` under `output_dir`.
- `evaluate_qa.py`: merge qa-dst model cross domain prediction with original single domain dials. `output/**/qa_single_domain/test_generated_predictions.json` + `train_single_domain.json` => `train_single_domain_qa.json` + `qa_slot_pairs.csv`.
- `evaluate_qr.py`: generate the coqr data and filter.
- `utils.py`: utility functions.

## Citing

If you use ConvLab-3 in your research, please cite:

```
@inproceedings{zhu-etal-2023-building,
    title = "Building Multi-domain Dialog State Trackers from Single-domain Dialogs",
    author = "Zhu, Qi  and
      Zhang, Zheng  and
      Zhu, Xiaoyan  and
      Huang, Minlie",
    editor = "Bouamor, Houda  and
      Pino, Juan  and
      Bali, Kalika",
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.emnlp-main.946",
    doi = "10.18653/v1/2023.emnlp-main.946",
    pages = "15323--15335"
}
```