# streamRec-forgetting
Extention of Stream-based recommender system library (https://github.com/joaoms/streamRec) with code to create checkpoints of factorization models and evaluate forgetting.

Folders:
* data: utilities to process and instanciate data
* dataset_evaluation_utils: utilities to assess datasets (user frequency over intervals, frequency of interactions, etc.)
* eval_implicit: utilities to evaluate using prequential evaluation and holdout evaluation.
* notebooks: contains dissertation experiments and new experiments notebooks as well as their outputs
* plot_utils: contains utilities to generate graphs
* recommenders_implicit: contains implementations of incremental recommender system algorithms. Algorithms generate models to provide top-k items recommendation from users' implicit feedback.
