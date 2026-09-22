# Sentiment Analysis of Short and Incomplete Texts using Transformers and Attention Mechanism

This repository implements sentiment analysis for short and incomplete texts using a Transformer-based backbone (RoBERTa) and a custom attention mechanism. It targets social media content (e.g., tweets) that often lacks context, combining denoising and classification to improve robustness. Experiments validate the approach on benchmark datasets; see the Paper Reference below.

## Paper Reference

- Nouralizadeh Ganji, R., & Tohidi, N. (2025). Sentiment Analysis of Short and Incomplete Text using Transformers and Attention Mechanism.  
  - Computer Engineering Faculty, Artificial Intelligence Department, K. N. Toosi University of Technology, Tehran, Iran  
  - Department of Computer Science, Wayne State University, USA

**BibTeX:**
```bibtex
@article{,
  title={Sentiment Analysis of Short and Incomplete Text using Transformers and Attention Mechanism},
  author={Reza Nouralizadeh Ganji and Nasim Tohidi},
  year={2025},
  journal={In Submission},
}
```

## Features

- Transformer-based backbone: RoBERTa for robust context modeling.
- Custom attention layer: Tailored for short/incomplete texts to strengthen signal extraction.
- Dual training loop: Denoising and classification phases for improved performance.
- Flexible tokenizers: Includes specialized MTP tokenizers.
- Reproducible experiments: Scripts for data loading, training, evaluation, and visualization.
- Integrated evaluation and plots: Confusion matrix, metrics, and pie charts.
- Notebook workflow: Interactive experimentation via Jupyter.

## Installation

### Prerequisites

- Python >= 3.8
- PyTorch >= 1.13
- Transformers >= 4.0.0
- scikit-learn
- wandb (optional, for experiment tracking)
- pandas, numpy, matplotlib

Install dependencies (example using pip):
```bash
pip install torch transformers scikit-learn pandas numpy matplotlib wandb
```

## Usage Guide

### Typical Workflow

```bash
# 1. Prepare data files (e.g., Sentiment140 TSV) and place them in the datasets/ folder.

# 2. Train the model
python main.py --mode train --epochs 5 --batch_size 8 --model_name cardiffnlp/twitter-roberta-base-sentiment

# 3. Evaluate / Infer
python main.py --mode eval --load_model_path path/to/saved/model.bin

# 4. Plot results
python mtp_plots.py --input path/to/results.csv
```

### Jupyter Notebook

Use `Sentiment-Analysis-Of-Short-and-Incomplete-Texts-V20.ipynb` for interactive runs, analysis, and visualization.

## Methodology

- Data Preprocessing: Cleans short/noisy text and applies custom tokenization (see `mtp_tokenizer.py`, `Tokenizer.py`).
- Model:
  - Transformer encoder (RoBERTa backbone).
  - Custom attention head (`mtp_model.py`), optimized for incomplete content.
  - Dual loss functions: Dice loss for denoising and CrossEntropy for classification.
- Training Scripts:
  - `main.py` orchestrates multi-phase training (denoising + classification) and evaluation.
  - Metrics and plots via `mtp_plots.py`.
- Tokenizer Loader:
  - `mtp_tok_load.py` and `mtp_tokenizer.py` manage vocabulary, config, and byte-pair encodings.

## Repository Structure

```plaintext
.
├── main.py                              # Entry point for training/evaluation
├── mtp_model.py                         # Model definition (custom attention/transformers)
├── mtp_train.py                         # Training loop and utilities
├── mtp_plots.py                         # Visualization scripts
├── mtp_tokenizer.py                     # Custom tokenizer logic
├── mtp_tok_load.py                      # Tokenizer loader
├── Tokenizer.py                         # Alternative tokenizer definition
├── rf_data.py                           # Data management and preprocessing
├── loader.py                            # Utility for dataset loading
├── dice_loss_rs.py                      # Dice loss function implementation
├── datasets/                            # Contains Sentiment140 dataset and others
├── outputs/                             # Model checkpoints and results
└── Sentiment-Analysis-Of-Short-and-Incomplete-Texts-V20.ipynb  # Experiment notebook
```

## License

This project is distributed under the MIT License.  
See the `LICENSE` file for details.

## Citation

If you use this work, please cite the paper (BibTeX above).

## Contact

Questions, suggestions, or collaboration offers?  
- Reza Nouralizadeh Ganji: [Rezang52@gmail.com](mailto:Rezang52@gmail.com)

---

Visual aids available: See the experiment notebook (`*.ipynb`) and output plots in the repository.
