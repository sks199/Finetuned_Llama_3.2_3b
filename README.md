# README.md for Fine-tuning Llama 3.2 Notebooks

## Overview

This repository contains two Jupyter notebooks for fine-tuning the Llama 3.2 model using the Unsloth framework. These notebooks demonstrate various techniques for training and evaluating the model, tracking performance metrics, and utilizing the model for inference tasks.

### Notebooks Included

1. **Llama_3_2_finetune_with_unsloth.ipynb**
2. **Finetune_Llama_3_2_3b_v2.ipynb**

## Requirements

To run these notebooks, ensure you have the following installed:

- Python 3.x
- Jupyter Notebook or Google Colab
- Required libraries (install via pip):
  ```bash
  pip install torch transformers unsloth wandb
  ```

## Notebook Descriptions

### 1. Llama_3_2_finetune_with_unsloth.ipynb

This notebook focuses on fine-tuning the Llama 3.2 model using the Unsloth library. It includes the following sections:

- **Setup**: Import necessary libraries and set up the environment.
- **Data Preparation**: Load and preprocess the dataset for training.
- **Model Initialization**: Initialize the Llama 3.2 model and tokenizer.
- **Training Loop**: Implement a training loop that logs training loss at each step. The training loss is recorded as follows:

| Step | Training Loss |
|------|---------------|
| 1    | 2.181300      |
| 2    | 2.581400      |
| ...  | ...           |
| 200  | 1.853300      |

- **Inference**: Use the trained model to generate responses based on user queries.
- **Model Saving**: Save the fine-tuned model and tokenizer for future use.

### 2. Finetune_Llama_3_2_3b_v2.ipynb

This notebook builds upon the first by introducing validation loss tracking and additional configurations for fine-tuning:

- **Setup**: Similar to the first notebook, with necessary imports.
- **Data Loading**: Load datasets and prepare them for training and validation.
- **Training Configuration**: Set parameters such as batch size, learning rate, and number of epochs.
- **Training Loop**: Track both training and validation losses throughout training:

| Step | Training Loss | Validation Loss |
|------|---------------|-----------------|
| 270  | 2.180700      | 2.901599        |
| 540  | 2.018000      | 3.080374        |
| ...  | ...           | ...             |
| 1350 | 1.408700      | 3.222132        |

- **Error Handling**: Includes warnings related to deprecated tokenizer settings.
- **Model Saving**: Save both the fine-tuned model and its tokenizer.

## Usage Instructions

To run these notebooks:

1. Open each notebook in Jupyter or Google Colab.
2. Follow the instructions in each cell to execute code blocks sequentially.
3. Monitor output logs for training progress and loss metrics.
4. After completion, utilize saved models for inference tasks as demonstrated in the notebooks.

## Contributing

Contributions are welcome! If you have suggestions or improvements, please create a pull request or open an issue.

