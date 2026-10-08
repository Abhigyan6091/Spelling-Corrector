# Spelling Corrector

A Python-based NLP project for spelling correction and text analysis. The project combines a dictionary-driven spelling corrector with a lightweight Streamlit interface for interactive testing.

## Project overview

This repository contains:

- `q3.py`: a spelling correction implementation using edit-distance candidates, symmetric delete indexing, and context-aware scoring.
- `streamlit_app.py`: a Streamlit web app that demonstrates the NLP analyzer and spelling correction workflow in a user-friendly interface.
- `NLP_Group_Assignment_with_output.ipynb`: the notebook version of the assignment with outputs and analysis.

## Features

- Non-word error correction based on candidate generation
- Candidate ranking using unigram probability
- Context-aware bigram scoring for better word selection
- Symmetric delete index for efficient candidate lookup
- Interactive demo app built with Streamlit

## Tech stack

- Python
- NLTK
- Streamlit
- Natural language processing techniques using Brown corpus statistics

## Setup

1. Clone the repository
2. Create and activate a virtual environment (optional but recommended)
3. Install the dependencies:

```bash
pip install streamlit nltk
```

4. Download the required NLTK datasets:

```python
import nltk
nltk.download('brown')
nltk.download('treebank')
nltk.download('punkt')
nltk.download('punkt_tab')
```

## Run the application

To run the Streamlit app:

```bash
streamlit run streamlit_app.py
```

## Run the spelling corrector script

```bash
python q3.py
```

## Example use cases

- Correct misspelled words in text
- Rank possible corrections using language-model probabilities
- Demonstrate NLP text analysis through a visual interface

## Repository structure

```text
.
├── README.md
├── NLP_Group_Assignment_with_output.ipynb
├── q3.py
├── streamlit_app.py
└── report.pdf
```

## Notes

This project is intended for educational and research-oriented NLP experimentation. The spelling correction logic is designed to demonstrate core language-model and candidate-generation concepts in a simple, understandable implementation.
