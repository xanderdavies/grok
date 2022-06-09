# Grokking: Generalization Beyond Overfitting in Small Algorithmic Datasets

Basic replication of grokking as observed in Power et al.'s ["Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets"](https://arxiv.org/abs/2201.02177). Code adapted from [OpenAI's implementation](https://github.com/openai/grok); behavior replicated for the modular division task (mod 97) with 50/50 train/test split and AdamW optimizer.

## Usage

Calling `python runner.py` replicates the results in the figures. Use `python runner.py -h` to see optional arguments.

## Results

Accuracy | Loss
:-------------------:|:-------------------------:
![accuracy](https://user-images.githubusercontent.com/55059966/172950363-0cde68df-c192-4267-8ca6-ea58273c3c5f.png) | ![loss](https://user-images.githubusercontent.com/55059966/172950382-dd7590b1-f180-4d57-a24b-c93410259a30.png)

## Citation

```BibTex
@article{power2022grokking,
  title={Grokking: Generalization beyond overfitting on small algorithmic datasets},
  author={Power, Alethea and Burda, Yuri and Edwards, Harri and Babuschkin, Igor and Misra, Vedant},
  journal={arXiv preprint arXiv:2201.02177},
  year={2022}
}
```
