# Grokking: Generalization Beyond Overfitting in Small Algorithmic Datasets

Basic replication of grokking as observed in Power et al.'s ["Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets"](https://arxiv.org/abs/2201.02177). Code adapted from [OpenAI's implementation](https://github.com/openai/grok); behavior replicated for the modular division task (mod 97) with 50/50 train/test split and AdamW optimizer.

## Results

Accuracy | Loss
:-------------------:|:-------------------------:
![accuracy](https://user-images.githubusercontent.com/55059966/172948851-29c155df-5ba2-4fdd-a2d3-a840bff944e4.png) | ![loss](https://user-images.githubusercontent.com/55059966/172949087-4fd7e54d-b4fd-4aa2-a164-d620d1ebf21f.png)


## Citation

```BibTex
@article{power2022grokking,
  title={Grokking: Generalization beyond overfitting on small algorithmic datasets},
  author={Power, Alethea and Burda, Yuri and Edwards, Harri and Babuschkin, Igor and Misra, Vedant},
  journal={arXiv preprint arXiv:2201.02177},
  year={2022}
}
```
