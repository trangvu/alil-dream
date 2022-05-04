# Dream-ALIL
Source code of paper ["Learning How to Active Learn by Dreaming"](https://aclanthology.org/P19-1401) - ACL2019 

Heuristic-based active learning methods are limited when the data distribution of the underlying learning problems vary as they are not flexible to exploit characteristics inherent to a given problem. On the other hand, data-driven active learning learn the AL acquisition function from the data of a source task via simulation and then applied to the target task. However, they are often restricted to learn from closely related domains. This repo implements a method to adapt the learned active learning acquisition function to the target domain to bridge the domain mismatch between them.

## Dependencies

* [TensorFlow 1.2+ for Python 3](https://www.tensorflow.org/get_started/os_setup.html)


## Experiments
### Active learning algorithm
This repo includes implementations of the following active learning algorithms:
- Random sampling
- Uncertainty sampling (Entropy-based)
- Diversity sampling based on Jaccard coefficient
- PAL[1]: a reinforcement learning based method
- ALIL[2]: an imitation learning based method
- ALIL-dream: our proposed method

### Training and evaluation scripts
* Training scripts and configuration for all experiments in the paper can be found under `./ner/experiments` folder (NER tasks) and `./tc/experiments` (task classification)

## Citing
Please cite the following papers if you found the resources in this repository useful.

#### Learning How to Active Learn by Dreaming
```
@inproceedings{vu-etal-2019-learning,
    title = "Learning How to Active Learn by Dreaming",
    author = "Vu, Thuy-Trang  and
      Liu, Ming  and
      Phung, Dinh  and
      Haffari, Gholamreza",
    booktitle = "Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/P19-1401",
    doi = "10.18653/v1/P19-1401",
    pages = "4091--4101"
}

```

#### Learning How to Actively Learn: A Deep Imitation Learning Approach
```
@inproceedings{liu-etal-2018-learning-actively,
    title = "Learning How to Actively Learn: A Deep Imitation Learning Approach",
    author = "Liu, Ming  and
      Buntine, Wray  and
      Haffari, Gholamreza",
    booktitle = "Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2018",
    address = "Melbourne, Australia",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/P18-1174",
    doi = "10.18653/v1/P18-1174",
    pages = "1874--1883"
}

```
## References
[1] Meng Fang, Yuan Li, and Trevor Cohn. 2017. Learning how to active learn: A deep reinforcement learning approach - EMNLP'17
[2] Ming Liu, Wray Buntine, and Gholamreza Haffari. 2018. Learning how to actively learn: A deep imitation learning approach - ACL'18
