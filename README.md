# AIAE-AbliterationBench
The entry for [Team Blue 2, Electric Boogaloo]  at the January 2025 AI-Plans AI Alignment Evals Hackathon. It uses Abliteration (Arditi et al) as a base for benchmarking model resilience to residual stream attacks.

## Building a benchmark for robustness to Abliteration

Abliteration ([Arditi _et al._](https://arxiv.org/abs/2406.11717)) is a residual stream attack that substantially impedes a model's ability to refuse a user's requests (e.g. if asking how to build a weapon).
We hypothesize that a well-aligned model which has internalized the value of keeping people safe will naturally refuse to cooperate without having to dedicate a conceptual category (instantiated as a direction in the vector space of the residual stream) to "refusal". Therefore, under this assumption, vulnerability to this attack would be an indication of a model "performatively" refusing to cooperate, understanding it is supposed to refuse, but without internalizing "keeping people safe" as an objective.

We aim to build a benchmark that evaluates the effectiveness of Abliteration at jailbreaking models (lower would be better). We also hope to investigate our hypothesis. If time allows, we have a number of directions in which we may extend the scope of the project.

## Example usage:

```bash
python ablit_bench.py -n Qwen/Qwen-1.5-0.5B-Chat google/gemma-1.1-2b-it -l 2 -i 50 -b 10
```

## Context

Arditi _et al._ found that refusal in LLMs (_e.g._ “As an AI language model, I can't assist with [...]”) is primarily mediated by a single "refusal direction" in the residual stream (_i.e._ the outputs of the multi-head self-attention layers and feed-forward layers inside Transformer Decoder blocks). We can directly modify these values at runtime or modify the weights so as to dramatically increase or decrease chances of refusal. Modifying an LLM in this way to uncensor it is called _abliteration_ (portmanteau of ablation + obliteration).

## Acknowledgements

First and foremost, we would like to thank Kabir Kumar and the AI-Plans team for organizing this hackathon. It was a fantastic learning experience, and we all had a very good time.

We also extend our thanks to Arditi _et al_ for their discovery; as well as Maxime Labonne for his detailed [blog post](https://mlabonne.github.io/blog/posts/2024-06-04_Uncensor_any_LLM_with_abliteration.html); the dev team for [abliterator](https://github.com/FailSpy/abliterator); and Tsadoq, developer of [ErisForge](https://github.com/Tsadoq/ErisForge), which we ended up making substantial use of and modifying to suit our use case.
