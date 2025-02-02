# Benchmarking robustness of LLMs to residual stream jailbreaks with Abliteration

[GitHub Repo](https://github.com/gpiat/AIAE-AbliterationBench/)

## Context
### Residual stream
In generative LLMs, the residual stream is a vector space which comprises the outputs of the multi-head self-attention layers and feed-forward layers inside Transformer Decoder blocks. During a forward pass, prompts are embedded in this vector space. By comparing the regions of the vector space in which different prompts are embedded, we can deduce the meaning of the directions in this space.

### Abliteration
By feeding LLMs prompts considered harmful, which the model had been trained not to answer, and prompts considered harmless, [Arditi _et al._](https://arxiv.org/abs/2406.11717) have found that refusal (e.g. “As an AI language model, I cannot help you with [...]”) is primarily mediated by a single "refusal direction" in the residual stream. We can directly modify these values at runtime to dramatically increase or decrease chances of refusal. Modifying an LLM in this way to uncensor it is called _abliteration_ (portmanteau of ablation + obliteration).

## Project Statement

The representation of "requiring refusal" as a direction in the vector space of the residual stream, making it a property of certain prompts, seems to imply that it is a learned category; as opposed to refusal being an emergent behavior of a well-aligned model. We hypothesize that a model which has internalized the value of keeping people safe will choose to refuse to cooperate without having to dedicate a conceptual category to "refusal".

Under this assumption, vulnerability to this attack would therefore be an indication of a model performatively refusing to cooperate, understanding it is supposed to refuse, but without internalizing "keeping people safe" as an objective. Furthermore, robustness to residual stream attacks is also likely a desirable feature and correlated to robustness against gradient-based adversarial-prompt-crafting attacks ([Winninger, 2024](https://aisafetyfundamentals.com/projects/exploring-the-use-of-mechanistic-interpretability-to-craft-adversarial-attacks/)).

Therefore, in this project, we aimed to build a benchmark that evaluates models’ robustness to Abliteration.


## Project Description

To evaluate a model's robustness to jailbreaking through abliteration, we measure refusal rate on a standard set of harmful prompts from the [TOXIC dataset](https://huggingface.co/datasets/Undi95/orthogonal-activation-steering-TOXIC/tree/main) before and after abliteration and return the model's Refusal Drop Rate (RDR), or the decrease in refusal rate in percentage points. We also provide a visualisation of the refusal rate before and after abliteration as seen in Fig 1.

![](https://cdn.discordapp.com/attachments/1331634298553635050/1335039823915782235/Screenshot_2025-02-01_at_00.11.27.png?ex=679f60f9&is=679e0f79&hm=30495859866b83ebdea354eaeab2ed113e7f78f7f9ae989b0e720c4e23898ced& "Fig. 1: Refusal drop rate output and refusal bar chart pre- and post-abliteration for Qwen 1.5 0.5B Chat.")

For the purposes of performing abliteration, we use a different subset of the TOXIC dataset, as well as a portion of the [Alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca) dataset for harmless prompts.


To benchmark a model, one simply runs our script on the command line and adds the HuggingFace name of the model to benchmark as an argument after the -n or --model_name option as so:
`python ablit_bench.py -n Qwen/Qwen-1.5-0.5B-Chat`

Mulitple models can be passed as arguments:
`python ablit_bench.py -n Qwen/Qwen-1.5-0.5B-Chat meta-llama/Llama-2-7b-chat-hf`

Further options are detailed in the help page of our script:
`python ablit_bench.py --help`

## Challenges

The original code for the paper on abliteration depended on the `transformer_lens library`. As this library must implement support for models on an individual basis and since most of the conversational models supported were too large to run on the cloud computing services we had rented, we decided to experiment with and ultimately switch to a different implementation called `erisforge`.

While `erisforge` did have a number of advantages including broader model support, automated identification of common model architectures and better support for exporting and running abliterated models, the functions it defines tend to have a higher level of abstraction than our use case required. Furthermore, we ran into severe memory management issues. Both problems were adressed by integrating the codebase for erisforge directly into ours, stripping it and modifying it to suit our needs. Additionally, to further improve on memory management, we implemented a batching mechanism in our data processing pipeline.

Another compatibility limitation we encountered was the need for chat template support in the model tokenizer. We implemented our own generic chat template to all back on for models which do not have a standard HuggingFace implementation of chat templates.

Finally, lack of standardization in HuggingFace Transformer models has further limited the range of models compatible with our benchmark. In particular, the libraries we have relied on and we have assumed models have a `model` attribute, which in turn has a standard set of attributes. This is fairly common but not universal.

As a consequence of our hardware limitations, compatibility limitations and model access limitations, the only models which we have confirmed fully work with our benchmark are Qwen 1.5 0.5B Chat and Gemma 1.1 2B (IT).

## Future Work
We intend to keep improving on this project past the end of this event. In particular, we hope to improve model compatibility by relying on the standard `model.state_dict()` method or `model.config` attribute.

Additionally, we plan to contribute back optimizations and other improvements to the ErisForge library.

Lastly, we plan to investigate the assumption that vulnerability to residual stream attacks is an indication of  further. While we believe it to be a plausible hypothesis, it remains untested and we have yet to come up with an experimental protocol or find a discussion of the topic in the literature.
 
## Conclusion
We have produced an automated test which measures the resilience of generative LLM assistants versus abliteration, a residual stream attack that substantially increases a model's cooperativeness on prompts it has been trained to reject. We believe this may have an application in differentiating aligned models, which have internalized the objective of keeping people safe, from performatively safe models, which understand taboo topics as a concept and behave in testing. Furthermore, we hope it can serve as a base for testing new residual stream attack vectors in the future.



## References
Abliteration Paper (Arditi _et al._ 2024):
https://arxiv.org/abs/2406.11717

Blog from the paper authors:
https://www.lesswrong.com/posts/jGuXSZgv6qfdhMCuJ/refusal-in-llms-is-mediated-by-a-single-direction

Code from the paper author: 
https://github.com/FailSpy/abliterator

Blog from Maxime Labonne (HuggingFace)
https://mlabonne.github.io/blog/posts/2024-06-04_Uncensor_any_LLM_with_abliteration.html

ErisForge library:
https://github.com/Tsadoq/ErisForge

Refusal in LLMs is an Affine function:
https://arxiv.org/pdf/2411.09003

Exploring the use of Mechanistic Interpretability to Craft Adversarial Attacks (Winninger, 2024):
https://aisafetyfundamentals.com/projects/exploring-the-use-of-mechanistic-interpretability-to-craft-adversarial-attacks/