#!/bin/bash

# for usage information, run
#     python ablit_bench -h
# There are currently some model compatibility restrictions:
#   - Models must have a `model` attribute when loaded with `transformers.AutoModelForCausalLM`. This works for most recent models but excludes older models like GPT2
#   - Chat templates should be supported by the models' tokenizers. The code shouldn't crash, but you may get unexpected results.
#   - [Qwen 1.5 0.5B Chat] and [gemma 1.1 2b it] have been verified as working correctly.

python ablit_bench.py \
  -n Qwen/Qwen-1.5-0.5B-Chat meta-llama/Llama-2-7b-chat-hf \
  -l 2 \
  -i 50 \
  -b 10


# google/gemma-1.1-2b-it
