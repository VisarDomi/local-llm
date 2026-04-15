# What

This repo is my attempt at trying to find the best local model in coding ability and context length for my local setup, which is a 12gb vram nvidia card.

# Why

I am trying to find use-cases for smaller llms so that I don't depend entirely on claude opus 4.6 and gpt5.4.

# How

I first ended up on focusing exclusively on qwen3.5 after going through the latest threads of r/localllama. I try MoE and dense models from 0.8B to 122B with different quants, trying to go for high speeds and high quality and see where's the drawback for each.

After trying the models out so that they don't OOM, i then try how can they handle context, as there the prompt processing speeds start being felt. But because of cache during the usual coding agent usage of the llm, this cost is paid only once at the beginning, after that only the delta gets processed. That was all about input speed.

Output speed is the usual tokens per second the model can generate. Which is affected by context length. The rule of thumb for 260k context is that you get half speed when close to full context as opposed to when empty context. These great speeds are only possible because of the latest features that are implemented in llamacpp, like the attention rotation.
