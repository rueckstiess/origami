# STORM Implementation Plan

STORM - **S**emi-s**T**ructured **O**bject **R**epresentation **M**odel (formerly known as "DocFormer").

## Goals

- Increased efficiency compared to our Axon research code base
- clean minimal re-implementation for productionization / tech transfer
- better integration into open source ecosystem, e.g. use with inference libraries such as vLLM for fast inference (KV cache, grouped query attention, ...)
- Better basis for open-sourcing our code
- more modularity for ablation studies, e.g. replace key/value tokenizer with classical BPE tokenizer

## 3rd Party Libraries

- Huggingface Transformers: https://huggingface.co/docs/transformers/en/index
- Outlines: https://outlines-dev.github.io/outlines/welcome/

## Roadmap

Incrementally go from Vanilla → STORM, always end-to-end training possible

#### Phase 1 – Vanilla Transformer + Structured Decoding for Classification

- Load JSON dataset (from file/MongoDB)
- Use regular tokenizer (BPE)
- Train model on JSON documents
- For classification, use [Outlines](https://outlines-dev.github.io/outlines/welcome/) to force one of the labels

Compare training convergence, accuracy with Axon DocFormers

#### Phase 2 – Key/Value Tokenizer

Implement a custom tokenizer that implements our Key/Value tokenization scheme plus grammar tokens from documents.

Repeat experiments, compare to Phase 1 and Axon DocFormer

#### Phase 3 – Document Position Encoding

Implement a custom position encoding based on field paths for each token.

#### Phase 4a – JSON Guard Rails for training + inference

Use Outlines to force structured decoding to only allow valid JSON

#### Phase 4b - Schema Guard Rails for training + inference

Use Outlines to force schema-based structured decoding to only allow valid keys/values/grammar tokens based on PDA states

#### Phase 5 - Shuffled factorisations, random chains
