# Generating plan with LLM (GPT-3.5)
This README provides step-by-step instructions for generating step-by-step plans with GPT-3.5 as part of our implementation in ReALFRED.


## Embed language instructions
Run `bert_embedding-trains.py` to embed language instructions with BERT encoder.

```
python bert_embedding-trains.py --dn dn
```
This will create `dn/train_all_emb.p` which contains language instruction embeddings.

## Retrieve top-k in-context examples
Run `bert_embedding-retriever.py` to retreive top-k (here. k=9) in-context examples for each tasks in valid and tests splits.

```
python bert_embedding-retriever.py --dn dn
```
This will create `dn/ReALFRED-{split}_retrieved_keys.json` which contains retrieved in-context examples for each tasks.

## Generate plan with GPT-3.5
**Modify your openai API key** in `generate_plans.py`.
Then run `generate_plans.py` to generate plans with GPT-3.5
```
python generate_plans.py --dn dn
```

Finally, run `postprocess.py` to postprocess llm generated plans to ALFRED executable action sequences.
```
python postprocess.py --dn dn
```

This will create `.json` files in `planner_results/dn`.
You can use them by editting `read_test_dict` in `$LLMPLANNER/models/instructions_processed_LP/ALFRED_task_helper.py` to use your files.
## Hardware 
Trained and Tested on:
- **GPU** - RTX A6000
- **CPU** - Intel(R) Core(TM) i7-12700K CPU @ 3.60GHz
- **RAM** - 64GB
- **OS** - Ubuntu 20.04