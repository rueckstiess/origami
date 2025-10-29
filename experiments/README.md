# Reproducing the results from our paper

This directory contains the code and instructions to reproduce the experiments from our paper:
[ORIGAMI: A generative transformer architecture for predictions from semi-structured data](https://arxiv.org/abs/2412.17348).

There are 3 sub-directories, each with their own `README.md` file:

- [`json2vec`](./json2vec/README.md) contains the experiments from section 3.1, where we compare on standard tabular benchmarks that have been converted to JSON against various baselines and the json2vec models from [A Framework for End-to-End Learning on Semantic Tree-Structured Data](https://arxiv.org/abs/2002.05707) by William Woof and Ke Chen.
- [`ddxplus`](./ddxplus/README.md) contains the experiments from section 3.2 for a medical diagnosis task on patient information. This experiment demonstrates prediction of multi-token values representing arrays of possible pathologies.
- [`codenet`](./codenet/README.md) contains the experiments from section 3.3 related to a Java code classification task. Here we demonstrate the model's ability to deal with complex and deeply nested JSON objects.

### Experiment Tracking

**Note:** As of October 2025, the `guildai` dependency has been removed from the main ORiGAMi codebase as it is no longer actively maintained. The experiments in `json2vec`, `ddxplus`, and `codenet` directories were originally designed to use [guild.ai](https://guild.ai) for experiment management and result tracking, but these have not been migrated. Newer experiments use [yanex](https://github.com/rueckstiess/yanex) for experiment tracking.

If you wish to run the legacy experiments (`json2vec`, `ddxplus`, `codenet`), you will need to:
1. Install `guildai` separately: `pip install guildai>=0.9.0`
2. Be aware that you may encounter deprecation warnings from the guild package

### Datasets

We bundled all datasets used in the paper in a [MongoDB dump file](https://drive.google.com/uc?export=download&id=1V1Tm92tAuCu1TU_QjYPfcmtEe9NGbXLs). To reproduce the results, first
you need MongoDB installed on your system (or a remote server). Then, download the dump file, unzip it, and restore it into your MongoDB instance:

```
mongorestore dump/
```

This assumes your `mongod` server is running on `localhost` on default port 27017 and without authentication. If your setup varies, consult the [documentation](https://www.mongodb.com/docs/database-tools/mongorestore/) for `mongorestore` on how to restore the data.

If your database setup (URI, port, authentication) differs, also make sure to update the [`.env.local`](.env.local) file in each sub-directory accordingly.
