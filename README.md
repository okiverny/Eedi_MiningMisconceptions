# Eedi - Mining Misconceptions in Mathematics

## 1. Kaggle Notebooks

- [Effiency Prize (CPU only)](https://www.kaggle.com/code/olehkivernyk/eedi-hybrid-cpu?scriptVersionId=212557482) - Run the misconception finding using a combination of relatively small embedding models.
- [Qwen 2.5 14B Retrieval Model + Qwen 72B Pointwise Reranking](https://www.kaggle.com/code/olehkivernyk/14b-retr-72b-logits?scriptVersionId=212064676) - Not selected notebook for the final submission but the private Score is 0.473 which would result into Bronze Medal!
- [7B Retrieval Model + Qwen 72B Pointwise Reranking](https://www.kaggle.com/code/olehkivernyk/14b-retr-72b-logits?scriptVersionId=212566841) - my best model on the Public Leaderboard scoring 0.500

## ToDo List

 **Efficiency code:**
- from comments "Merging multiple models with SLERP/TIES increases the scores significantly while keeping exact same runtime."

**GPU code:**
- Try listwise rerankes with logit scores before pointwise reranker.