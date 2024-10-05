A very simple reponse evaluator to show token probabilies at each sample and allow selecting alternative tokens to generate altenate reponses
Create to explore the token sample data, the selection process, and how to visualize it.


Initially I thought I might be able to help spot when a model is hallucinating, I need to do more research, but when the frequency and count of candidate tokens with many options increases, this could indicate uncertainty or difficulty in generating coherent text.

Other experiments I would like to try:
-compare fine tuned models against their base models to see how the cadidate token distribution changes.
-try different sampling techniques (top-k, top-p) and visualize the differences
   *including macro sampling techniques, generate a number of chunks of tokens, then select from those chunks based on some criteria like coherence or relevance to the prompt
-explore how temperature affects the candidate token distribution and response generation