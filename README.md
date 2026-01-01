A very simple reponse evaluator to show token probabilies at each sample and allow selecting alternative tokens to generate altenate reponses
Create to explore the token sample data, the selection process, and how to visualize it.


Initially I thought I might be able to help spot when a model is hallucinating, I need to do more research, but when the frequency and count of candidate tokens with many options increases, this could indicate uncertainty or difficulty in generating coherent text.

Other experiments I would like to try:
-compare fine tuned models against their base models to see how the cadidate token distribution changes.
-try different sampling techniques (top-k, top-p) and visualize the differences
   *including macro sampling techniques, generate a number of chunks of tokens, then select from those chunks based on some criteria like coherence or relevance to the prompt
-explore how temperature affects the candidate token distribution and response generation


Installation notes (that worked for me):

git clone https://github.com/mite51/LLMExplorer

cd llama_cpp_python
git submodule update --init --recursive

conda create -n llm_explorer python=3.12
conda activate llm_explorer

# llama install
# I had some missing libs from ggml, this helped
export LD_LIBRARY_PATH=/home/jwylie/anaconda3/envs/rlgpu/lib
pip install -e .[all]

# for this app
sudo apt-get install -y libxcb-cursor-dev
pip3 install PySide6
