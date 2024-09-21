import llama_cpp

from typing import List, Tuple
from PyQt6.QtCore import QThread, pyqtSignal, QMutex, QMutexLocker

def top_x_values_with_indices(values, top_X) -> List[Tuple[str, float]]:
    indexed_values = list(enumerate(values))
    indexed_values.sort(key=lambda x: x[1], reverse=True)
    top_x = indexed_values[:top_X]
    top_x_sorted = sorted(top_x, key=lambda x: x[1], reverse=True)
    return [(token, attention_score) for token, attention_score in top_x_sorted]

class TokenNode:
    def __init__(self, token_index: int, top_n_results:List[Tuple[str, float]]  ):
        self.token_index:int = token_index
        self.top_n: List[Tuple[str, float]] = top_n_results
        self.next:TokenNode = None
        self.alt:TokenNode = None
        self.widget = None

    def get_token(self)->float:
        return self.top_n[self.token_index][0]

    def get_attention_score(self)->float:
        return self.top_n[self.token_index][1]

class ResponseGeneratorThread(QThread):
    new_data_signal = pyqtSignal(TokenNode, str)

    llm = None
    response_graph = None

    def __init__(self):
        super().__init__()
        self._is_running = False
        self.llm = None
        self.response_graph = TokenNode(0,[])
        self.response_text = ""
        self.mutex = QMutex()  # Create a mutex for thread synchronization

    def load_model(self, model_path) -> str:
        result = ""
        if not self._is_running:
            try:
                self.llm = llama_cpp.Llama(model_path=model_path, n_gpu_layers=-1)
                result = "Model loaded successfully."
            except Exception as e:
                result = f"Error loading model: {str(e)}"
        else:
            result = "load aborted, model in use"                
        return result        

    def start(self, prompt):
        if self.llm:
            self._is_running = True
            self.response_text = ""
            loop = True
            prompt_tokens = self.llm.tokenize(prompt.encode())
            sample_idx = self.llm.n_tokens + len(prompt_tokens) - 1            
            while self._is_running and loop:
                self.llm.eval(prompt_tokens)
                while sample_idx < self.llm.n_tokens:
                    token = self.llm.sample(idx=sample_idx) #todo add temp, top_k, top_p

                    loop = not llama_cpp.llama_token_is_eog(self.llm._model.model, token)

                    #tokens_or_none = yield token
                    tokens_or_none = token
                    prompt_tokens.clear()
                    prompt_tokens.append(token)

                    # Ensure thread-safe access to data
                    with QMutexLocker(self.mutex):
                        detokenized = self.llm.detokenize([tokens_or_none])
                        response_token_decoded = detokenized.decode("utf-8")
                        self.response_text += response_token_decoded

                        attention_scores = self.llm.scores[sample_idx]
                        top_n_scores = top_x_values_with_indices(attention_scores, 10)
                        
                        token_node = TokenNode(0, top_n_scores)
                        self.response_graph.next = token_node
                        
                        self.new_data_signal.emit(token_node, response_token_decoded)

                    #
                    sample_idx += 1

                    if sample_idx < self.llm.n_tokens and token != self.llm._input_ids[sample_idx]:
                        self.llm.n_tokens = sample_idx
                        self._ctx.kv_cache_seq_rm(-1, self.llm.n_tokens, -1)
                        break


    def stop(self):
        self._is_running = False

    def get_response_graph(self):
        # Ensure thread-safe access to data
        with QMutexLocker(self.mutex):
            return self.response_graph.copy()
        
    def get_response_text(self):
        # Ensure thread-safe access to data
        with QMutexLocker(self.mutex):
            return self.response_text