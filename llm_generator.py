import llama_cpp
from llama_cpp._internals import _LlamaTokenDataArray

from typing import List, Tuple
from PySide6.QtCore import QThread, Signal, QMutex, QMutexLocker
from NodeGraphQt import NodeGraph, BaseNode 
from NodeGraphQt.widgets.node_widgets import NodeLabel

class SampleData:
    def __init__(self, decoded_token: str, token_index: int, token_data_array:_LlamaTokenDataArray, decoded_tokens: List[str]):
        self.decoded_token = decoded_token
        self.token_index:int = token_index
        self.token_data_array:_LlamaTokenDataArray = token_data_array
        self.decoded_tokens:List[str] = decoded_tokens

    def get_logit(self) -> float:
        return self.token_data_array.candidates_data[self.token_index].logit
    
    # could be softmax, but not necessarily
    def get_p(self) -> float:
        return self.token_data_array.candidates_data[self.token_index].p    

class TokenNode(BaseNode):
    # unique node identifier domain.
    __identifier__ = 'LLMExplorer'
    # initial default node name.
    NODE_NAME = 'TokenNode'    

    def __init__(self, data: SampleData ):
        super(TokenNode, self).__init__()
        self.data:SampleData = data
        self.prev:TokenNode = None
        self.next:TokenNode = None
        self.alt:TokenNode = None

        # create input ports.
        self.add_input('prev', color=(180, 80, 0))
        self.add_input('parent', color=(180, 80, 0))

        # create output ports.
        self.add_output('next')
        self.add_output('alt')

        self.add_custom_widget(NodeLabel(self.view, "token:",text=self.data.decoded_token))
        self.add_custom_widget(NodeLabel(self.view, "logit:",text=str(self.data.get_logit())))
        self.add_custom_widget(NodeLabel(self.view, "p:",text=str(self.data.get_p())))
        
        # create the QComboBox menu.
        items = self.data.decoded_tokens
        self.add_combo_menu('candidates', 'Menu Test', items=items)

class ResponseGeneratorThread(QThread):
    new_data_signal = Signal(SampleData, str)

    llm = None
    response_data = []
    prompt = ""

    def __init__(self):
        super().__init__()
        self._is_running = False
        self.llm = None
        self.response_data = []
        self.response_text = ""
        self.max_response_length = 0
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

    # thread entry point
    def run(self):    
        if self.llm and len(self.prompt) > 0:
            self._is_running = True
            self.response_text = ""
            response_length = 0
            loop = True
            prompt_tokens = self.llm.tokenize(self.prompt.encode())
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

                        selected_token_index = -1
                        decoded_tokens: List[str] = []
                        for i in range(self.llm.token_data_array.candidates.size):  
                            candidate_data = self.llm.token_data_array.candidates_data
                            candidate_token = candidate_data.id[i]
                            detokenized = self.llm.detokenize([candidate_token])
                            decoded = detokenized.decode("utf-8")                            
                            decoded_tokens.append(decoded)
                            
                            if token == candidate_token:
                                selected_token_index = i

                        #print(f"*** decoded_tokens={decoded_tokens}")
                        sample_data = SampleData(response_token_decoded, selected_token_index, self.llm.token_data_array, decoded_tokens)
                        self.response_data.append(sample_data)
                        
                        self.new_data_signal.emit(sample_data, response_token_decoded)

                    #
                    sample_idx += 1
                    response_length += 1

                    if self.max_response_length > 0 and response_length > self.max_response_length:
                        loop = False
                        break

                    if sample_idx < self.llm.n_tokens and token != self.llm._input_ids[sample_idx]:
                        self.llm.n_tokens = sample_idx
                        self._ctx.kv_cache_seq_rm(-1, self.llm.n_tokens, -1)
                        break


    def stop(self):
        self._is_running = False

    def get_response_data(self):#
        # Ensure thread-safe access to data
        with QMutexLocker(self.mutex):
            return self.response_data.copy()
        
    def get_response_text(self):
        # Ensure thread-safe access to data
        with QMutexLocker(self.mutex):
            return self.response_text