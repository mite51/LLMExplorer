import llama_cpp
from llama_cpp._internals import _LlamaTokenDataArray

from typing import List, Tuple
from PySide6.QtCore import QThread, Signal, QMutex, QMutexLocker
from util.serializable import ISerializable

class SampleData:
    # decoded_token decoded token from sample
    # the index selected from the cadidate tokens for this response.
    # the full response up until this sample
    # token_data_array sample info a cadidates
    def __init__(self, decoded_token: str, token_index: int, token_data_array:_LlamaTokenDataArray):
        self.decoded_token = decoded_token
        self.token_index:int = token_index
        self.token_data_array:_LlamaTokenDataArray = token_data_array

    def get_logit(self) -> float:
        return self.token_data_array.candidates_data[self.token_index].logit
    
    # could be softmax, but not necessarily
    def get_p(self) -> float:
        return self.token_data_array.candidates_data[self.token_index].p  

    def get_candidate_count(self) -> int:
        return self.token_data_array.candidates.size

    def get_canidate_logit(self, i:int) -> float:
        return self.token_data_array.candidates_data[i].logit

    def get_canidate_p(self, i:int) -> float:
        return self.token_data_array.candidates_data[i].p

    def get_canidate_decodedtoken(self, i:int, llm) -> float:
        candidate_token = self.token_data_array.candidates_data.id[i]
        detokenized = llm.detokenize([candidate_token])
        decoded = detokenized.decode("utf-8")    
        return decoded

class SampleSettings(ISerializable):
    def __init__(self):  
        self.max_samples:int = 40
        self.top_k: int = 40
        self.top_p: float = 0.95
        self.min_p: float = 0.05
        self.typical_p: float = 1.0
        self.temp: float = 0.80
        self.repeat_penalty: float = 1.0
        self.frequency_penalty: float = 0.0
        self.presence_penalty: float = 0.0
        self.tfs_z: float = 1.0
        self.mirostat_mode: int = 0
        self.mirostat_eta: float = 0.1
        self.mirostat_tau: float = 5.0
        self.penalize_nl: bool = True

class ResponseGeneratorThread(QThread):
    new_data_signal = Signal(SampleData, str)
    end_of_response = Signal()

    def __init__(self):
        super().__init__()
        self._is_running = False
        self.llm = None
        self.settings:SampleSettings = SampleSettings()
        self.prompt =""
        self.response_data = []
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
                    token = self.llm.sample(idx=sample_idx,
                                            top_k = self.settings.top_k,
                                            top_p = self.settings.top_p,
                                            min_p = self.settings.min_p,
                                            typical_p = self.settings.typical_p,
                                            temp = self.settings.temp,
                                            repeat_penalty = self.settings.repeat_penalty,
                                            frequency_penalty = self.settings.frequency_penalty,
                                            presence_penalty = self.settings.presence_penalty,
                                            tfs_z = self.settings.tfs_z,
                                            mirostat_mode = self.settings.mirostat_mode,
                                            mirostat_eta = self.settings.mirostat_eta,
                                            mirostat_tau = self.settings.mirostat_tau,
                                            penalize_nl = self.settings.penalize_nl,                                            
                                            )

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

                        # find the index of the selected token
                        selected_token_index = -1
                        for i in range(self.llm.token_data_array.candidates.size):  
                            candidate_data = self.llm.token_data_array.candidates_data
                            candidate_token = candidate_data.id[i]
                            if token == candidate_token:
                                selected_token_index = i

                        sample_data = SampleData(response_token_decoded, selected_token_index, self.llm.token_data_array)
                        self.response_data.append(sample_data)
                        
                        self.new_data_signal.emit(sample_data, response_token_decoded)

                    #
                    sample_idx += 1
                    response_length += 1

                    if self.settings.max_samples > 0 and response_length > self.settings.max_samples:
                        loop = False
                        break

                    if sample_idx < self.llm.n_tokens and token != self.llm._input_ids[sample_idx]:
                        self.llm.n_tokens = sample_idx
                        self._ctx.kv_cache_seq_rm(-1, self.llm.n_tokens, -1)
                        break

            self.end_of_response.emit()

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