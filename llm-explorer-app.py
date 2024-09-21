import sys
from typing import List, Tuple
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QTextEdit, QLineEdit, QPushButton, QFileDialog, QGraphicsView, 
                             QGraphicsScene, QLabel, QSplitter)
from PyQt6.QtGui import QColor, QPen, QBrush
from PyQt6.QtCore import Qt, QRectF
import llama_cpp


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

    def get_token(self)->float:
        return self.top_n[self.token_index][0]

    def get_attention_score(self)->float:
        return self.top_n[self.token_index][1]

class LLMExplorer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LLM Explorer")
        self.setGeometry(100, 100, 1200, 800)

        self.llm = None
        self.response_graph = None

        main_widget = QWidget()
        main_layout = QHBoxLayout()
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # Create a splitter for resizable panels
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Chat Session Panel
        chat_widget = QWidget()
        chat_layout = QVBoxLayout(chat_widget)
        chat_label = QLabel("Chat Session")
        chat_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.chat_history = QTextEdit()
        self.chat_history.setReadOnly(True)
        chat_layout.addWidget(chat_label)
        chat_layout.addWidget(self.chat_history)
        splitter.addWidget(chat_widget)

        # Current Response Graph Panel
        graph_widget = QWidget()
        graph_layout = QVBoxLayout(graph_widget)
        graph_label = QLabel("Response Graph")
        graph_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.graph_view = QGraphicsView()
        self.graph_scene = QGraphicsScene()
        self.graph_view.setScene(self.graph_scene)
        graph_layout.addWidget(graph_label)
        graph_layout.addWidget(self.graph_view)
        splitter.addWidget(graph_widget)

        # Prompt Panel
        prompt_widget = QWidget()
        prompt_layout = QVBoxLayout(prompt_widget)
        prompt_label = QLabel("Prompt")
        prompt_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.prompt_input = QLineEdit()
        self.prompt_input.setPlaceholderText("Enter your prompt here")
        self.prompt_input.returnPressed.connect(self.on_prompt_entered)
        
        # Model Path Input
        model_label = QLabel("Model Selection")
        model_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.model_path_input = QLineEdit()
        self.model_path_input.setPlaceholderText("Model path")
        self.model_path_button = QPushButton("Select Model")
        self.model_path_button.clicked.connect(self.select_model)
        model_layout = QHBoxLayout()
        model_layout.addWidget(self.model_path_input)
        model_layout.addWidget(self.model_path_button)

        prompt_layout.addWidget(prompt_label)
        prompt_layout.addWidget(self.prompt_input)
        prompt_layout.addSpacing(20)
        prompt_layout.addWidget(model_label)
        prompt_layout.addLayout(model_layout)
        prompt_layout.addStretch(1)
        splitter.addWidget(prompt_widget)

        # Set stretch factors for the splitter
        splitter.setStretchFactor(0, 1)  # Chat session
        splitter.setStretchFactor(1, 2)  # Response graph
        splitter.setStretchFactor(2, 1)  # Prompt panel

        main_layout.addWidget(splitter)
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

    def select_model(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Select LLM Model")
        if file_name:
            self.model_path_input.setText(file_name)
            self.load_model(file_name)

    def load_model(self, model_path):
        try:
            self.llm = llama_cpp.Llama(model_path=model_path, n_gpu_layers=-1)
            self.chat_history.append("Model loaded successfully.")
        except Exception as e:
            self.chat_history.append(f"Error loading model: {str(e)}")

    def on_prompt_entered(self):
        if not self.llm:
            self.chat_history.append("Please load a model first.")
            return

        prompt = self.prompt_input.text()
        self.chat_history.append(f"<font color='green'>User:</font> {prompt}")
        self.prompt_input.clear()

        # Generate response
        response = self.generate_response(prompt)
        self.chat_history.append(f"<font color='red'>LLM:</font> {response}")

        # Update graph
        self.update_response_graph(response)

    def generate_response(self, prompt):
        # This is a simplified version. You'll need to implement the token-by-token
        # generation and probability calculation as described in your example.
        prompt_tokens = self.llm.tokenize(prompt.encode())
        sample_idx = self.llm.n_tokens + len(prompt_tokens) - 1
        #self.llm.eval(prompt_tokens)

        # start with a dummy node
        self.response_graph = TokenNode(0,[])
        response_text = ""

        # Eval and sample
        loop = True
        while loop:
            self.llm.eval(prompt_tokens)
            while sample_idx < self.llm.n_tokens:
                token = self.llm.sample(idx=sample_idx) #todo add temp, top_k, top_p

                loop = not llama_cpp.llama_token_is_eog(self.llm._model.model, token)

                #tokens_or_none = yield token
                tokens_or_none = token
                prompt_tokens.clear()
                prompt_tokens.append(token)

                #
                detokenized = self.llm.detokenize([tokens_or_none])
                response_text += detokenized.decode("utf-8")

                attention_scores = self.llm.scores[sample_idx]
                top_n_scores = top_x_values_with_indices(attention_scores, 10)
                
                token_node = TokenNode(0, top_n_scores)
                self.response_graph.next = token_node                    

                #
                sample_idx += 1

                if sample_idx < self.llm.n_tokens and token != self.llm._input_ids[sample_idx]:
                    self.llm.n_tokens = sample_idx
                    self._ctx.kv_cache_seq_rm(-1, self.llm.n_tokens, -1)
                    break

                

        # remove dummy node
        self.response_graph = self.response_graph.next 
 
        return response_text
    
        """
        attention_score = result[0]
        response_token = result[1]
        response = self.llm.detokenize([response_token])
        return response.decode("utf-8"), attention_score         
        """    


    def update_response_graph(self, response):
        # Clear previous graph
        self.graph_scene.clear()

        # Create and layout nodes for each token in the response
        # This is a placeholder implementation. You'll need to create the actual
        # graph structure with probabilities and alternatives.
        tokens = response.split()
        for i, token in enumerate(tokens):
            node = self.graph_scene.addRect(QRectF(i * 100, 0, 80, 40))
            text = self.graph_scene.addText(token)
            text.setPos(i * 100, 0)



if __name__ == "__main__":
    app = QApplication(sys.argv)
    explorer = LLMExplorer()
    explorer.show()
    sys.exit(app.exec())
