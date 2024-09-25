import sys
import llama_cpp
import llm_generator

from typing import List, Tuple
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                               QTextEdit, QLineEdit, QPushButton, QFileDialog,  
                               QLabel, QSplitter)
from PySide6.QtGui import QColor, QPen, QBrush, QTextCursor
from PySide6.QtCore import QCoreApplication, Qt, QRectF, Slot

from CustomNodeWidget import Node, CustomLayout, ScrollArea

GO = "go"
STOP = "stop"
NODE_SPACING = 50.0

"""
TODO:
[ ] Add temp, top_n, top_k gui
[ ] Hook up drop down selection to create a new response stream
[ ] Add the full resonse above the nodes in the node row.. just for easy readability hopefully
[ ] create a metric from logits and or softmax to quantify hallicination/certainty
[ ] Try generating N responses, then ask LLM to pick the best
    [ ] can the LLM identify hallicinated reponses? 
"""

class LLMExplorer(QMainWindow):
    llm_button:QPushButton = None

    def __init__(self):
        super().__init__()

        self.response_generator = llm_generator.ResponseGeneratorThread()
        self.response_generator.new_data_signal.connect(self.update_data)
        self.response_generator.max_response_length = 12

        self.setWindowTitle("LLM Explorer")
        self.setGeometry(100, 100, 1200, 800)

        main_widget = QWidget()
        main_layout = QHBoxLayout()
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # Create a splitter for resizable panels
        splitter = QSplitter(Qt.Vertical)

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

        # Current Response Node Panel
        self.node_scroll_area = ScrollArea()
        self.current_node_row = 0

        node_view_widget = QWidget()
        node_view_layout = QVBoxLayout(node_view_widget)
        node_view_label = QLabel("Response Nodes")
        node_view_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        node_view_layout.addWidget(node_view_label)
        node_view_layout.addWidget(self.node_scroll_area)
        splitter.addWidget(node_view_widget)

        # Prompt Panel
        prompt_widget = QWidget()
        prompt_layout = QVBoxLayout(prompt_widget)
        prompt_label = QLabel("Prompt")
        prompt_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.prompt_input = QLineEdit()
        self.prompt_input.setPlaceholderText("Enter your prompt here")
        self.llm_button = QPushButton(GO)
        self.llm_button.clicked.connect(self.on_llm_button_pressed)
        prompt_input_layout = QHBoxLayout()
        prompt_input_layout.addWidget(prompt_label)
        prompt_input_layout.addWidget(self.prompt_input)
        prompt_input_layout.addWidget(self.llm_button)

        
        # Model Path Input
        model_label = QLabel("Model Selection")
        model_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.model_path_input = QLineEdit()
        self.model_path_input.setPlaceholderText("Model path")
        self.model_path_button = QPushButton("Select Model")
        self.model_path_button.clicked.connect(self.select_model)
        model_layout = QHBoxLayout()
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_path_input)
        model_layout.addWidget(self.model_path_button)

        prompt_layout.addLayout(prompt_input_layout)
        prompt_layout.addSpacing(20)
        prompt_layout.addLayout(model_layout)
        prompt_layout.addStretch(1)
        splitter.addWidget(prompt_widget)

        # Set stretch factors for the splitter
        splitter.setStretchFactor(0, 1)  # Chat session
        splitter.setStretchFactor(1, 2)  # Response nodes
        splitter.setStretchFactor(2, 1)  # Prompt panel

        main_layout.addWidget(splitter)
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

    @Slot()
    def start_generating(self):
        self.response_generator.start()

    @Slot()
    def stop_generating(self):
        self.response_generator.stop()
        self.response_generator.wait()

    @Slot(llm_generator.SampleData, str)
    def update_data(self, sample_data: llm_generator.SampleData, decoded_token: str):
        self.chat_history.insertPlainText(decoded_token)

        node = Node(decoded_token)
        self.node_scroll_area.add_node(node, self.current_node_row)
        node.alternatives_combo.addItems(sample_data.alt_decoded_tokens)

        #force an update
        QCoreApplication.processEvents()


    def select_model(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Select LLM Model")
        if file_name:
            self.model_path_input.setText(file_name)
            result = self.response_generator.load_model(file_name)
            self.chat_history.append(f"{result}\n")

    def on_llm_button_pressed(self):
        if not self.response_generator.llm:
            self.chat_history.append("Please load a model first.")
            return
        if self.llm_button.text() == GO:
            prompt = self.prompt_input.text()
            self.chat_history.append(f"<font color='green'>User:</font> {prompt}")
            self.prompt_input.clear()

            # Generate response
            self.response_generator.prompt = prompt
            self.response_generator.start()

            self.llm_button.setText(STOP)
            QCoreApplication.processEvents()
        elif self.llm_button.text() == STOP:
            self.response_generator.stop()
            
            self.llm_button.setText(GO)
            QCoreApplication.processEvents()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    explorer = LLMExplorer()
    explorer.show()
    sys.exit(app.exec())