import sys
import llama_cpp
import llm_generator

from typing import List, Tuple
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QTextEdit, QLineEdit, QPushButton, QFileDialog, QGraphicsView, 
                             QGraphicsScene, QLabel, QSplitter)
from PyQt6.QtGui import QColor, QPen, QBrush, QTextCursor
from PyQt6.QtCore import QCoreApplication, Qt, QRectF, pyqtSlot


class LLMExplorer(QMainWindow):
    def __init__(self):
        super().__init__()

        self.response_generator = llm_generator.ResponseGeneratorThread()
        self.response_generator.new_data_signal.connect(self.update_data)
        self.response_graph = None

        self.setWindowTitle("LLM Explorer")
        self.setGeometry(100, 100, 1200, 800)

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

    @pyqtSlot()
    def start_generating(self):
       self.response_generator.start()
        #self.start_button.setEnabled(False)
        #self.stop_button.setEnabled(True)

    @pyqtSlot()
    def stop_generating(self):
        self.response_generator.stop()
        self.response_generator.wait()
        #self.start_button.setEnabled(True)
        #self.stop_button.setEnabled(False)

    @pyqtSlot(llm_generator.TokenNode, str)
    def update_data(self, new_node:llm_generator.TokenNode, decoded_token:str):
        self.chat_history.insertPlainText(decoded_token)
        #force an update
        QCoreApplication.processEvents()

    def select_model(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Select LLM Model")
        if file_name:
            self.model_path_input.setText(file_name)
            result = self.response_generator.load_model(file_name)
            self.chat_history.append(f"{result}\n")

    def on_prompt_entered(self):
        if not self.response_generator.llm:
            self.chat_history.append("Please load a model first.")
            return

        prompt = self.prompt_input.text()
        self.chat_history.append(f"<font color='green'>User:</font> {prompt}")
        self.prompt_input.clear()

        # Generate response
        self.response_generator.start(prompt)

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
