from typing import List
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                               QScrollArea, QPushButton, QApplication, QFrame, QComboBox)
from PySide6.QtCore import Qt, QSize, QRect, QPoint
from PySide6.QtGui import QPainter, QColor

from llm_generator import SampleData

class NoScrollComboBox(QComboBox):
    def wheelEvent(self, event):
        # Ignore the wheel event
        event.ignore()

class ArrowOnlyComboBox(QComboBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("""
            QComboBox {
                border: 0px solid #ccc;
                border-radius: 0px;
                padding-left: 0px;
                padding-right: 10px;  /* Space for arrow */
                min-width: 0;
                background: transparent;
            }
            QComboBox:editable {
                width: 0px;
                min-width: 0; 
            }
        """)
        self.view().setMinimumWidth(100)  # Set a minimum width for the dropdown

    def showPopup(self):
        # Adjust the width of the popup before showing it
        width = max(self.view().sizeHintForColumn(0) + 20, self.width())
        self.view().setMinimumWidth(width)
        super().showPopup()

    def sizeHint(self):
        return QSize(20, 20)  # Adjust size as needed for the arrow 

class OldNode(QFrame):
    expander_padding = 10
    def __init__(self, decoded_token, logit, p, sample_data: SampleData, row:int, column:int, llm, parent=None):
        super().__init__(parent)
        self.decoded_token = decoded_token
        self.logit = logit
        self.p = p
        self.is_expanded = False
        self.sample_data: SampleData = sample_data
        self.row = row
        self.column = column
        self.response_text: str = "" # cache response up to this point, helpful to have when branching alternatives
        self.setup_ui(llm)

    def setup_ui(self, llm):
        self.setFrameStyle(QFrame.Box | QFrame.Raised)#QFrame.StyledPanel | QFrame.Raised
        self.setLineWidth(1)
        self.setMidLineWidth(1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(2)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        header = QHBoxLayout()
        self.token_label = QLabel(self.decoded_token.strip())
        self.expand_button = QPushButton("▼" if self.is_expanded else "▲")
        self.expand_button.setFixedSize(20, 20)
        self.expand_button.clicked.connect(self.toggle_expand)

        header.addWidget(self.token_label)
        header.addStretch()
        header.addWidget(self.expand_button)
        header.addWidget(self.test_combo)
        layout.addLayout(header)

        self.content = QWidget()
        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(2)
        self.content_layout.addWidget(QLabel(f"Logit = {self.logit:.2f}"))
        self.content_layout.addWidget(QLabel(f"p={self.p:.2f}"))
        self.alternatives_combo = NoScrollComboBox()
        self.content_layout.addWidget(self.alternatives_combo)

        items = self.get_candidate_items(llm)
        self.alternatives_combo.addItems(items)
        self.token_label.setToolTip(str(items))

        self.content.setVisible(self.is_expanded)

        layout.addWidget(self.content)
        self.setLayout(layout)
       
    def get_candidate_items(self, llm):
        items = []
        for i in range(self.sample_data.get_candidate_count()):
            decoded_token = self.sample_data.get_canidate_decodedtoken(i, llm)
            logit = self.sample_data.get_canidate_logit(i)
            p = self.sample_data.get_canidate_p(i)
            items.append(f"{decoded_token.strip()},{p:.2f},{logit:.2f}")

        return items


    def get_desired_height(self):
        return 100 if self.is_expanded else 36

    def toggle_expand(self):
        self.is_expanded = not self.is_expanded
        self.setFixedHeight(self.get_desired_height())

        self.content.setVisible(self.is_expanded)
        self.expand_button.setText("▼" if self.is_expanded else "▲")
        self.updateGeometry()
        if self.parent():
            self.parent().update_layout()

    def sizeHint(self):
        #size = super.sizeHint()
        label_size = self.token_label.sizeHint()
        alternative_size = self.alternatives_combo.sizeHint()
        width = label_size.width() + 30
        if self.is_expanded:
            width = max(label_size.width(), alternative_size.width())
        return QSize(width + self.expander_padding, self.get_desired_height())


class Node(QFrame):
    expander_padding = 10
    def __init__(self, decoded_token, logit, p, sample_data: SampleData, row:int, column:int, llm, parent=None):
        super().__init__(parent)
        self.decoded_token = decoded_token
        self.logit = logit
        self.p = p
        self.is_expanded = False
        self.sample_data: SampleData = sample_data
        self.row = row
        self.column = column
        self.response_text: str = "" # cache response up to this point, helpful to have when branching alternatives
        self.setup_ui(llm)

    def setup_ui(self, llm):
        self.setFrameStyle(QFrame.Box | QFrame.Raised)#QFrame.StyledPanel | QFrame.Raised
        self.setLineWidth(1)
        self.setMidLineWidth(1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(1, 1, 1, 1)
        layout.setSpacing(2)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        header = QHBoxLayout()
        self.token_label = QLabel(repr(self.decoded_token))
        self.token_label.setToolTip(f"p={self.p:.2f} logit={self.logit:.2f}")

        self.alternatives_combo = ArrowOnlyComboBox()
        items = self.get_candidate_items(llm)
        self.alternatives_combo.addItems(items)

        header.addWidget(self.token_label)
        if len(items) > 1:
            header.addStretch()
            header.addWidget(self.alternatives_combo)
        layout.addLayout(header)
       
    def get_candidate_items(self, llm):
        items = []
        for i in range(self.sample_data.get_candidate_count()):
            decoded_token = self.sample_data.get_canidate_decodedtoken(i, llm)
            logit = self.sample_data.get_canidate_logit(i)
            p = self.sample_data.get_canidate_p(i)
            items.append(f"{repr(decoded_token)},{p:.2f},{logit:.2f}")

        return items


    def get_desired_height(self):
        return 100 if self.is_expanded else 36


    def sizeHint(self):
        label_size = self.token_label.sizeHint()
        width = label_size.width() + 20
        if self.alternatives_combo.count() <= 1:
            width = label_size.width()
        return QSize(width + self.expander_padding, 24)# self.get_desired_height())


class CustomLayout(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.rows : List[List[Node]] = [[]]
        self.setContentsMargins(2, 2, 2, 2)

    def insert_row_after(self, row_index:int):
        # Ensure the row_index is valid
        if row_index < 0 or row_index >= len(self.rows):
            raise ValueError("Invalid row index")
        self.rows.insert(row_index+1, [])

    def add_node(self, node):
        while len(self.rows) <= node.row:
            self.rows.append([])
        self.rows[node.row].append(node)
        node.setParent(self)
        self.update_layout()

    def clear_nodes(self):
        for row in self.rows:
            for node in row:
                node.setParent(None)

        self.rows = []        

    def update_layout(self):
        max_x = 0
        y = self.contentsMargins().top()
        for irow, row_nodes in enumerate(self.rows):
            if len(row_nodes) == 0:
                continue#?

            row_height = 0
            row_height = max(node.sizeHint().height() for node in row_nodes)
            x = self.contentsMargins().left()
            
            #
            first_node_in_row = row_nodes[0]
            first_column_index = first_node_in_row.column
            found = False
            if irow > 0 and first_column_index != 0:
                for previous_row_index in reversed(range(irow)):
                    previous_row = self.rows[previous_row_index]
                    for search_node in previous_row:
                        if search_node.column == first_node_in_row.column:
                            x = search_node.frameGeometry().left()
                            found = True
                            break
                    if found:
                        break

            for node in row_nodes:
                node_size = node.sizeHint()
                node_width = node_size.width()
                node.setGeometry(QRect(QPoint(x, y), QSize(node_width, row_height)))
                x += node_width

                max_x = max(max_x,x)
            y += row_height + 5  # Add some vertical spacing between rows

        self.setMinimumSize(max_x + self.contentsMargins().left() + self.contentsMargins().right(),
                            y + self.contentsMargins().top() + self.contentsMargins().bottom())

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(240, 240, 240))

class ScrollArea(QScrollArea):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.custom_layout = CustomLayout()
        self.setWidget(self.custom_layout)
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.NoFrame)

    def add_node(self, node):
        self.custom_layout.add_node(node)
        node.show()

if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    scroll_area = ScrollArea()

    scroll_area.add_node(Node("Token 1"), 0)
    scroll_area.add_node(Node("Token 21234"), 0)
    scroll_area.add_node(Node("Token 3"), 0)
    scroll_area.add_node(Node("Token 4", column=2), 1)

    scroll_area.resize(600, 400)
    scroll_area.show()

    sys.exit(app.exec())