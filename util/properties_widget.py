from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit
from PySide6.QtCore import Qt

class PropertiesWidget(QWidget):
    def __init__(self, orientation=Qt.Vertical):
        super().__init__()
        self.orientation = orientation
        self.layout = QVBoxLayout() if orientation == Qt.Vertical else QHBoxLayout()
        self.setLayout(self.layout)
        self.current_object = None

    def set_object(self, obj):
        self.current_object = obj
        self.clear_layout()
        
        if hasattr(obj, 'get_properties'):
            properties = obj.get_properties()
            for prop_name, prop_value in properties.items():
                read_only = obj.is_property_readonly(prop_name) if hasattr(obj, 'is_property_readonly') else False
                
                if self.orientation == Qt.Vertical:
                    self.layout.addWidget(QLabel(prop_name.capitalize()))
                    widget = self.create_widget(prop_name, prop_value)
                    self.layout.addWidget(widget)
                else:
                    hlayout = QHBoxLayout()
                    hlayout.addWidget(QLabel(prop_name.capitalize()))
                    widget = self.create_widget(prop_name, prop_value)
                    hlayout.addWidget(widget)
                    self.layout.addLayout(hlayout)
                
                if read_only:
                    widget.setReadOnly(True)
                    widget.setStyleSheet("QLineEdit[readOnly=\"true\"] {color: #808080; background-color: #F0F0F0;}")
            
            self.layout.addStretch()

    def create_widget(self, prop_name, prop_value):
        if isinstance(prop_value, str):
            widget = QLineEdit(prop_value)
            widget.textChanged.connect(lambda text, name=prop_name: self.update_property(name, text))
        elif isinstance(prop_value, int):
            widget = QLineEdit(str(prop_value))
            widget.textChanged.connect(lambda text, name=prop_name: self.update_property(name, int(text) if text.isdigit() else 0))
        elif isinstance(prop_value, float):
            widget = QLineEdit(str(prop_value))
            widget.textChanged.connect(lambda text, name=prop_name: self.update_property(name, float(text) if text.replace('.', '').isdigit() else 0.0))
        else:
            widget = QLineEdit(str(prop_value))
            widget.textChanged.connect(lambda text, name=prop_name: self.update_property(name, text))
        return widget

    def update_property(self, name: str, value):
        if self.current_object and hasattr(self.current_object, 'set_property'):
            self.current_object.set_property(name, value)

    def clear_layout(self):
        while self.layout.count():
            item = self.layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                self.clear_layout_recursive(item.layout())

    def clear_layout_recursive(self, layout):
        while layout.count():
            item = layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                self.clear_layout_recursive(item.layout())
