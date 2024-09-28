import importlib
import json
from typing import Any, Dict, Type, TypeVar, List
import datetime

T = TypeVar('T', bound='ISerializable')

class SerializationError(Exception):
    """Custom exception for serialization errors"""
    pass

class ISerializable:
    _class_registry = {}
    _exclude_from_json: List[str] = []
    _exclude_from_properties: List[str] = []
    _readonly_properties: List[str] = []

    @classmethod
    def register_class(cls):
        cls._class_registry[f"{cls.__module__}.{cls.__name__}"] = cls

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.register_class()

    def get_properties(self) -> Dict[str, Any]:
        return {
            key: value for key, value in self.__dict__.items()
            if not key.startswith('_') and key not in self._exclude_from_properties
        }
    
    def is_property_readonly(self, property_name) -> bool:
        return property_name in self._readonly_properties
    
    def set_property(self, name: str, value: Any):  
        if name not in self._exclude_from_properties:  
            setattr(self, name, value)
    
    @classmethod
    def post_deserialize(self):
        pass

    def to_dict(self) -> Dict[str, Any]:
        data = {
            key: self._serialize_value(value) for key, value in self.__dict__.items()
            if not key.startswith('_') and key not in self._exclude_from_json
        }
        data['__type__'] = f"{self.__class__.__module__}.{self.__class__.__name__}"
        return data

    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        if '__type__' in data:
            class_type = cls._get_class(data['__type__'])
            if not issubclass(class_type, cls):
                raise TypeError(f"Class {data['__type__']} is not a subclass of {cls.__name__}")
            instance = class_type()
        else:
            instance = cls()
        
        for name, value in data.items():
            if name != '__type__' and not name.startswith('_') and name not in cls._exclude_from_json:
                setattr(instance, name, cls._deserialize_value(value))

        instance.post_deserialize()

        return instance

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=self._json_default)

    @classmethod
    def from_json_file(cls: Type[T], filepath: str) -> T:
        try:
            with open(filepath, 'r') as f:
                json_data = json.load(f)
                return cls.from_dict(json_data)
        except Exception as e:
            print(f"from_json_file failed to read {filepath}: {e}")
        return None

    @classmethod
    def from_json(cls: Type[T], json_str: str) -> T:
        data = json.loads(json_str)
        return cls.from_dict(data)

    @classmethod
    def _get_class(cls, full_class_name: str) -> Type:
        if full_class_name in cls._class_registry:
            return cls._class_registry[full_class_name]
        
        module_name, class_name = full_class_name.rsplit('.', 1)
        try:
            module = importlib.import_module(module_name)
            class_type = getattr(module, class_name)
            cls._class_registry[full_class_name] = class_type  # Cache the class for future use
            return class_type
        except (ImportError, AttributeError):
            raise TypeError(f"Class {full_class_name} not found")

    @staticmethod
    def _serialize_value(value: Any) -> Any:
        if isinstance(value, ISerializable):
            return value.to_dict()
        elif isinstance(value, list):
            return [ISerializable._serialize_value(item) for item in value]
        elif isinstance(value, datetime.datetime):
            return value.isoformat()
        return value

    @staticmethod
    def _deserialize_value(value: Any) -> Any:
        if isinstance(value, dict) and '__type__' in value:
            class_type = ISerializable._get_class(value['__type__'])
            if issubclass(class_type, ISerializable):
                return class_type.from_dict(value)
        elif isinstance(value, list):
            return [ISerializable._deserialize_value(item) for item in value]
        elif isinstance(value, str):
            try:
                return datetime.datetime.fromisoformat(value)
            except ValueError:
                return value
        return value

    @staticmethod
    def _json_default(obj: Any) -> Any:
        if isinstance(obj, ISerializable):
            return obj.to_dict()
        elif isinstance(obj, datetime.datetime):
            return obj.isoformat()
        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")
