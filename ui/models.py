from dataclasses import dataclass, field
from typing import Optional
import numpy as np

@dataclass
class Param:
    key: str
    label: str
    kind: str  # "int" | "float" | "choice" | "bool"
    default: object
    lo: float = 0 #valor mínimo permitido
    hi: float = 0 #valor máximo permitido
    step: float = 1
    choices: tuple = ()
    odd: bool = False

@dataclass
class Op: 
    fn: callable #função que realiza a operação
    '''
    Em kind:
    "point" -> recebe a imagem inteira (RGB ou cinza)
    "filter" -> recebe um canal 2D
    "edge" -> recebe cinza 2D
    "action" -> não gera imagem diretamente (chama fn(app, **params))
    '''
    kind: str = "point"
    params: list = field(default_factory=list)

@dataclass
class HistoryEntry:
    title: str
    image: np.ndarray 
    raw: Optional[np.ndarray] = None 