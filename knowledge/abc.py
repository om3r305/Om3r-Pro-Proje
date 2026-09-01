# -*- coding: utf-8 -*-
from __future__ import annotations
# Auto-created module: Proje1.knowledge.abc
# created_at: 1757975314
# id: 75d780549c

from __future__ import annotations
from abc import ABC, abstractmethod

class BaseStrategy(ABC):
    NAME = 'BaseStrategy'
    @abstractmethod
    def update(self, price: float) -> None: ...
    @abstractmethod
    def signal(self) -> dict: ...
