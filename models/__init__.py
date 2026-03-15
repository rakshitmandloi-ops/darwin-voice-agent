"""
Domain models for the Darwin Voice Agent system.

Design principles:
- Immutable where possible (frozen=True) — prevents accidental mutation
- Enums for closed sets — no stringly-typed agent names or outcomes
- Validators at boundaries — catch bad data at creation, not downstream
- Separate config snapshots from runtime state
"""

from models.enums import *
from models.domain import *
from models.scoring import *
from models.evolution import *
from models.cost import *
