"""Grammar constraint module for ORIGAMI.

Provides pushdown automaton (PDA) based grammar constraints
for valid JSON token generation.
"""

from .json_grammar import JSONGrammarPDA

__all__ = ["JSONGrammarPDA"]
