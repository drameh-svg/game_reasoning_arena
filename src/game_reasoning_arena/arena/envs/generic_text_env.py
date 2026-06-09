"""Generic text-rendered OpenSpiel environment.

This environment is useful for OpenSpiel games that already provide readable
state and observation strings but do not need a custom board renderer yet.
"""

from typing import Any, Dict, Optional

from .open_spiel_env import OpenSpielEnv


class GenericTextEnv(OpenSpielEnv):
    """Environment wrapper that renders OpenSpiel's observation string."""

    def __init__(
        self,
        game: Any,
        game_name: str,
        player_types: Dict[str, str],
        max_game_rounds: int = None,
        seed: Optional[int] = None,
    ):
        super().__init__(game, game_name, player_types, max_game_rounds, seed)

    def get_player_symbol(self, agent_id: int) -> str:
        return f"Player {agent_id}"

    def describe_legal_actions(self, agent_id: int) -> str:
        legal = self.state.legal_actions(agent_id)
        labelled_actions = []
        for action in legal:
            try:
                label = self.state.action_to_string(agent_id, action)
            except Exception:
                label = str(action)
            labelled_actions.append(f"{action}: {label}")

        if not labelled_actions:
            return "[]"
        return "\n".join(labelled_actions)

    def render_board(self, agent_id: int) -> str:
        try:
            return self.state.observation_string(agent_id)
        except Exception:
            return str(self.state)
