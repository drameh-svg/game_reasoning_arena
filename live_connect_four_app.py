#!/usr/bin/env python3
"""Lightweight Gradio app for live Connect Four LLM-vs-random matches.

Run from the repository root:

    python3 live_connect_four_app.py

Then open the printed local URL, choose a provider/model preset, paste the
matching API key, and click "Run selected model vs Random".
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Generator, List, Tuple


ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*args: Any, **kwargs: Any) -> bool:
        return False


DEFAULT_PRESET = "OpenAI: GPT-4o-mini"
GAME_NAME = "connect_four"
MODEL_PRESETS = {
    "OpenAI: GPT-4o-mini": {
        "model": "litellm_gpt-4o-mini",
        "env_var": "OPENAI_API_KEY",
    },
    "Groq: Llama 3.1 8B Instant": {
        "model": "litellm_groq/llama-3.1-8b-instant",
        "env_var": "GROQ_API_KEY",
    },
    "OpenRouter: Claude 3.5 Sonnet": {
        "model": "openrouter_anthropic/claude-3.5-sonnet",
        "env_var": "OPENROUTER_API_KEY",
    },
    "OpenRouter: Gemini 2.5 Flash": {
        "model": "openrouter_google/gemini-2.5-flash",
        "env_var": "OPENROUTER_API_KEY",
    },
    "Cursor API key (not a playable chat model)": {
        "model": "cursor-api",
        "env_var": "CURSOR_API_KEY",
        "unsupported_reason": (
            "Cursor API keys are for Cursor Cloud Agent/admin endpoints, "
            "not a chat-completions model endpoint. This live game needs a "
            "provider that can answer one prompt per move, such as OpenRouter "
            "or Groq."
        ),
    },
}


def _resolve_preset(model_preset: str) -> Tuple[str, str, str]:
    preset = MODEL_PRESETS.get(model_preset) or MODEL_PRESETS[DEFAULT_PRESET]
    return (
        preset["model"],
        preset["env_var"],
        preset.get("unsupported_reason", ""),
    )


def _build_config(seed: int, model_name: str) -> Dict[str, Any]:
    return {
        "env_config": {
            "game_name": GAME_NAME,
            "max_game_rounds": None,
        },
        "num_episodes": 1,
        "seed": seed,
        "use_ray": False,
        "mode": "llm_vs_random",
        "agents": {
            "player_0": {
                "type": "llm",
                "model": model_name,
            },
            "player_1": {
                "type": "random",
            },
        },
        "llm_backend": {
            "max_tokens": 250,
            "temperature": 0.1,
            "default_model": model_name,
        },
        "tensorboard_logging": False,
        "run_post_processing": False,
    }


def _agent_metadata(config: Dict[str, Any], player_id: int) -> Tuple[str, str]:
    agent_config = config["agents"].get(f"player_{player_id}", {})
    agent_type = agent_config.get("type", "unknown")
    model_name = agent_config.get("model", "None") if agent_type == "llm" else "None"
    return agent_type, model_name


def _opponent_label(config: Dict[str, Any], player_id: int) -> str:
    labels = []
    for key, agent_config in config["agents"].items():
        if key == f"player_{player_id}":
            continue
        agent_type = agent_config.get("type", "unknown")
        model_name = agent_config.get("model", "None") if agent_type == "llm" else "None"
        labels.append(f"{agent_type}_{model_name.replace('-', '_')}")
    return ", ".join(labels)


def _extract_action_and_reasoning(response: Any) -> Tuple[int, str]:
    if isinstance(response, dict):
        return response.get("action", -1), response.get("reasoning", "None")
    return response, "None"


def _render_board(env: Any) -> str:
    return f"```text\n{env.render_board(0)}\n```"


def _format_status(turn: int, rewards: Dict[int, float], done: bool) -> str:
    status = "finished" if done else "running"
    return (
        f"Status: {status}\n"
        f"Turn: {turn}\n"
        f"Rewards: Player 0 = {rewards.get(0, 0)}, "
        f"Player 1 = {rewards.get(1, 0)}"
    )


def run_live_match(
    api_key: str,
    seed: int,
    model_preset: str,
    delay_seconds: float,
    max_turns: int,
) -> Generator[Tuple[str, str, str], None, None]:
    """Run one Connect Four game and stream board/log/status updates."""
    seed = int(seed)
    max_turns = int(max_turns)
    delay_seconds = float(delay_seconds)
    model_name, api_key_env_var, unsupported_reason = _resolve_preset(
        model_preset
    )

    if unsupported_reason:
        yield (
            "No board yet.",
            f"{model_preset} cannot run this match.\n\n{unsupported_reason}",
            "Status: unsupported provider",
        )
        return

    load_dotenv()
    api_key = (api_key or "").strip()
    if api_key:
        os.environ[api_key_env_var] = api_key

    if not os.getenv(api_key_env_var):
        yield (
            "No board yet.",
            f"Missing API key for {model_preset}. Paste it in the password "
            f"box or set {api_key_env_var}.",
            "Status: not started",
        )
        return

    try:
        from game_reasoning_arena.arena.agents.policy_manager import (
            initialize_policies,
        )
        from game_reasoning_arena.arena.games.registry import registry
        from game_reasoning_arena.arena.utils.loggers import SQLiteLogger
        from game_reasoning_arena.arena.utils.seeding import set_seed
        from game_reasoning_arena.backends import initialize_llm_registry
    except ImportError as exc:
        yield (
            "No board yet.",
            "Missing dependency while starting the live app:\n"
            f"{exc}\n\n"
            "Install the UI/runtime dependencies, for example:\n"
            "python3 -m pip install open-spiel litellm gradio python-dotenv",
            "Status: dependency error",
        )
        return

    config = _build_config(seed=seed, model_name=model_name)
    transcript: List[str] = [
        "# Live Connect Four: LLM vs Random",
        "",
        f"- Preset: {model_preset}",
        f"- Player 0: LLM `{model_name}`",
        "- Player 1: repo RandomAgent",
        f"- Seed: {seed}",
        "",
    ]

    try:
        set_seed(seed)
        initialize_llm_registry()
        policies = initialize_policies(config, GAME_NAME, seed)
        player_to_agent = {
            player_id: policy
            for player_id, policy in enumerate(policies.values())
        }
        env = registry.make_env(GAME_NAME, config)
        observations, _ = env.reset(seed=seed)
    except Exception as exc:
        yield (
            "No board yet.",
            "Failed to initialize the match:\n"
            f"{type(exc).__name__}: {exc}",
            "Status: initialization error",
        )
        return

    loggers = {}
    for player_id in player_to_agent:
        agent_type, agent_model = _agent_metadata(config, player_id)
        sanitized_model = agent_model.replace("-", "_").replace("/", "_")
        loggers[player_id] = SQLiteLogger(agent_type, sanitized_model)

    rewards = {0: 0.0, 1: 0.0}
    terminated = truncated = False
    turn = 0

    transcript.append("Initial board:")
    yield _render_board(env), "\n".join(transcript), _format_status(turn, rewards, False)

    while not (terminated or truncated):
        if turn >= max_turns:
            truncated = True
            transcript.append(f"\nStopped after max_turns={max_turns}.")
            break

        current_player = env.state.current_player()
        observation = observations[current_player]
        legal_actions = observation["legal_actions"]
        agent = player_to_agent[current_player]

        transcript.append(f"\n## Turn {turn}: Player {current_player}")
        transcript.append(f"Legal actions: `{legal_actions}`")

        try:
            response = agent(observation)
            action, reasoning = _extract_action_and_reasoning(response)
        except Exception as exc:
            transcript.append(
                "Action generation failed:\n"
                f"`{type(exc).__name__}: {exc}`"
            )
            yield (
                _render_board(env),
                "\n".join(transcript),
                "Status: action generation error",
            )
            return

        agent_type, agent_model = _agent_metadata(config, current_player)
        transcript.append(f"Chosen action: `{action}`")
        if agent_type == "llm":
            transcript.append("Reasoning trace:")
            transcript.append(f"> {reasoning}")

        if action not in legal_actions:
            loggers[current_player].log_illegal_move(
                game_name=GAME_NAME,
                episode=1,
                turn=turn,
                agent_id=current_player,
                illegal_action=action,
                reason="Illegal action",
                board_state=observation["state_string"],
            )
            transcript.append(f"Illegal move detected: `{action}`")
            yield (
                _render_board(env),
                "\n".join(transcript),
                "Status: illegal move",
            )
            return

        loggers[current_player].log_move(
            game_name=GAME_NAME,
            episode=1,
            turn=turn,
            action=action,
            reasoning=reasoning,
            opponent=_opponent_label(config, current_player),
            generation_time=0.0,
            agent_type=agent_type,
            agent_model=agent_model,
            seed=seed,
            board_state=observation["state_string"],
        )

        observations, step_rewards, terminated, truncated, _ = env.step(
            {current_player: action}
        )
        rewards.update(step_rewards)
        turn += 1

        yield (
            _render_board(env),
            "\n".join(transcript),
            _format_status(turn, rewards, terminated or truncated),
        )

        if delay_seconds > 0:
            time.sleep(delay_seconds)

    final_status = "truncated" if truncated else "terminated"
    transcript.append(f"\n## Game {final_status}")
    transcript.append(
        f"Final rewards: Player 0 = `{rewards.get(0, 0)}`, "
        f"Player 1 = `{rewards.get(1, 0)}`"
    )

    for player_id, reward in rewards.items():
        loggers[player_id].log_rewards(GAME_NAME, 1, reward)
        loggers[player_id].log_game_result(
            game_name=GAME_NAME,
            episode=1,
            status=final_status,
            reward=reward,
            opponent=_opponent_label(config, player_id),
        )

    yield (
        _render_board(env),
        "\n".join(transcript),
        _format_status(turn, rewards, True),
    )


def build_app() -> Any:
    try:
        import gradio as gr
    except ImportError as exc:
        raise ImportError(
            "gradio is required for the live frontend. Install it with:\n"
            "python3 -m pip install gradio"
        ) from exc

    with gr.Blocks(title="Live Connect Four LLM Match") as demo:
        gr.Markdown(
            "# Live Connect Four: LLM vs Random\n"
            "Choose a provider/model preset, paste that provider's API key "
            "locally, click the button, and watch the match stream turn by "
            "turn. The key is placed in this Python process as the selected "
            "provider's API-key environment variable; it is not written to "
            "result logs."
        )

        with gr.Row():
            api_key = gr.Textbox(
                label="Provider API key",
                type="password",
                placeholder="OpenAI, Groq, or OpenRouter key",
            )
            model_preset = gr.Dropdown(
                label="Model preset",
                choices=list(MODEL_PRESETS.keys()),
                value=DEFAULT_PRESET,
            )

        with gr.Row():
            seed = gr.Number(label="Seed", value=42, precision=0)
            delay_seconds = gr.Slider(
                label="Delay between turns",
                minimum=0.0,
                maximum=5.0,
                value=0.5,
                step=0.25,
            )
            max_turns = gr.Number(label="Max turns", value=42, precision=0)

        run_button = gr.Button("Run selected model vs Random", variant="primary")

        with gr.Row():
            board = gr.Markdown(label="Board")
            status = gr.Textbox(label="Status", lines=4)

        transcript = gr.Markdown(label="Turn log and reasoning trace")

        run_button.click(
            fn=run_live_match,
            inputs=[api_key, seed, model_preset, delay_seconds, max_turns],
            outputs=[board, transcript, status],
        )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=7860)
