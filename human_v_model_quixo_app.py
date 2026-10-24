from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st

from lib import agents_quixo, game_quixo, model_quixo
from lib.mcts_quixo import MCTS


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SAVES_DIR = Path("saves")
DEFAULT_MCTS_SIMULATIONS = 200
SIZE = game_quixo.SIZE

# Soft, low-contrast palette (pair with .streamlit/config.toml for widgets).
C_BG = "#f2efe9"
C_SURFACE = "#fbfaf7"
C_BORDER = "#ddd8cc"
C_TEXT = "#2e3b43"
C_MUTED = "#7a858b"
C_X, C_X_TEXT = "#9fbfd9", "#1f3446"
C_O, C_O_TEXT = "#e3a994", "#4a2a20"
C_BLANK = "#ebe6da"
C_SAGE, C_SAGE_DARK = "#5f8f76", "#2f6b50"
C_CHANGED = "#e0b94f"

ARROWS = {"UP": "↑", "DOWN": "↓", "LEFT": "←", "RIGHT": "→"}

# Agent pacing (seconds)
AGENT_START_DELAY = 0.5      # pause after your move before the agent starts thinking
AGENT_PREVIEW_SECONDS = 1.2  # how long the agent's chosen move is shown before it is played

st.set_page_config(
    page_title="Quixo Arena",
    page_icon="🎲",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    f"""
    <style>
      .stApp {{ background: {C_BG}; color: {C_TEXT}; }}
      [data-testid="stHeader"] {{ background: rgba(0,0,0,0); }}
      .stApp h1, .stApp h2, .stApp h3, .stApp p, .stApp label,
      .stApp [data-testid="stMarkdownContainer"] {{ color: {C_TEXT}; }}
      [data-testid="stMetric"] {{
        background: {C_SURFACE};
        border: 1px solid {C_BORDER};
        border-radius: 12px;
        padding: 12px 16px;
      }}
      .stApp [data-testid="stBaseButton-secondary"] {{
        background: {C_SURFACE}; color: {C_TEXT}; border: 1px solid {C_BORDER};
      }}
      .stApp [data-testid="stBaseButton-primary"] {{
        background: #5f7f8c; color: #fff; border: 1px solid #5f7f8c;
      }}
      div.stButton > button {{ border-radius: 10px; min-height: 2.8rem; font-weight: 600; }}
      .hero {{
        padding: 1.2rem 1.4rem;
        border: 1px solid {C_BORDER};
        border-radius: 16px;
        background: {C_SURFACE};
        margin-bottom: 1rem;
      }}
      .muted {{ color: {C_MUTED} !important; }}

      /* Board layout: fixed 5x5, never stacks on narrow screens */
      .st-key-board {{ max-width: 460px; margin: 0 auto; }}
      .st-key-board [data-testid="stHorizontalBlock"] {{
        flex-wrap: nowrap !important; gap: 6px !important;
      }}
      .st-key-board [data-testid="stColumn"],
      .st-key-board [data-testid="column"] {{
        min-width: 0 !important; flex: 1 1 0 !important; width: auto !important;
      }}
      .st-key-board [data-testid="stVerticalBlock"] {{ gap: 6px !important; }}
      .st-key-board button {{
        height: 76px !important; min-height: 76px !important;
        border-radius: 12px !important; padding: 0 !important;
      }}
      .st-key-board button p {{
        font-size: 30px !important; font-weight: 800 !important; line-height: 1 !important;
      }}
      .st-key-board button:not(:disabled):hover {{ filter: brightness(.95); }}
      .st-key-board button:disabled {{ cursor: default; }}

      /* Read-only board used to preview the agent's chosen move */
      .qpreview {{
        display: grid; grid-template-columns: repeat(5, 1fr); gap: 6px;
        max-width: 460px; margin: 0 auto;
      }}
      .qpreview div {{
        height: 76px; display: flex; align-items: center; justify-content: center;
        border-radius: 12px; box-sizing: border-box;
        font: 800 30px/1 sans-serif;
      }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Model loading and agent setup
# ---------------------------------------------------------------------------
@st.cache_data(ttl=10, show_spinner=False)
def find_checkpoints(root: str) -> list[str]:
    root_path = Path(root)
    if not root_path.exists():
        return []
    return sorted((str(p) for p in root_path.rglob("*.npz")), reverse=True)


def _load_model_uncached(checkpoint_path: Path):
    net = model_quixo.NTupleNetwork()

    with np.load(checkpoint_path, allow_pickle=False) as checkpoint:
        for i, tup in enumerate(net.tuples):
            weights_key = f"tuple_{i}_weights"
            positions_key = f"tuple_{i}_positions"

            if weights_key not in checkpoint.files:
                raise ValueError(
                    f"Checkpoint is missing {weights_key}. "
                    "It may have been saved using a different format."
                )

            weights = checkpoint[weights_key]
            if weights.shape != tup.weights.shape:
                raise ValueError(
                    f"{weights_key} has shape {weights.shape}; "
                    f"expected {tup.weights.shape}."
                )

            if positions_key in checkpoint.files:
                saved_positions = checkpoint[positions_key]
                if not np.array_equal(saved_positions, tup.positions):
                    raise ValueError(
                        f"Tuple {i} positions differ from the current "
                        "model definition. Load the matching tuple definition."
                    )

            tup.weights[:] = weights

    return net


@st.cache_resource(show_spinner=False)
def load_model(checkpoint_path: str, mtime: float):
    """Cached per (path, modification time). Shared across sessions, so agents
    must only read from the network (true for play)."""
    return _load_model_uncached(Path(checkpoint_path))


def make_agent(agent_type: str, net: Any, n_simulations: int):
    if agent_type == "TD Agent":
        return agents_quixo.TDAgent(net)
    return agents_quixo.MCTSAgent(
        net=net,
        mcts=MCTS(c_puct=1.0),
        n_sims=n_simulations,
    )


# ---------------------------------------------------------------------------
# Game / action helpers
# ---------------------------------------------------------------------------
def player_name(player: int) -> str:
    return "X" if player == game_quixo.PLAYER_X else "O"


def board_array(state: Any) -> np.ndarray:
    return np.asarray(state).reshape(SIZE, SIZE).astype(int)


def action_description(action: int) -> str:
    border_index, direction = game_quixo.ACTION_MAP[action]
    row, col = game_quixo.BORDER_PIECES[border_index]
    return f"row {row + 1}, col {col + 1} {ARROWS[direction]} {direction.lower()}"


def destination_cell(border_index: int, direction: str) -> tuple[int, int]:
    """Cell where the picked-up cube is re-inserted.

    Assumes `direction` is the way the line of cubes shifts, which is what
    legal_directions() implies (e.g. row 0 has UP but no DOWN: the cube goes
    to the bottom of its column and the column shifts up).
    """
    row, col = game_quixo.BORDER_PIECES[border_index]
    return {
        "UP": (SIZE - 1, col),
        "DOWN": (0, col),
        "LEFT": (row, SIZE - 1),
        "RIGHT": (row, 0),
    }[direction]


def legal_by_source(player: int) -> dict[int, dict[str, int]]:
    """{border_index: {direction: action}} for the current position.
    Computed once per position, not on every rerun."""
    key = (st.session_state.move_number, player)
    cache = st.session_state.legal_cache
    if key not in cache:
        cache.clear()
        result: dict[int, dict[str, int]] = {}
        for a in game_quixo.possible_moves(st.session_state.game_state, player):
            border_index, direction = game_quixo.ACTION_MAP[int(a)]
            result.setdefault(border_index, {})[direction] = int(a)
        cache[key] = result
    return cache[key]


def agent_choose_action(agent: Any, state: np.ndarray, player: int, agent_type: str) -> int:
    if agent_type == "MCTS Agent":
        result = agent.select_action(state, player, tau=0)
    else:
        result = agent.select_action(state, player)
    action = result[0] if isinstance(result, tuple) else result
    return int(action)


def reset_game() -> None:
    st.session_state.game_state = np.array(game_quixo.INITIAL_STATE, copy=True)
    st.session_state.current_player = game_quixo.PLAYER_X
    st.session_state.winner = None
    st.session_state.changed_cells = []
    st.session_state.selected_src = None
    st.session_state.last_move_text = "The game is ready."
    st.session_state.move_number = 0
    st.session_state.history = []
    st.session_state.legal_cache = {}
    agent = st.session_state.get("agent")
    if agent is not None and hasattr(agent, "mcts"):
        agent.mcts.clear()


def apply_move(action: int, player: int, move_source: str) -> None:
    old_board = board_array(st.session_state.game_state)
    new_state, won = game_quixo.move(
        st.session_state.game_state.copy(), int(action), int(player)
    )
    st.session_state.game_state = np.array(new_state, copy=True)

    new_board = board_array(st.session_state.game_state)
    st.session_state.changed_cells = [
        int(i) for i in np.flatnonzero((old_board != new_board).ravel())
    ]
    st.session_state.selected_src = None

    st.session_state.move_number += 1
    description = action_description(int(action))
    st.session_state.last_move_text = (
        f"{move_source} played **{player_name(player)}** — {description}."
    )
    st.session_state.history.append(
        f"{st.session_state.move_number}. {move_source}: {player_name(player)} — "
        f"{description}"
    )

    if isinstance(won, (bool, np.bool_)):
        if won:
            st.session_state.winner = player
    elif won == player:
        st.session_state.winner = player
    elif won == -player:
        st.session_state.winner = -player

    if st.session_state.winner is None:
        st.session_state.current_player = -player


# Button callbacks (run before the rerun, so the board draws the new state once)
def on_pick_source(border_index: int) -> None:
    st.session_state.selected_src = (
        None if st.session_state.selected_src == border_index else border_index
    )


def on_pick_destination(action: int) -> None:
    apply_move(action, st.session_state.human_player, "You")


# ---------------------------------------------------------------------------
# Board rendering
# ---------------------------------------------------------------------------
def render_board() -> None:
    state = board_array(st.session_state.game_state)
    human = st.session_state.human_player
    human_turn = (
        st.session_state.winner is None and st.session_state.current_player == human
    )
    changed = set(st.session_state.changed_cells)

    by_src = legal_by_source(human) if human_turn else {}
    src_cells = {game_quixo.BORDER_PIECES[bi]: bi for bi in by_src}

    selected = st.session_state.selected_src
    if selected not in by_src:
        selected = None
    dest_cells: dict[tuple[int, int], tuple[str, int]] = {}
    if selected is not None:
        for direction, action in by_src[selected].items():
            dest_cells[destination_cell(selected, direction)] = (direction, action)
    selected_cell = game_quixo.BORDER_PIECES[selected] if selected is not None else None

    css: list[str] = []
    with st.container(key="board"):
        for r in range(SIZE):
            cols = st.columns(SIZE, gap="small")
            for c in range(SIZE):
                v = int(state[r, c])
                bg, fg = {1: (C_X, C_X_TEXT), -1: (C_O, C_O_TEXT)}.get(v, (C_BLANK, C_TEXT))
                label = {1: "X", -1: "O"}.get(v, "\u00a0")
                border = "2px solid rgba(46,59,67,.12)"
                shadow = f"0 0 0 3px {C_CHANGED}" if r * SIZE + c in changed else "none"

                on_click, args, disabled = None, (), True
                if (r, c) == selected_cell:
                    border = f"3px solid {C_SAGE_DARK}"
                    shadow = f"0 0 0 4px rgba(95,143,118,.35)"
                    on_click, args, disabled = on_pick_source, (selected,), False
                elif (r, c) in dest_cells:
                    direction, action = dest_cells[(r, c)]
                    label = ARROWS[direction]
                    fg = C_SAGE_DARK
                    border = f"3px dashed {C_SAGE_DARK}"
                    on_click, args, disabled = on_pick_destination, (action,), False
                elif (r, c) in src_cells:
                    border = f"3px solid {C_SAGE}"
                    on_click, args, disabled = on_pick_source, (src_cells[(r, c)],), False

                key = f"cell_{r}_{c}"
                css.append(
                    f".st-key-{key} button, .st-key-{key} button:disabled {{"
                    f"background:{bg} !important; color:{fg} !important;"
                    f"border:{border} !important; box-shadow:{shadow} !important;"
                    f"opacity:1 !important;}}"
                    f".st-key-{key} button p {{color:{fg} !important;}}"
                )
                with cols[c]:
                    st.button(
                        label,
                        key=key,
                        disabled=disabled,
                        on_click=on_click,
                        args=args,
                        use_container_width=True,
                    )

    st.markdown(f"<style>{''.join(css)}</style>", unsafe_allow_html=True)

    if human_turn:
        if selected is None:
            st.caption("Green-outlined tiles can be picked up. Click one to see where it can go.")
        else:
            row, col = game_quixo.BORDER_PIECES[selected]
            st.caption(
                f"Picked up row {row + 1}, col {col + 1}. Click a dashed tile to insert it "
                "there (arrow = direction the line slides), or click the tile again to cancel."
            )
    else:
        st.caption("Amber outline marks tiles changed by the last move.")


def preview_board_html(
    source: tuple[int, int], dest: tuple[int, int], arrow: str
) -> str:
    """Static board showing the agent's chosen move before it is played:
    amber = tile picked up, dashed green arrow = where it is inserted."""
    state = board_array(st.session_state.game_state)
    cells = []
    for r in range(SIZE):
        for c in range(SIZE):
            v = int(state[r, c])
            bg, fg = {1: (C_X, C_X_TEXT), -1: (C_O, C_O_TEXT)}.get(v, (C_BLANK, C_TEXT))
            label = {1: "X", -1: "O"}.get(v, "&nbsp;")
            border = "2px solid rgba(46,59,67,.12)"
            shadow = "none"
            if (r, c) == source:
                border = f"3px solid {C_CHANGED}"
                shadow = "0 0 0 4px rgba(224,185,79,.45)"
            elif (r, c) == dest:
                label, fg = arrow, C_SAGE_DARK
                border = f"3px dashed {C_SAGE_DARK}"
            cells.append(
                f'<div style="background:{bg};color:{fg};border:{border};'
                f'box-shadow:{shadow}">{label}</div>'
            )
    return f'<div class="qpreview">{"".join(cells)}</div>'


def show_agent_preview(slot: Any, action: int, agent_type: str) -> None:
    border_index, direction = game_quixo.ACTION_MAP[action]
    source = game_quixo.BORDER_PIECES[border_index]
    dest = destination_cell(border_index, direction)
    with slot.container():
        st.markdown(
            preview_board_html(source, dest, ARROWS[direction]),
            unsafe_allow_html=True,
        )
        st.caption(
            f"{agent_type} picks up row {source[0] + 1}, col {source[1] + 1} "
            f"and slides {direction.lower()} {ARROWS[direction]}"
        )


# ---------------------------------------------------------------------------
# Setup screen
# ---------------------------------------------------------------------------
def setup_screen() -> None:
    st.markdown(
        """
        <div class="hero">
          <h1 style="margin:0">🎲 Quixo Arena</h1>
          <p class="muted" style="margin:.45rem 0 0">
            Play against your saved temporal-difference model or search-based MCTS agent.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    checkpoints = find_checkpoints(str(SAVES_DIR))
    if not checkpoints:
        st.error(
            f"No .npz checkpoints found under `{SAVES_DIR.resolve()}`. "
            "Place a saved model in a folder inside `saves/` and refresh."
        )
        st.code("saves/20261008/td_agent_step_10000.npz", language="text")
        return

    display_names = [str(Path(p).relative_to(SAVES_DIR)) for p in checkpoints]

    left, right = st.columns([1.15, 1])
    with left:
        st.subheader("1. Choose a saved model")
        selected_name = st.selectbox(
            "Checkpoint",
            options=display_names,
            index=0,
            help="All .npz files found recursively inside saves/ are listed.",
        )
        st.caption(f"Found {len(checkpoints)} checkpoint(s) in `{SAVES_DIR}`.")

        st.subheader("2. Choose your opponent")
        agent_type = st.radio(
            "Agent",
            ["TD Agent", "MCTS Agent"],
            horizontal=True,
            help="TD selects from the tuple network. MCTS performs a search guided by the network.",
        )
        simulations = DEFAULT_MCTS_SIMULATIONS
        if agent_type == "MCTS Agent":
            simulations = st.slider(
                "MCTS simulations per move",
                min_value=25,
                max_value=1000,
                value=DEFAULT_MCTS_SIMULATIONS,
                step=25,
                help="More simulations usually take longer per move.",
            )

    with right:
        st.subheader("3. Choose your side")
        human_side = st.radio(
            "You play as",
            ["X (moves first)", "O (moves second)"],
            horizontal=False,
        )
        st.markdown(
            f"""
            <div class="hero">
              <h3 style="margin-top:0">Board guide</h3>
              <p><span style="color:{C_X}">■</span> X — blue</p>
              <p><span style="color:{C_O}">■</span> O — coral</p>
              <p><span style="color:{C_BLANK}; text-shadow:0 0 1px #999">■</span> Blank — sand</p>
              <p class="muted">Click a green-outlined edge tile, then click where to insert it.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if st.button("Start game", type="primary", use_container_width=True):
        selected_path = SAVES_DIR / selected_name
        try:
            with st.spinner("Loading checkpoint and preparing agent..."):
                net = load_model(str(selected_path), selected_path.stat().st_mtime)
                agent = make_agent(agent_type, net, simulations)
        except Exception as exc:
            st.error(f"Could not load the selected checkpoint: {exc}")
            return

        st.session_state.agent = agent
        st.session_state.agent_type = agent_type
        st.session_state.model_path = str(selected_path)
        st.session_state.human_player = (
            game_quixo.PLAYER_X if human_side.startswith("X") else game_quixo.PLAYER_O
        )
        st.session_state.mcts_simulations = simulations
        st.session_state.screen = "game"
        reset_game()
        st.rerun()


# ---------------------------------------------------------------------------
# Game screen
# ---------------------------------------------------------------------------
def game_screen() -> None:
    agent_type = st.session_state.agent_type
    human_player = st.session_state.human_player
    current_player = st.session_state.current_player
    winner = st.session_state.winner
    agent_to_move = winner is None and current_player != human_player

    st.markdown(
        """
        <div class="hero">
          <h1 style="margin:0">Quixo Arena</h1>
          <p class="muted" style="margin:.45rem 0 0">Your match is in progress.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric1, metric2, metric3 = st.columns(3)
    metric1.metric("You play", player_name(human_player))
    metric2.metric("Opponent", agent_type)
    metric3.metric("Move number", st.session_state.move_number)

    board_col, side_col = st.columns([1.2, 1], gap="large")
    with board_col:
        board_slot = st.empty()  # lets us swap in the agent-move preview
        with board_slot.container():
            render_board()

    with side_col:
        st.subheader("Match status")
        if winner is not None:
            if winner == human_player:
                st.success(f"You win as {player_name(human_player)}.")
            else:
                st.error(f"The agent wins as {player_name(winner)}.")
        elif current_player == human_player:
            st.info(f"Your turn — play as {player_name(human_player)}.")
        else:
            st.info(f"{agent_type} is choosing a move as {player_name(current_player)}.")

        st.markdown("**Last move**")
        st.markdown(st.session_state.last_move_text)

        st.markdown("---")
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("New game", use_container_width=True):
                reset_game()
                st.rerun()
        with col_b:
            if st.button("Change setup", use_container_width=True):
                st.session_state.screen = "setup"
                st.rerun()

        with st.expander("Move history", expanded=False):
            if st.session_state.history:
                for item in reversed(st.session_state.history):
                    st.write(item)
            else:
                st.caption("No moves have been made yet.")

    # The page is already drawn, so your move stays visible while the agent
    # pauses, thinks, previews its choice, then plays it.
    if agent_to_move:
        time.sleep(AGENT_START_DELAY)
        with st.spinner(f"{agent_type} is thinking..."):
            action = agent_choose_action(
                st.session_state.agent,
                st.session_state.game_state,
                current_player,
                agent_type,
            )
        show_agent_preview(board_slot, action, agent_type)
        time.sleep(AGENT_PREVIEW_SECONDS)
        apply_move(action, current_player, agent_type)
        st.rerun()


# ---------------------------------------------------------------------------
# Session initialisation and app entry point
# ---------------------------------------------------------------------------
if "screen" not in st.session_state:
    st.session_state.screen = "setup"

if st.session_state.screen == "setup":
    setup_screen()
else:
    game_screen()
