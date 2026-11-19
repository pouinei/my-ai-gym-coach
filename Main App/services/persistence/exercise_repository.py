import sqlite3
import streamlit as st
from pathlib import Path

_DB_PATH = str(Path(__file__).parent.parent.parent / "data.db")


@st.cache_resource
def _get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path=None) -> None:
    if db_path:
        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
    else:
        conn = _get_connection()

    with conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                username   TEXT UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS exercises (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id       INTEGER NOT NULL REFERENCES users(id),
                exercise_name TEXT    NOT NULL,
                reps          INTEGER NOT NULL DEFAULT 0,
                sets          INTEGER NOT NULL DEFAULT 0,
                time          INTEGER NOT NULL DEFAULT 0,
                created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS workout_sessions (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id       TEXT UNIQUE NOT NULL,
                user_id          INTEGER NOT NULL REFERENCES users(id),
                exercise_name    TEXT NOT NULL,
                target_sets      INTEGER NOT NULL DEFAULT 0,
                target_reps      INTEGER NOT NULL DEFAULT 0,
                sets_completed   INTEGER NOT NULL DEFAULT 0,
                total_reps       INTEGER NOT NULL DEFAULT 0,
                duration_seconds INTEGER NOT NULL DEFAULT 0,
                status           TEXT NOT NULL DEFAULT 'COMPLETED',
                created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )


def get_user(username: str, conn=None) -> sqlite3.Row:
    c = conn or _get_connection()
    return c.execute(
        "SELECT * FROM users WHERE username = ?", (username,)
    ).fetchone()


def create_user(username: str, conn=None) -> sqlite3.Row:
    c = conn or _get_connection()
    with c:
        c.execute(
            "INSERT INTO users (username) VALUES (?)", (username,)
        )
    return get_user(username, conn=c)


def get_or_create_user(username: str, conn=None) -> sqlite3.Row:
    user = get_user(username, conn=conn)
    if user is None:
        user = create_user(username, conn=conn)
    return user


def add_exercise(user_id, exercise_name, reps, sets, time_taken, conn=None):
    """
    Idempotently records completed sets/reps into the exercises table.
    Aggregates per user, exercise, and date.
    """
    c = conn or _get_connection()

    with c:
        existing = c.execute("""
            SELECT id, reps, sets, time FROM exercises 
            WHERE user_id = ? AND exercise_name = ? AND date(created_at) = date('now')
            ORDER BY id DESC LIMIT 1
        """, (user_id, exercise_name)).fetchone()

        if existing:
            c.execute("""
                UPDATE exercises 
                SET reps = reps + ?, sets = sets + ?, time = time + ?
                WHERE id = ?
            """, (reps, sets, time_taken, existing['id']))
        else:
            c.execute("""
                INSERT INTO exercises (user_id, exercise_name, sets, reps, time)
                VALUES (?, ?, ?, ?, ?)
            """, (user_id, exercise_name, sets, reps, time_taken))


def save_workout_session(
    user_id: int,
    session_id: str,
    exercise_name: str,
    target_sets: int,
    target_reps: int,
    sets_completed: int,
    total_reps: int,
    duration_seconds: int,
    status: str = "COMPLETED",
    conn=None,
):
    """
    Saves or updates a discrete workout session.
    Idempotent by session_id.
    """
    c = conn or _get_connection()
    with c:
        c.execute(
            """
            INSERT INTO workout_sessions (
                session_id, user_id, exercise_name, target_sets,
                target_reps, sets_completed, total_reps, duration_seconds, status
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                sets_completed = excluded.sets_completed,
                total_reps = excluded.total_reps,
                duration_seconds = excluded.duration_seconds,
                status = excluded.status
            """,
            (
                session_id,
                user_id,
                exercise_name,
                target_sets,
                target_reps,
                sets_completed,
                total_reps,
                duration_seconds,
                status,
            ),
        )


def get_users_exercises(user_id, conn=None):
    c = conn or _get_connection()
    return c.execute("""
        SELECT * FROM exercises 
        WHERE user_id = ?
        ORDER BY created_at DESC
    """, (user_id,)).fetchall()


def get_users_sessions(user_id, conn=None):
    c = conn or _get_connection()
    return c.execute("""
        SELECT * FROM workout_sessions 
        WHERE user_id = ?
        ORDER BY created_at DESC
    """, (user_id,)).fetchall()
