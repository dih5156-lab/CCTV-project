"""SQLite persistence for model-versioned face identities and embeddings."""

from __future__ import annotations

import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ._face_gallery import FaceSearchResult, InMemoryFaceGallery


class SQLiteFaceGallery:
    def __init__(
        self,
        database_path: Path,
        *,
        model_id: str,
        embedding_size: int = 128,
    ) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.model_id = str(model_id)
        self.embedding_size = int(embedding_size)
        self._lock = threading.RLock()
        self._memory = InMemoryFaceGallery(
            model_id=self.model_id, embedding_size=self.embedding_size
        )
        self._initialize_schema()
        self.reload()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize_schema(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS face_persons (
                    person_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    phone TEXT NOT NULL UNIQUE,
                    category TEXT NOT NULL DEFAULT 'employee',
                    active INTEGER NOT NULL DEFAULT 1,
                    model_id TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS face_samples (
                    sample_id TEXT PRIMARY KEY,
                    person_id TEXT NOT NULL,
                    image_path TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    dimension INTEGER NOT NULL,
                    model_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(person_id) REFERENCES face_persons(person_id)
                        ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_face_samples_person
                    ON face_samples(person_id);
                """
            )

    def enroll_person(
        self,
        person: Mapping[str, object],
        embeddings: Sequence[np.ndarray],
        image_paths: Sequence[str],
    ) -> dict[str, object]:
        vectors = self._normalize_embeddings(embeddings)
        if len(image_paths) != len(vectors):
            raise ValueError("one image path is required for each embedding")
        person_id = str(person.get("person_id", "")).strip()
        name = str(person.get("name", "")).strip()
        phone = str(person.get("phone", "")).strip()
        category = str(person.get("category", "employee")).strip() or "employee"
        if not person_id or not name or not phone:
            raise ValueError("person_id, name, and phone are required")
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self._connect() as connection:
            connection.execute(
                "INSERT INTO face_persons "
                "(person_id, name, phone, category, active, model_id, created_at) "
                "VALUES (?, ?, ?, ?, 1, ?, ?)",
                (person_id, name, phone, category, self.model_id, now),
            )
            self._insert_samples(connection, person_id, vectors, image_paths, now)
        self.reload()
        return self.get_person(person_id)

    def add_samples(
        self,
        person_id: str,
        embeddings: Sequence[np.ndarray],
        image_paths: Sequence[str],
    ) -> dict[str, object]:
        vectors = self._normalize_embeddings(embeddings)
        if len(image_paths) != len(vectors):
            raise ValueError("one image path is required for each embedding")
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self._connect() as connection:
            exists = connection.execute(
                "SELECT 1 FROM face_persons WHERE person_id = ? AND model_id = ?",
                (person_id, self.model_id),
            ).fetchone()
            if exists is None:
                raise KeyError(f"person not found: {person_id}")
            self._insert_samples(connection, person_id, vectors, image_paths, now)
        self.reload()
        return self.get_person(person_id)

    def _insert_samples(
        self,
        connection: sqlite3.Connection,
        person_id: str,
        vectors: Sequence[np.ndarray],
        image_paths: Sequence[str],
        created_at: str,
    ) -> None:
        connection.executemany(
            "INSERT INTO face_samples "
            "(sample_id, person_id, image_path, embedding, dimension, model_id, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    uuid.uuid4().hex,
                    person_id,
                    str(image_path),
                    vector.astype(np.float32).tobytes(),
                    self.embedding_size,
                    self.model_id,
                    created_at,
                )
                for vector, image_path in zip(vectors, image_paths)
            ],
        )

    def reload(self) -> int:
        memory = InMemoryFaceGallery(
            model_id=self.model_id, embedding_size=self.embedding_size
        )
        with self._lock, self._connect() as connection:
            people = connection.execute(
                "SELECT * FROM face_persons WHERE model_id = ? ORDER BY created_at",
                (self.model_id,),
            ).fetchall()
            for person in people:
                samples = connection.execute(
                    "SELECT embedding FROM face_samples "
                    "WHERE person_id = ? AND model_id = ? ORDER BY created_at",
                    (person["person_id"], self.model_id),
                ).fetchall()
                vectors = [
                    np.frombuffer(sample["embedding"], dtype=np.float32).copy()
                    for sample in samples
                ]
                if not vectors:
                    continue
                memory.enroll(
                    {
                        "person_id": person["person_id"],
                        "name": person["name"],
                        "category": person["category"],
                    },
                    vectors,
                )
                if not bool(person["active"]):
                    memory.deactivate(person["person_id"])
            self._memory = memory
        return self._memory.size

    def search(self, embedding: np.ndarray, **kwargs: object) -> FaceSearchResult:
        with self._lock:
            memory = self._memory
        return memory.search(embedding, **kwargs)

    def list_people(self) -> list[dict[str, object]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT p.*, COUNT(s.sample_id) AS sample_count, "
                "GROUP_CONCAT(s.image_path, '|') AS images "
                "FROM face_persons p LEFT JOIN face_samples s "
                "ON s.person_id = p.person_id AND s.model_id = p.model_id "
                "WHERE p.model_id = ? GROUP BY p.person_id ORDER BY p.created_at",
                (self.model_id,),
            ).fetchall()
        return [self._row_to_person(row) for row in rows]

    def get_person(self, person_id: str) -> dict[str, object]:
        for person in self.list_people():
            if person["person_id"] == person_id:
                return person
        raise KeyError(f"person not found: {person_id}")

    def deactivate(self, person_id: str) -> dict[str, object]:
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                "UPDATE face_persons SET active = 0 "
                "WHERE person_id = ? AND model_id = ?",
                (person_id, self.model_id),
            )
            if cursor.rowcount == 0:
                raise KeyError(f"person not found: {person_id}")
        self.reload()
        return self.get_person(person_id)

    def delete(self, person_id: str) -> bool:
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM face_persons WHERE person_id = ? AND model_id = ?",
                (person_id, self.model_id),
            )
        if cursor.rowcount == 0:
            return False
        self.reload()
        return True

    def _normalize_embeddings(
        self, embeddings: Sequence[np.ndarray]
    ) -> list[np.ndarray]:
        if not embeddings:
            raise ValueError("at least one face embedding is required")
        vectors: list[np.ndarray] = []
        for embedding in embeddings:
            vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
            if vector.size != self.embedding_size:
                raise ValueError(f"face embedding must contain {self.embedding_size} values")
            if not np.all(np.isfinite(vector)):
                raise ValueError("face embedding must contain only finite values")
            norm = float(np.linalg.norm(vector))
            if norm <= 0:
                raise ValueError("face embedding has zero-norm")
            vectors.append(np.ascontiguousarray(vector / norm))
        return vectors

    @staticmethod
    def _row_to_person(row: sqlite3.Row) -> dict[str, object]:
        sample_count = int(row["sample_count"])
        images = str(row["images"] or "").split("|") if row["images"] else []
        return {
            "id": row["person_id"],
            "person_id": row["person_id"],
            "name": row["name"],
            "phone": row["phone"],
            "category": row["category"],
            "active": bool(row["active"]),
            "sample_count": sample_count,
            "enrollment_status": "single_sample" if sample_count == 1 else "multi_sample",
            "embedding_model": row["model_id"],
            "images": images,
            "image": images[0] if images else "",
            "registered_at": row["created_at"],
        }
