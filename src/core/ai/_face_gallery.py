"""Model-versioned in-memory face gallery with vectorized cosine search."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class EnrolledPerson:
    person_id: str
    name: str
    category: str
    active: bool
    sample_count: int
    enrollment_status: str
    model_id: str


@dataclass(frozen=True)
class FaceCandidate:
    person_id: str
    name: str
    category: str
    similarity: float
    sample_count: int


@dataclass(frozen=True)
class FaceSearchResult:
    matched: bool
    decision: str
    best: FaceCandidate | None
    second_best_similarity: float
    margin: float
    candidates: tuple[FaceCandidate, ...]
    model_id: str


@dataclass
class _GalleryRecord:
    person: EnrolledPerson
    embeddings: np.ndarray


class InMemoryFaceGallery:
    """Thread-safe gallery optimized for small/medium edge deployments."""

    def __init__(self, *, model_id: str, embedding_size: int = 128) -> None:
        self.model_id = str(model_id)
        self.embedding_size = int(embedding_size)
        self._records: dict[str, _GalleryRecord] = {}
        self._matrix = np.empty((0, self.embedding_size), dtype=np.float32)
        self._sample_person_ids: tuple[str, ...] = ()
        self._lock = threading.RLock()

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._records)

    def enroll(
        self,
        person: Mapping[str, object],
        embeddings: Sequence[np.ndarray],
    ) -> EnrolledPerson:
        person_id = str(person.get("person_id", "")).strip()
        name = str(person.get("name", "")).strip()
        if not person_id or not name:
            raise ValueError("person_id and name are required")
        normalized = self._normalize_embeddings(embeddings)
        with self._lock:
            if person_id in self._records:
                raise ValueError(f"person is already enrolled: {person_id}")
            enrolled = self._person_from(person, normalized.shape[0], active=True)
            self._records[person_id] = _GalleryRecord(enrolled, normalized)
            self._rebuild_snapshot()
            return enrolled

    def update(
        self,
        person_id: str,
        embeddings: Sequence[np.ndarray],
        *,
        active: bool | None = None,
    ) -> EnrolledPerson:
        normalized = self._normalize_embeddings(embeddings)
        with self._lock:
            record = self._require_record(person_id)
            current = record.person
            enrolled = EnrolledPerson(
                person_id=current.person_id,
                name=current.name,
                category=current.category,
                active=current.active if active is None else bool(active),
                sample_count=normalized.shape[0],
                enrollment_status=self._enrollment_status(normalized.shape[0]),
                model_id=self.model_id,
            )
            self._records[person_id] = _GalleryRecord(enrolled, normalized)
            self._rebuild_snapshot()
            return enrolled

    def deactivate(self, person_id: str) -> EnrolledPerson:
        with self._lock:
            record = self._require_record(person_id)
            current = record.person
            inactive = EnrolledPerson(
                person_id=current.person_id,
                name=current.name,
                category=current.category,
                active=False,
                sample_count=current.sample_count,
                enrollment_status=current.enrollment_status,
                model_id=current.model_id,
            )
            self._records[person_id] = _GalleryRecord(inactive, record.embeddings)
            self._rebuild_snapshot()
            return inactive

    def delete(self, person_id: str) -> bool:
        with self._lock:
            if self._records.pop(person_id, None) is None:
                return False
            self._rebuild_snapshot()
            return True

    def reload(self) -> int:
        """Keep the interface compatible with future persistent implementations."""
        with self._lock:
            self._rebuild_snapshot()
            return len(self._records)

    def search(
        self,
        embedding: np.ndarray,
        *,
        top_k: int = 5,
        threshold: float = 0.5,
        margin: float = 0.1,
    ) -> FaceSearchResult:
        query = self._normalize_vector(embedding)
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        with self._lock:
            matrix = self._matrix
            sample_person_ids = self._sample_person_ids
            records = dict(self._records)
        if matrix.shape[0] == 0:
            return self._empty_result()

        similarities = matrix @ query
        best_by_person: dict[str, float] = {}
        for index, person_id in enumerate(sample_person_ids):
            score = float(similarities[index])
            best_by_person[person_id] = max(best_by_person.get(person_id, -1.0), score)
        ranked = sorted(best_by_person.items(), key=lambda item: item[1], reverse=True)
        candidates = tuple(
            FaceCandidate(
                person_id=person_id,
                name=records[person_id].person.name,
                category=records[person_id].person.category,
                similarity=score,
                sample_count=records[person_id].person.sample_count,
            )
            for person_id, score in ranked[:top_k]
        )
        best = candidates[0]
        second_score = candidates[1].similarity if len(candidates) > 1 else 0.0
        score_margin = best.similarity - second_score
        if best.similarity < threshold:
            decision, matched = "unknown", False
        elif len(candidates) > 1 and score_margin < margin:
            decision, matched = "ambiguous", False
        else:
            decision, matched = "matched", True
        return FaceSearchResult(
            matched=matched,
            decision=decision,
            best=best,
            second_best_similarity=second_score,
            margin=score_margin,
            candidates=candidates,
            model_id=self.model_id,
        )

    def _normalize_embeddings(
        self, embeddings: Sequence[np.ndarray]
    ) -> np.ndarray:
        if not embeddings:
            raise ValueError("at least one face embedding is required")
        return np.stack([self._normalize_vector(item) for item in embeddings])

    def _normalize_vector(self, embedding: np.ndarray) -> np.ndarray:
        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if vector.size != self.embedding_size:
            raise ValueError(
                f"face embedding must contain {self.embedding_size} values, "
                f"received {vector.size}"
            )
        if not np.all(np.isfinite(vector)):
            raise ValueError("face embedding must contain only finite values")
        norm = float(np.linalg.norm(vector))
        if norm <= 0:
            raise ValueError("face embedding has zero-norm")
        return np.ascontiguousarray(vector / norm)

    def _person_from(
        self, person: Mapping[str, object], sample_count: int, *, active: bool
    ) -> EnrolledPerson:
        return EnrolledPerson(
            person_id=str(person["person_id"]).strip(),
            name=str(person["name"]).strip(),
            category=str(person.get("category", "employee")).strip() or "employee",
            active=active,
            sample_count=sample_count,
            enrollment_status=self._enrollment_status(sample_count),
            model_id=self.model_id,
        )

    @staticmethod
    def _enrollment_status(sample_count: int) -> str:
        return "single_sample" if sample_count == 1 else "multi_sample"

    def _require_record(self, person_id: str) -> _GalleryRecord:
        try:
            return self._records[person_id]
        except KeyError as exc:
            raise KeyError(f"person not found: {person_id}") from exc

    def _rebuild_snapshot(self) -> None:
        matrices: list[np.ndarray] = []
        person_ids: list[str] = []
        for person_id, record in self._records.items():
            if not record.person.active:
                continue
            matrices.append(record.embeddings)
            person_ids.extend([person_id] * record.embeddings.shape[0])
        self._matrix = (
            np.concatenate(matrices, axis=0)
            if matrices
            else np.empty((0, self.embedding_size), dtype=np.float32)
        )
        self._sample_person_ids = tuple(person_ids)

    def _empty_result(self) -> FaceSearchResult:
        return FaceSearchResult(
            matched=False,
            decision="unknown",
            best=None,
            second_best_similarity=0.0,
            margin=0.0,
            candidates=(),
            model_id=self.model_id,
        )
