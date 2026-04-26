from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATABASE_DIRECTORY = PROJECT_ROOT / "data"
DATABASE_PATH = DATABASE_DIRECTORY / "facemark.db"


@dataclass(slots=True)
class Person:
    id: int
    name: str
    notes: str | None
    createdAt: str
    updatedAt: str


@dataclass(slots=True)
class PersonSummary:
    id: int
    name: str
    notes: str | None
    sampleCount: int
    createdAt: str
    updatedAt: str


@dataclass(slots=True)
class NewFaceSample:
    imagePath: str
    rectX: int
    rectY: int
    rectWidth: int
    rectHeight: int
    embedding: bytes
    embeddingLength: int
    faceImage: bytes
    originalImage: bytes


@dataclass(slots=True)
class FaceSample:
    id: int
    personId: int
    imagePath: str
    rectX: int
    rectY: int
    rectWidth: int
    rectHeight: int
    embedding: bytes | None
    embeddingLength: int
    faceImage: bytes | None
    hasOriginalImage: bool
    createdAt: str


class PersonRepository:
    def __init__(self, databasePath: Path = DATABASE_PATH):
        self.databasePath = databasePath
        self.databasePath.parent.mkdir(parents=True, exist_ok=True)
        self._initializeDatabase()

    def createPerson(self, name: str, notes: str | None = None) -> Person:
        with self._connect() as connection:
            self._ensurePersonNameAvailable(connection, name)
            cursor = connection.execute(
                """
                INSERT INTO people (name, notes)
                VALUES (?, ?)
                RETURNING id, name, notes, created_at, updated_at
                """,
                (name, notes),
            )
            row = cursor.fetchone()

        return self._personFromRow(row)

    def createPersonWithSamples(
        self,
        name: str,
        faceSamples: list[NewFaceSample],
        notes: str | None = None,
    ) -> Person:
        if not faceSamples:
            raise ValueError("At least one face sample is required.")

        with self._connect() as connection:
            self._ensurePersonNameAvailable(connection, name)
            cursor = connection.execute(
                """
                INSERT INTO people (name, notes)
                VALUES (?, ?)
                RETURNING id, name, notes, created_at, updated_at
                """,
                (name, notes),
            )
            row = cursor.fetchone()
            person = self._personFromRow(row)
            self._insertFaceSamples(connection, person.id, faceSamples)

        return person

    def addFaceSamples(self, personId: int, faceSamples: list[NewFaceSample]) -> int:
        if not faceSamples:
            return 0

        with self._connect() as connection:
            personRow = connection.execute(
                "SELECT id FROM people WHERE id = ?",
                (personId,),
            ).fetchone()
            if personRow is None:
                raise ValueError(f"Person with id {personId} does not exist.")

            self._insertFaceSamples(connection, personId, faceSamples)
            connection.execute(
                """
                UPDATE people
                SET updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (personId,),
            )

        return len(faceSamples)

    def listPeople(self) -> list[Person]:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                SELECT id, name, notes, created_at, updated_at
                FROM people
                ORDER BY name COLLATE NOCASE, id
                """
            )
            rows = cursor.fetchall()

        return [self._personFromRow(row) for row in rows]

    def listPeopleSummaries(self) -> list[PersonSummary]:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                SELECT
                    people.id,
                    people.name,
                    people.notes,
                    COUNT(face_samples.id) AS sample_count,
                    people.created_at,
                    people.updated_at
                FROM people
                LEFT JOIN face_samples ON face_samples.person_id = people.id
                GROUP BY people.id, people.name, people.notes, people.created_at, people.updated_at
                ORDER BY people.name COLLATE NOCASE, people.id
                """
            )
            rows = cursor.fetchall()

        return [self._personSummaryFromRow(row) for row in rows]

    def listFaceSamplesForPerson(self, personId: int) -> list[FaceSample]:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                SELECT
                    id,
                    person_id,
                    image_path,
                    rect_x,
                    rect_y,
                    rect_width,
                    rect_height,
                    embedding,
                    embedding_length,
                    face_image,
                    original_image IS NOT NULL AS has_original_image,
                    created_at
                FROM face_samples
                WHERE person_id = ?
                ORDER BY created_at, id
                """,
                (personId,),
            )
            rows = cursor.fetchall()

        return [self._faceSampleFromRow(row) for row in rows]

    def getStoredOriginalImageForFaceSample(self, faceSampleId: int) -> bytes | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT original_image
                FROM face_samples
                WHERE id = ?
                """,
                (faceSampleId,),
            ).fetchone()

        if row is None or row[0] is None:
            return None

        return bytes(row[0])

    def renamePerson(self, personId: int, name: str) -> Person | None:
        with self._connect() as connection:
            personRow = connection.execute(
                "SELECT id FROM people WHERE id = ?",
                (personId,),
            ).fetchone()
            if personRow is None:
                return None

            self._ensurePersonNameAvailable(connection, name, excludedPersonId=personId)
            cursor = connection.execute(
                """
                UPDATE people
                SET name = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                RETURNING id, name, notes, created_at, updated_at
                """,
                (name, personId),
            )
            row = cursor.fetchone()

        return self._personFromRow(row)

    def deletePerson(self, personId: int) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM people WHERE id = ?",
                (personId,),
            )
            return cursor.rowcount > 0

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.databasePath)
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initializeDatabase(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS people (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL COLLATE NOCASE,
                    notes TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS face_samples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id INTEGER NOT NULL,
                    image_path TEXT NOT NULL,
                    rect_x INTEGER NOT NULL,
                    rect_y INTEGER NOT NULL,
                    rect_width INTEGER NOT NULL,
                    rect_height INTEGER NOT NULL,
                    embedding BLOB,
                    embedding_length INTEGER NOT NULL DEFAULT 0,
                    face_image BLOB,
                    original_image BLOB,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (person_id) REFERENCES people(id) ON DELETE CASCADE
                );
                """
            )
            self._migrateFaceSamplesTable(connection)
            self._migratePeopleTable(connection)

    def _migrateFaceSamplesTable(self, connection: sqlite3.Connection) -> None:
        existingColumns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(face_samples)")
        }

        if "embedding_length" not in existingColumns:
            connection.execute(
                """
                ALTER TABLE face_samples
                ADD COLUMN embedding_length INTEGER NOT NULL DEFAULT 0
                """
            )

        if "face_image" not in existingColumns:
            connection.execute(
                """
                ALTER TABLE face_samples
                ADD COLUMN face_image BLOB
                """
            )

        if "original_image" not in existingColumns:
            connection.execute(
                """
                ALTER TABLE face_samples
                ADD COLUMN original_image BLOB
                """
            )

        connection.execute(
            """
            UPDATE face_samples
            SET embedding_length = LENGTH(embedding) / 4
            WHERE embedding IS NOT NULL
              AND embedding_length <= 0
              AND LENGTH(embedding) % 4 = 0
            """
        )

    def _migratePeopleTable(self, connection: sqlite3.Connection) -> None:
        try:
            connection.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS people_name_unique_nocase
                ON people(name COLLATE NOCASE)
                """
            )
        except sqlite3.IntegrityError:
            # Keep the app usable with older databases that already contain
            # conflicting case-only duplicates. Repository-level checks still
            # prevent creating additional duplicates.
            pass

    def _ensurePersonNameAvailable(
        self,
        connection: sqlite3.Connection,
        name: str,
        excludedPersonId: int | None = None,
    ) -> None:
        if excludedPersonId is None:
            row = connection.execute(
                """
                SELECT id
                FROM people
                WHERE name = ? COLLATE NOCASE
                """,
                (name,),
            ).fetchone()
        else:
            row = connection.execute(
                """
                SELECT id
                FROM people
                WHERE name = ? COLLATE NOCASE
                  AND id != ?
                """,
                (name, excludedPersonId),
            ).fetchone()
        if row is None:
            return

        raise sqlite3.IntegrityError(f'A person named "{name}" already exists.')

    def _insertFaceSamples(
        self,
        connection: sqlite3.Connection,
        personId: int,
        faceSamples: list[NewFaceSample],
    ) -> None:
        connection.executemany(
            """
            INSERT INTO face_samples (
                person_id,
                image_path,
                rect_x,
                rect_y,
                rect_width,
                rect_height,
                embedding,
                embedding_length,
                face_image,
                original_image
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    personId,
                    sample.imagePath,
                    sample.rectX,
                    sample.rectY,
                    sample.rectWidth,
                    sample.rectHeight,
                    sqlite3.Binary(sample.embedding),
                    sample.embeddingLength,
                    sqlite3.Binary(sample.faceImage),
                    sqlite3.Binary(sample.originalImage),
                )
                for sample in faceSamples
            ],
        )

    @staticmethod
    def _personFromRow(row: Iterable[object] | None) -> Person:
        if row is None:
            raise ValueError("Expected a database row, got None.")

        personId, name, notes, createdAt, updatedAt = row
        return Person(
            id=int(personId),
            name=str(name),
            notes=notes if notes is None else str(notes),
            createdAt=str(createdAt),
            updatedAt=str(updatedAt),
        )

    @staticmethod
    def _personSummaryFromRow(row: Iterable[object]) -> PersonSummary:
        personId, name, notes, sampleCount, createdAt, updatedAt = row
        return PersonSummary(
            id=int(personId),
            name=str(name),
            notes=notes if notes is None else str(notes),
            sampleCount=int(sampleCount),
            createdAt=str(createdAt),
            updatedAt=str(updatedAt),
        )

    @staticmethod
    def _faceSampleFromRow(row: Iterable[object]) -> FaceSample:
        (
            sampleId,
            personId,
            imagePath,
            rectX,
            rectY,
            rectWidth,
            rectHeight,
            embedding,
            embeddingLength,
            faceImage,
            hasOriginalImage,
            createdAt,
        ) = row

        return FaceSample(
            id=int(sampleId),
            personId=int(personId),
            imagePath=str(imagePath),
            rectX=int(rectX),
            rectY=int(rectY),
            rectWidth=int(rectWidth),
            rectHeight=int(rectHeight),
            embedding=bytes(embedding) if embedding is not None else None,
            embeddingLength=int(embeddingLength),
            faceImage=bytes(faceImage) if faceImage is not None else None,
            hasOriginalImage=bool(hasOriginalImage),
            createdAt=str(createdAt),
        )
