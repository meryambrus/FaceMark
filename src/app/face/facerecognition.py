from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from insightface.app import FaceAnalysis


type Rectangle = tuple[int, int, int, int]


@dataclass(slots=True)
class DetectedFace:
    boundingBox: Rectangle
    embedding: np.ndarray


@dataclass(slots=True)
class PreparedFaceSample:
    imagePath: str
    boundingBox: Rectangle
    embedding: bytes
    embeddingLength: int
    faceImage: bytes
    originalImage: bytes


class FaceRecognitionService:
    defaultSimilarityThreshold = 0.45

    def __init__(self, modelName: str = "buffalo_l"):
        self.modelName = modelName
        self._faceAnalysis: FaceAnalysis | None = None

    def readImage(self, imagePath: str | Path) -> np.ndarray:
        image = cv2.imread(str(imagePath))
        if image is None:
            raise ValueError(f"Could not read image: {imagePath}")

        return image

    def readImageBytes(self, imagePath: str | Path) -> bytes:
        try:
            return Path(imagePath).read_bytes()
        except OSError as error:
            raise ValueError(f"Could not read image bytes: {imagePath}") from error

    def detectFaces(self, imageBgr: np.ndarray) -> list[DetectedFace]:
        faceAnalysis = self._getFaceAnalysis()
        imageHeight, imageWidth = imageBgr.shape[:2]
        detectedFaces: list[DetectedFace] = []

        for face in faceAnalysis.get(imageBgr):
            boundingBox = self._normalizeBoundingBox(face.bbox, imageWidth, imageHeight)
            embedding = np.asarray(face.embedding, dtype=np.float32).reshape(-1).copy()
            detectedFaces.append(
                DetectedFace(
                    boundingBox=boundingBox,
                    embedding=embedding,
                )
            )

        return detectedFaces

    def prepareFaceSample(
        self,
        imagePath: str | Path,
        selectedRectangle: Rectangle,
        imageBgr: np.ndarray | None = None,
        detectedFaces: list[DetectedFace] | None = None,
    ) -> PreparedFaceSample:
        if imageBgr is None:
            imageBgr = self.readImage(imagePath)

        normalizedRectangle = self.normalizeRectangle(selectedRectangle, imageBgr)
        faceCrop = self.cropImage(imageBgr, normalizedRectangle)
        matchedFace = self.findDetectedFace(normalizedRectangle, detectedFaces or self.detectFaces(imageBgr))

        if matchedFace is None:
            cropDetections = self.detectFaces(faceCrop)
            if cropDetections:
                matchedFace = max(
                    cropDetections,
                    key=lambda face: self.rectangleArea(face.boundingBox),
                )

        if matchedFace is None:
            raise ValueError("No detectable face was found inside the selected rectangle.")

        embeddingBytes, embeddingLength = self.serializeEmbedding(matchedFace.embedding)
        faceImageBytes = self.encodeImage(faceCrop)
        originalImageBytes = self.readImageBytes(imagePath)

        return PreparedFaceSample(
            imagePath=str(Path(imagePath)),
            boundingBox=normalizedRectangle,
            embedding=embeddingBytes,
            embeddingLength=embeddingLength,
            faceImage=faceImageBytes,
            originalImage=originalImageBytes,
        )

    def normalizeRectangle(self, rectangle: Rectangle, imageBgr: np.ndarray) -> Rectangle:
        imageHeight, imageWidth = imageBgr.shape[:2]
        x, y, width, height = rectangle

        x = max(0, min(int(round(x)), imageWidth - 1))
        y = max(0, min(int(round(y)), imageHeight - 1))
        width = max(1, int(round(width)))
        height = max(1, int(round(height)))

        if x + width > imageWidth:
            width = imageWidth - x

        if y + height > imageHeight:
            height = imageHeight - y

        return x, y, width, height

    def cropImage(self, imageBgr: np.ndarray, rectangle: Rectangle) -> np.ndarray:
        x, y, width, height = self.normalizeRectangle(rectangle, imageBgr)
        return imageBgr[y : y + height, x : x + width].copy()

    def findDetectedFace(
        self,
        selectedRectangle: Rectangle,
        detectedFaces: list[DetectedFace],
    ) -> DetectedFace | None:
        bestMatch: DetectedFace | None = None
        bestScore = 0.0

        for detectedFace in detectedFaces:
            overlapScore = self._faceOverlapScore(selectedRectangle, detectedFace.boundingBox)
            if overlapScore > bestScore:
                bestScore = overlapScore
                bestMatch = detectedFace

        if bestScore < 0.5:
            return None

        return bestMatch

    def serializeEmbedding(self, embedding: np.ndarray) -> tuple[bytes, int]:
        normalizedEmbedding = np.asarray(embedding, dtype=np.float32).reshape(-1)
        return normalizedEmbedding.tobytes(), int(normalizedEmbedding.size)

    def deserializeEmbedding(
        self,
        embeddingBytes: bytes | None,
        embeddingLength: int,
    ) -> np.ndarray | None:
        if not embeddingBytes or embeddingLength <= 0:
            return None

        embedding = np.frombuffer(embeddingBytes, dtype=np.float32, count=embeddingLength)
        if embedding.size != embeddingLength:
            return None

        return embedding.copy()

    def normalizeEmbedding(self, embedding: np.ndarray) -> np.ndarray:
        normalizedEmbedding = np.asarray(embedding, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(normalizedEmbedding))
        if norm <= 0.0:
            return normalizedEmbedding.copy()

        return normalizedEmbedding / norm

    def cosineSimilarity(self, firstEmbedding: np.ndarray, secondEmbedding: np.ndarray) -> float:
        normalizedFirst = self.normalizeEmbedding(firstEmbedding)
        normalizedSecond = self.normalizeEmbedding(secondEmbedding)
        return float(np.dot(normalizedFirst, normalizedSecond))

    def findMatchingPeople(
        self,
        imageBgr: np.ndarray,
        personEmbeddingsById: dict[int, list[np.ndarray]],
        similarityThreshold: float | None = None,
        candidateEmbeddingsNormalized: bool = False,
    ) -> set[int]:
        if not personEmbeddingsById:
            return set()

        threshold = self.defaultSimilarityThreshold if similarityThreshold is None else similarityThreshold
        matchedPersonIds: set[int] = set()

        for detectedFace in self.detectFaces(imageBgr):
            faceEmbedding = self.normalizeEmbedding(detectedFace.embedding)
            for personId, candidateEmbeddings in personEmbeddingsById.items():
                if personId in matchedPersonIds:
                    continue

                for candidateEmbedding in candidateEmbeddings:
                    comparisonEmbedding = (
                        candidateEmbedding
                        if candidateEmbeddingsNormalized
                        else self.normalizeEmbedding(candidateEmbedding)
                    )
                    similarity = float(np.dot(faceEmbedding, comparisonEmbedding))
                    if similarity >= threshold:
                        matchedPersonIds.add(personId)
                        break

        return matchedPersonIds

    def encodeImage(self, imageBgr: np.ndarray) -> bytes:
        success, encodedImage = cv2.imencode(".png", imageBgr)
        if not success:
            raise ValueError("Could not encode the selected face image.")

        return encodedImage.tobytes()

    @staticmethod
    def rectangleArea(rectangle: Rectangle) -> int:
        _, _, width, height = rectangle
        return width * height

    def _getFaceAnalysis(self) -> FaceAnalysis:
        if self._faceAnalysis is None:
            providers = ort.get_available_providers()
            deviceId = 0 if "CUDAExecutionProvider" in providers else -1

            faceAnalysis = FaceAnalysis(name=self.modelName)
            faceAnalysis.prepare(ctx_id=deviceId)
            self._faceAnalysis = faceAnalysis

        return self._faceAnalysis

    def _faceOverlapScore(self, selectedRectangle: Rectangle, detectedRectangle: Rectangle) -> float:
        intersectionArea = self._intersectionArea(selectedRectangle, detectedRectangle)
        detectedArea = max(1, self.rectangleArea(detectedRectangle))
        return intersectionArea / detectedArea

    @staticmethod
    def _intersectionArea(firstRectangle: Rectangle, secondRectangle: Rectangle) -> int:
        firstX, firstY, firstWidth, firstHeight = firstRectangle
        secondX, secondY, secondWidth, secondHeight = secondRectangle

        intersectionLeft = max(firstX, secondX)
        intersectionTop = max(firstY, secondY)
        intersectionRight = min(firstX + firstWidth, secondX + secondWidth)
        intersectionBottom = min(firstY + firstHeight, secondY + secondHeight)

        if intersectionRight <= intersectionLeft or intersectionBottom <= intersectionTop:
            return 0

        return (intersectionRight - intersectionLeft) * (intersectionBottom - intersectionTop)

    @staticmethod
    def _normalizeBoundingBox(
        boundingBox: np.ndarray,
        imageWidth: int,
        imageHeight: int,
    ) -> Rectangle:
        left = max(0, min(int(np.floor(boundingBox[0])), imageWidth - 1))
        top = max(0, min(int(np.floor(boundingBox[1])), imageHeight - 1))
        right = max(left + 1, min(int(np.ceil(boundingBox[2])), imageWidth))
        bottom = max(top + 1, min(int(np.ceil(boundingBox[3])), imageHeight))

        return left, top, right - left, bottom - top
