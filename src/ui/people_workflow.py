from __future__ import annotations

import wx

from src.app.face.facerecognition import FaceRecognitionService
from src.app.imagefiles import SUPPORTED_IMAGE_WILDCARD
from src.data.store import NewFaceSample
from src.ui.people_dialogs import FaceSelectionDialog


def promptForImagePaths(
    parent: wx.Window,
    message: str = "Choose one or more images",
) -> list[str] | None:
    dialog = wx.FileDialog(
        parent,
        message=message,
        wildcard=SUPPORTED_IMAGE_WILDCARD,
        style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST | wx.FD_MULTIPLE,
    )
    try:
        if dialog.ShowModal() != wx.ID_OK:
            return None

        imagePaths = dialog.GetPaths()
        if not imagePaths:
            return None

        return imagePaths
    finally:
        dialog.Destroy()


def collectFaceSamples(
    parent: wx.Window,
    faceRecognitionService: FaceRecognitionService,
    imagePaths: list[str],
) -> list[NewFaceSample] | None:
    faceSamples: list[NewFaceSample] = []

    for imagePath in imagePaths:
        faceSample, cancelled = _collectFaceSampleForImage(parent, faceRecognitionService, imagePath)
        if cancelled:
            return None

        if faceSample is not None:
            faceSamples.append(faceSample)

    return faceSamples


def _collectFaceSampleForImage(
    parent: wx.Window,
    faceRecognitionService: FaceRecognitionService,
    imagePath: str,
) -> tuple[NewFaceSample | None, bool]:
    try:
        imageBgr = faceRecognitionService.readImage(imagePath)
        detectedFaces = faceRecognitionService.detectFaces(imageBgr)
    except Exception as error:
        wx.MessageBox(str(error), "People")
        return None, False

    while True:
        dialog = FaceSelectionDialog(parent, imagePath, imageBgr, detectedFaces)
        try:
            dialogResult = dialog.ShowModal()
            selectedRectangle = dialog.getSelectedRectangle()
        finally:
            dialog.Destroy()

        if dialogResult == wx.ID_CANCEL:
            return None, True

        if dialogResult == FaceSelectionDialog.skipResultId:
            return None, False

        if selectedRectangle is None:
            wx.MessageBox("Select a face rectangle before continuing.", "People")
            continue

        try:
            preparedSample = faceRecognitionService.prepareFaceSample(
                imagePath=imagePath,
                selectedRectangle=selectedRectangle,
                imageBgr=imageBgr,
                detectedFaces=detectedFaces,
            )
        except Exception as error:
            wx.MessageBox(
                f"{error}\nSelect a different rectangle or skip the image.",
                "People",
            )
            continue

        rectX, rectY, rectWidth, rectHeight = preparedSample.boundingBox
        return (
            NewFaceSample(
                imagePath=preparedSample.imagePath,
                rectX=rectX,
                rectY=rectY,
                rectWidth=rectWidth,
                rectHeight=rectHeight,
                embedding=preparedSample.embedding,
                embeddingLength=preparedSample.embeddingLength,
                faceImage=preparedSample.faceImage,
                originalImage=preparedSample.originalImage,
            ),
            False,
        )
