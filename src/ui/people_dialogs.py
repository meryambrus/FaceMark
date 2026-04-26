from __future__ import annotations

from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import wx

from src.app.face.facerecognition import DetectedFace, Rectangle
from src.ui.theme import SURFACE_BG_SUBTLE, TEXT_MUTED, styleButton, styleDialog, styleFrame, stylePanel, styleText


class FaceSelectionDialog(wx.Dialog):
    skipResultId = wx.ID_NO

    def __init__(
        self,
        parent: wx.Window,
        imagePath: str,
        imageBgr,
        detectedFaces: list[DetectedFace],
    ):
        super().__init__(parent, title="Select Face", size=(980, 760))
        styleDialog(self)
        self.facePanel = FaceSelectionPanel(
            self,
            imageBgr=imageBgr,
            detectedFaces=detectedFaces,
            onSelectionChanged=self._onSelectionChanged,
        )
        self.okButton = wx.Button(self, wx.ID_OK, "Use Selected Face")
        self.skipButton = wx.Button(self, self.skipResultId, "Skip Image")
        self.cancelButton = wx.Button(self, wx.ID_CANCEL, "Cancel")

        imageLabel = wx.StaticText(self, label=f"Image: {Path(imagePath).name}")
        instructionsText = (
            "Click a detected face box or drag a new rectangle around the target face. "
            "Only one face sample is saved from each image."
        )
        if len(detectedFaces) > 1:
            instructionsText += " Multiple faces were detected, so select the correct face before continuing."
        instructions = wx.StaticText(
            self,
            label=instructionsText,
        )
        detectionLabel = wx.StaticText(self, label=f"Detected faces: {len(detectedFaces)}")
        styleText(imageLabel, "section_title")
        styleText(instructions, "muted")
        styleText(detectionLabel, "eyebrow")
        instructions.Wrap(860)
        styleButton(self.okButton, "primary")
        styleButton(self.skipButton, "secondary")
        styleButton(self.cancelButton, "subtle")

        rootSizer = wx.BoxSizer(wx.VERTICAL)
        rootSizer.Add(imageLabel, 0, wx.TOP | wx.LEFT | wx.RIGHT, 12)
        rootSizer.Add(instructions, 0, wx.TOP | wx.LEFT | wx.RIGHT, 8)
        rootSizer.Add(detectionLabel, 0, wx.TOP | wx.LEFT | wx.RIGHT | wx.BOTTOM, 8)
        rootSizer.Add(self.facePanel, 1, wx.LEFT | wx.RIGHT | wx.BOTTOM | wx.EXPAND, 12)

        buttonSizer = wx.BoxSizer(wx.HORIZONTAL)
        buttonSizer.AddStretchSpacer()
        buttonSizer.Add(self.okButton, 0, wx.RIGHT, 8)
        buttonSizer.Add(self.skipButton, 0, wx.RIGHT, 8)
        buttonSizer.Add(self.cancelButton, 0)
        rootSizer.Add(buttonSizer, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM | wx.EXPAND, 12)

        self.okButton.Bind(wx.EVT_BUTTON, self._onUseSelection)
        self.skipButton.Bind(wx.EVT_BUTTON, self._onSkipImage)
        self.cancelButton.Bind(wx.EVT_BUTTON, self._onCancel)

        self.SetSizer(rootSizer)
        self._onSelectionChanged(self.facePanel.getSelectedRectangle())

    def getSelectedRectangle(self) -> Rectangle | None:
        return self.facePanel.getSelectedRectangle()

    def _onSelectionChanged(self, selectedRectangle: Rectangle | None) -> None:
        self.okButton.Enable(selectedRectangle is not None)

    def _onUseSelection(self, event: wx.CommandEvent) -> None:
        if self.facePanel.getSelectedRectangle() is None:
            wx.MessageBox("Select a face rectangle before continuing.", "Face Selection")
            return

        self.EndModal(wx.ID_OK)

    def _onSkipImage(self, event: wx.CommandEvent) -> None:
        self.EndModal(self.skipResultId)

    def _onCancel(self, event: wx.CommandEvent) -> None:
        self.EndModal(wx.ID_CANCEL)


class OriginalImageFrame(wx.Frame):
    def __init__(
        self,
        parent: wx.Window,
        imagePath: str,
        imageBytes: bytes | None,
    ):
        super().__init__(
            parent,
            title="Original Image",
            size=(1120, 860),
            style=wx.DEFAULT_FRAME_STYLE,
        )
        styleFrame(self)

        contentPanel = wx.Panel(self)
        stylePanel(contentPanel)

        self.previewPanel = OriginalImagePanel(
            contentPanel,
            imagePath=imagePath,
            imageBytes=imageBytes,
        )
        imageLabel = wx.StaticText(contentPanel, label=f"Image: {Path(imagePath).name}")
        pathLabel = wx.StaticText(contentPanel, label=imagePath)
        styleText(imageLabel, "section_title")
        styleText(pathLabel, "muted")
        pathLabel.Wrap(1030)

        closeButton = wx.Button(contentPanel, wx.ID_CLOSE, "Close")
        styleButton(closeButton, "primary")
        closeButton.Bind(wx.EVT_BUTTON, self._onClose)

        rootSizer = wx.BoxSizer(wx.VERTICAL)
        rootSizer.Add(imageLabel, 0, wx.TOP | wx.LEFT | wx.RIGHT, 12)
        rootSizer.Add(pathLabel, 0, wx.TOP | wx.LEFT | wx.RIGHT | wx.BOTTOM | wx.EXPAND, 8)
        rootSizer.Add(self.previewPanel, 1, wx.LEFT | wx.RIGHT | wx.BOTTOM | wx.EXPAND, 12)

        buttonSizer = wx.BoxSizer(wx.HORIZONTAL)
        buttonSizer.AddStretchSpacer()
        buttonSizer.Add(closeButton, 0)
        rootSizer.Add(buttonSizer, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM | wx.EXPAND, 12)
        contentPanel.SetSizer(rootSizer)

        frameSizer = wx.BoxSizer(wx.VERTICAL)
        frameSizer.Add(contentPanel, 1, wx.EXPAND)
        self.SetSizer(frameSizer)
        self.CentreOnParent()

    def _onClose(self, event: wx.CommandEvent) -> None:
        self.Close()


class OriginalImagePanel(wx.Panel):
    def __init__(
        self,
        parent: wx.Window,
        imagePath: str,
        imageBytes: bytes | None,
    ):
        super().__init__(parent)
        stylePanel(self, background=SURFACE_BG_SUBTLE)
        self.SetMinSize((900, 620))
        self.SetBackgroundStyle(wx.BG_STYLE_PAINT)
        self.displayBitmap: wx.Bitmap | None = None
        self.displayOffsetX = 0
        self.displayOffsetY = 0

        self.imageBgr = self._loadImage(imagePath, imageBytes)
        self.Bind(wx.EVT_PAINT, self._onPaint)
        self.Bind(wx.EVT_SIZE, self._onSize)
        self._rebuildDisplayBitmap()

    def _loadImage(
        self,
        imagePath: str,
        imageBytes: bytes | None,
    ):
        if imageBytes:
            imageBuffer = np.frombuffer(imageBytes, dtype=np.uint8)
            imageBgr = cv2.imdecode(imageBuffer, cv2.IMREAD_COLOR)
            if imageBgr is not None:
                return imageBgr

        return cv2.imread(imagePath)

    def _onSize(self, event: wx.SizeEvent) -> None:
        self._rebuildDisplayBitmap()
        self.Refresh()
        event.Skip()

    def _rebuildDisplayBitmap(self) -> None:
        clientWidth, clientHeight = self.GetClientSize()
        if clientWidth <= 0 or clientHeight <= 0 or self.imageBgr is None:
            self.displayBitmap = None
            return

        imageHeight, imageWidth = self.imageBgr.shape[:2]
        displayScale = min(clientWidth / imageWidth, clientHeight / imageHeight, 1.0)
        displayWidth = max(1, int(imageWidth * displayScale))
        displayHeight = max(1, int(imageHeight * displayScale))
        self.displayOffsetX = (clientWidth - displayWidth) // 2
        self.displayOffsetY = (clientHeight - displayHeight) // 2

        rgbImage = cv2.cvtColor(self.imageBgr, cv2.COLOR_BGR2RGB)
        wxImage = wx.Image(imageWidth, imageHeight)
        wxImage.SetData(rgbImage.tobytes())
        if displayWidth != imageWidth or displayHeight != imageHeight:
            wxImage = wxImage.Scale(displayWidth, displayHeight, wx.IMAGE_QUALITY_HIGH)

        self.displayBitmap = wx.Bitmap(wxImage)

    def _onPaint(self, event: wx.PaintEvent) -> None:
        paintDc = wx.AutoBufferedPaintDC(self)
        paintDc.SetBackground(wx.Brush(self.GetBackgroundColour()))
        paintDc.Clear()

        if self.displayBitmap is None:
            paintDc.SetTextForeground(TEXT_MUTED)
            paintDc.DrawLabel(
                "Original image preview unavailable",
                self.GetClientRect(),
                alignment=wx.ALIGN_CENTER_HORIZONTAL | wx.ALIGN_CENTER_VERTICAL,
            )
            return

        paintDc.DrawBitmap(self.displayBitmap, self.displayOffsetX, self.displayOffsetY, True)


class FaceSelectionPanel(wx.Panel):
    def __init__(
        self,
        parent: wx.Window,
        imageBgr,
        detectedFaces: list[DetectedFace],
        onSelectionChanged: Callable[[Rectangle | None], None] | None = None,
    ):
        super().__init__(parent)
        stylePanel(self, background=SURFACE_BG_SUBTLE)
        self.SetMinSize((900, 580))
        self.SetBackgroundStyle(wx.BG_STYLE_PAINT)

        self.imageBgr = imageBgr
        self.imageHeight, self.imageWidth = imageBgr.shape[:2]
        self.detectedRectangles = [face.boundingBox for face in detectedFaces]
        self.onSelectionChanged = onSelectionChanged
        self.selectedRectangle: Rectangle | None = (
            self.detectedRectangles[0]
            if len(self.detectedRectangles) == 1
            else None
        )
        self.dragStartPoint: tuple[int, int] | None = None
        self.dragCurrentPoint: tuple[int, int] | None = None
        self.displayBitmap: wx.Bitmap | None = None
        self.displayScale = 1.0
        self.displayOffsetX = 0
        self.displayOffsetY = 0
        self.displayWidth = 0
        self.displayHeight = 0

        self.Bind(wx.EVT_PAINT, self._onPaint)
        self.Bind(wx.EVT_SIZE, self._onSize)
        self.Bind(wx.EVT_LEFT_DOWN, self._onLeftDown)
        self.Bind(wx.EVT_LEFT_UP, self._onLeftUp)
        self.Bind(wx.EVT_MOTION, self._onMouseMove)

        self._rebuildDisplayBitmap()

    def getSelectedRectangle(self) -> Rectangle | None:
        return self.selectedRectangle

    def _onSize(self, event: wx.SizeEvent) -> None:
        self._rebuildDisplayBitmap()
        self.Refresh()
        event.Skip()

    def _onPaint(self, event: wx.PaintEvent) -> None:
        paintDc = wx.AutoBufferedPaintDC(self)
        paintDc.SetBackground(wx.Brush(self.GetBackgroundColour()))
        paintDc.Clear()

        if self.displayBitmap is None:
            return

        paintDc.DrawBitmap(self.displayBitmap, self.displayOffsetX, self.displayOffsetY)
        self._drawDetectedRectangles(paintDc)
        self._drawSelectedRectangle(paintDc)
        self._drawDragRectangle(paintDc)

    def _onLeftDown(self, event: wx.MouseEvent) -> None:
        panelPoint = event.GetPosition()
        clickedRectangle = self._getRectangleAtPoint(panelPoint.x, panelPoint.y)

        if clickedRectangle is not None:
            self._setSelectedRectangle(clickedRectangle)
            self.Refresh()
            return

        imagePoint = self._panelPointToImagePoint(panelPoint.x, panelPoint.y)
        if imagePoint is None:
            return

        self.dragStartPoint = imagePoint
        self.dragCurrentPoint = imagePoint
        if not self.HasCapture():
            self.CaptureMouse()
        self.Refresh()

    def _onLeftUp(self, event: wx.MouseEvent) -> None:
        if self.dragStartPoint is None or self.dragCurrentPoint is None:
            return

        if self.HasCapture():
            self.ReleaseMouse()

        candidateRectangle = self._pointsToRectangle(self.dragStartPoint, self.dragCurrentPoint)
        self.dragStartPoint = None
        self.dragCurrentPoint = None

        if candidateRectangle[2] >= 10 and candidateRectangle[3] >= 10:
            self._setSelectedRectangle(candidateRectangle)

        self.Refresh()

    def _onMouseMove(self, event: wx.MouseEvent) -> None:
        if self.dragStartPoint is None or not event.Dragging() or not event.LeftIsDown():
            return

        imagePoint = self._panelPointToImagePoint(
            event.GetPosition().x,
            event.GetPosition().y,
            clampToImage=True,
        )
        if imagePoint is None:
            return

        self.dragCurrentPoint = imagePoint
        self.Refresh()

    def _rebuildDisplayBitmap(self) -> None:
        clientWidth, clientHeight = self.GetClientSize()
        if clientWidth <= 0 or clientHeight <= 0:
            return

        self.displayScale = min(clientWidth / self.imageWidth, clientHeight / self.imageHeight)
        self.displayWidth = max(1, int(self.imageWidth * self.displayScale))
        self.displayHeight = max(1, int(self.imageHeight * self.displayScale))
        self.displayOffsetX = (clientWidth - self.displayWidth) // 2
        self.displayOffsetY = (clientHeight - self.displayHeight) // 2

        rgbImage = cv2.cvtColor(self.imageBgr, cv2.COLOR_BGR2RGB)
        wxImage = wx.Image(self.imageWidth, self.imageHeight)
        wxImage.SetData(rgbImage.tobytes())
        if self.displayWidth != self.imageWidth or self.displayHeight != self.imageHeight:
            wxImage = wxImage.Scale(self.displayWidth, self.displayHeight, wx.IMAGE_QUALITY_HIGH)

        self.displayBitmap = wx.Bitmap(wxImage)

    def _drawDetectedRectangles(self, paintDc: wx.DC) -> None:
        paintDc.SetBrush(wx.TRANSPARENT_BRUSH)
        paintDc.SetPen(wx.Pen(wx.Colour(0, 180, 0), 2))

        for rectangle in self.detectedRectangles:
            displayRectangle = self._imageRectangleToDisplayRectangle(rectangle)
            paintDc.DrawRectangle(*displayRectangle)

    def _drawSelectedRectangle(self, paintDc: wx.DC) -> None:
        if self.selectedRectangle is None:
            return

        paintDc.SetBrush(wx.TRANSPARENT_BRUSH)
        paintDc.SetPen(wx.Pen(wx.Colour(255, 210, 0), 3))
        displayRectangle = self._imageRectangleToDisplayRectangle(self.selectedRectangle)
        paintDc.DrawRectangle(*displayRectangle)

    def _drawDragRectangle(self, paintDc: wx.DC) -> None:
        if self.dragStartPoint is None or self.dragCurrentPoint is None:
            return

        paintDc.SetBrush(wx.TRANSPARENT_BRUSH)
        paintDc.SetPen(wx.Pen(wx.Colour(220, 60, 60), 2, style=wx.PENSTYLE_SHORT_DASH))
        dragRectangle = self._pointsToRectangle(self.dragStartPoint, self.dragCurrentPoint)
        displayRectangle = self._imageRectangleToDisplayRectangle(dragRectangle)
        paintDc.DrawRectangle(*displayRectangle)

    def _getRectangleAtPoint(self, panelX: int, panelY: int) -> Rectangle | None:
        for rectangle in self.detectedRectangles:
            displayX, displayY, displayWidth, displayHeight = self._imageRectangleToDisplayRectangle(rectangle)
            if displayX <= panelX <= displayX + displayWidth and displayY <= panelY <= displayY + displayHeight:
                return rectangle

        return None

    def _panelPointToImagePoint(
        self,
        panelX: int,
        panelY: int,
        clampToImage: bool = False,
    ) -> tuple[int, int] | None:
        if self.displayBitmap is None:
            return None

        insideImage = (
            self.displayOffsetX <= panelX <= self.displayOffsetX + self.displayWidth
            and self.displayOffsetY <= panelY <= self.displayOffsetY + self.displayHeight
        )
        if not insideImage and not clampToImage:
            return None

        clampedPanelX = min(max(panelX, self.displayOffsetX), self.displayOffsetX + self.displayWidth)
        clampedPanelY = min(max(panelY, self.displayOffsetY), self.displayOffsetY + self.displayHeight)
        imageX = int((clampedPanelX - self.displayOffsetX) / self.displayScale)
        imageY = int((clampedPanelY - self.displayOffsetY) / self.displayScale)

        imageX = min(max(imageX, 0), self.imageWidth - 1)
        imageY = min(max(imageY, 0), self.imageHeight - 1)
        return imageX, imageY

    def _imageRectangleToDisplayRectangle(self, rectangle: Rectangle) -> tuple[int, int, int, int]:
        x, y, width, height = rectangle
        displayX = self.displayOffsetX + int(x * self.displayScale)
        displayY = self.displayOffsetY + int(y * self.displayScale)
        displayWidth = max(1, int(width * self.displayScale))
        displayHeight = max(1, int(height * self.displayScale))
        return displayX, displayY, displayWidth, displayHeight

    def _pointsToRectangle(
        self,
        firstPoint: tuple[int, int],
        secondPoint: tuple[int, int],
    ) -> Rectangle:
        left = min(firstPoint[0], secondPoint[0])
        top = min(firstPoint[1], secondPoint[1])
        right = max(firstPoint[0], secondPoint[0])
        bottom = max(firstPoint[1], secondPoint[1])
        return left, top, max(1, right - left), max(1, bottom - top)

    def _setSelectedRectangle(self, rectangle: Rectangle | None) -> None:
        self.selectedRectangle = rectangle
        if self.onSelectionChanged is not None:
            self.onSelectionChanged(rectangle)
