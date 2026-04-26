from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
import os
from pathlib import Path
import threading
import time
from typing import Callable

import numpy as np
import wx
from wx.lib.wordwrap import wordwrap

from src.app.face.facerecognition import FaceRecognitionService
from src.app.imagefiles import listSupportedImagesInFolder
from src.data.store import PersonRepository, PersonSummary
from src.ui.image_utils import (
    PreparedBitmapData,
    bitmapFromPreparedBitmapSized,
    preparedBitmapFromImagePath,
)
from src.ui.theme import (
    ACCENT,
    BORDER,
    SUCCESS,
    SURFACE_BG_ALT,
    SURFACE_BG_SUBTLE,
    TEXT_MUTED,
    TEXT_PRIMARY,
    CardPanel,
    styleButton,
    stylePanel,
    styleScrolledWindow,
    styleText,
)


@dataclass(slots=True)
class FolderImageEntry:
    imagePath: Path
    matchedPersonIds: set[int] = field(default_factory=set)
    matchedPersonNames: tuple[str, ...] = ()


@dataclass(slots=True)
class FolderSearchResult:
    filteredPersonNames: list[str]
    matchingImagePaths: list[Path]
    matchedPersonIdsByPath: dict[Path, set[int]]
    matchedPersonNamesByPath: dict[Path, tuple[str, ...]]
    failureCount: int


def prepareThumbnailData(
    imagePath: Path,
    maxWidth: int,
    maxHeight: int,
    cancelEvent: threading.Event | None = None,
) -> tuple[Path, PreparedBitmapData | None]:
    if cancelEvent is not None and cancelEvent.is_set():
        return imagePath, None

    preparedBitmap = preparedBitmapFromImagePath(
        imagePath,
        maxWidth=maxWidth,
        maxHeight=maxHeight,
    )
    return imagePath, preparedBitmap


def buildWorkspaceNavigationSizer(
    parent: wx.Window,
    activeView: str,
    onRequestShowPeople: Callable[[], None] | None,
    onRequestShowFolder: Callable[[], None] | None,
) -> wx.BoxSizer:
    peopleNavButton = wx.Button(parent, label="People")
    folderNavButton = wx.Button(parent, label="Folder Search")
    styleButton(peopleNavButton, "nav_active" if activeView == "people" else "nav")
    styleButton(folderNavButton, "nav_active" if activeView == "folder" else "nav")
    if onRequestShowPeople is not None:
        peopleNavButton.Bind(wx.EVT_BUTTON, lambda event: onRequestShowPeople())
    else:
        peopleNavButton.Enable(False)

    if onRequestShowFolder is not None:
        folderNavButton.Bind(wx.EVT_BUTTON, lambda event: onRequestShowFolder())
    else:
        folderNavButton.Enable(False)

    navSizer = wx.BoxSizer(wx.HORIZONTAL)
    navSizer.Add(peopleNavButton, 0, wx.RIGHT, 8)
    navSizer.Add(folderNavButton, 0)
    return navSizer


class ThumbnailSizeSlider(wx.Panel):
    def __init__(
        self,
        parent: wx.Window,
        values: tuple[int, ...],
        initialValue: int,
        onValueChanged: Callable[[int], None],
    ):
        super().__init__(parent)
        if not values:
            raise ValueError("Thumbnail size slider needs at least one value.")

        self.values = tuple(values)
        self.value = initialValue if initialValue in self.values else self.values[0]
        self.onValueChanged = onValueChanged
        self._isDragging = False

        self.SetMinSize((188, 28))
        self.SetMaxSize((188, 28))
        self.SetCursor(wx.Cursor(wx.CURSOR_HAND))
        self.SetToolTip("Thumbnail Size")
        self.SetBackgroundStyle(wx.BG_STYLE_PAINT)
        self.SetDoubleBuffered(True)
        self.SetBackgroundColour(self.GetParent().GetBackgroundColour())

        self.Bind(wx.EVT_PAINT, self._onPaint)
        self.Bind(wx.EVT_LEFT_DOWN, self._onLeftDown)
        self.Bind(wx.EVT_LEFT_UP, self._onLeftUp)
        self.Bind(wx.EVT_MOTION, self._onMouseMove)
        self.Bind(wx.EVT_LEAVE_WINDOW, self._onMouseLeave)

    def _onPaint(self, event: wx.PaintEvent) -> None:
        paintDc = wx.AutoBufferedPaintDC(self)
        paintDc.SetBackground(wx.Brush(self.GetBackgroundColour()))
        paintDc.Clear()

        clientRect = self.GetClientRect()
        if clientRect.width <= 0 or clientRect.height <= 0:
            return

        centerY = clientRect.height // 2
        smallIconRect = wx.Rect(2, centerY - 6, 12, 12)
        largeIconRect = wx.Rect(clientRect.width - 18, centerY - 8, 16, 16)
        self._drawGridIcon(paintDc, smallIconRect, cellSize=2, gap=2)
        self._drawGridIcon(paintDc, largeIconRect, cellSize=3, gap=2)

        trackStartX = smallIconRect.GetRight() + 12
        trackEndX = largeIconRect.x - 10
        handleX = self._handleX(trackStartX, trackEndX)

        paintDc.SetPen(wx.Pen(BORDER, 2))
        paintDc.DrawLine(trackStartX, centerY, trackEndX, centerY)
        paintDc.SetPen(wx.Pen(wx.Colour(118, 132, 154), 2))
        paintDc.DrawLine(trackStartX, centerY, handleX, centerY)

        paintDc.SetPen(wx.Pen(BORDER, 1))
        paintDc.SetBrush(wx.Brush(TEXT_PRIMARY))
        paintDc.DrawCircle(handleX, centerY, 6)

    def _drawGridIcon(
        self,
        paintDc: wx.DC,
        iconRect: wx.Rect,
        cellSize: int,
        gap: int,
    ) -> None:
        paintDc.SetPen(wx.Pen(TEXT_MUTED, 1))
        paintDc.SetBrush(wx.Brush(TEXT_MUTED))
        iconWidth = cellSize * 2 + gap
        iconHeight = cellSize * 2 + gap
        originX = iconRect.x + max(0, (iconRect.width - iconWidth) // 2)
        originY = iconRect.y + max(0, (iconRect.height - iconHeight) // 2)
        for rowIndex in range(2):
            for columnIndex in range(2):
                paintDc.DrawRectangle(
                    originX + columnIndex * (cellSize + gap),
                    originY + rowIndex * (cellSize + gap),
                    cellSize,
                    cellSize,
                )

    def _onLeftDown(self, event: wx.MouseEvent) -> None:
        self._isDragging = True
        self._updateValueFromX(event.GetX())
        if not self.HasCapture():
            self.CaptureMouse()

    def _onLeftUp(self, event: wx.MouseEvent) -> None:
        if self.HasCapture():
            self.ReleaseMouse()
        self._isDragging = False
        self._updateValueFromX(event.GetX())

    def _onMouseMove(self, event: wx.MouseEvent) -> None:
        if not self._isDragging or not event.LeftIsDown():
            return

        self._updateValueFromX(event.GetX())

    def _onMouseLeave(self, event: wx.MouseEvent) -> None:
        if self._isDragging or not self.HasCapture():
            return

        self._isDragging = False

    def _updateValueFromX(self, x: int) -> None:
        trackStartX, trackEndX = self._trackBounds()
        trackWidth = max(1, trackEndX - trackStartX)
        clampedX = min(max(x, trackStartX), trackEndX)
        position = (clampedX - trackStartX) / trackWidth
        targetIndex = round(position * (len(self.values) - 1))
        newValue = self.values[targetIndex]
        if newValue == self.value:
            self.Refresh()
            return

        self.value = newValue
        self.Refresh()
        self.onValueChanged(newValue)

    def _trackBounds(self) -> tuple[int, int]:
        clientRect = self.GetClientRect()
        return 26, max(27, clientRect.width - 28)

    def _handleX(self, trackStartX: int, trackEndX: int) -> int:
        if len(self.values) == 1:
            return trackStartX

        valueIndex = self.values.index(self.value)
        ratio = valueIndex / (len(self.values) - 1)
        return trackStartX + int((trackEndX - trackStartX) * ratio)


class FolderImageGallery(wx.ScrolledWindow):
    defaultThumbnailWidth = 200
    defaultThumbnailHeight = 160
    cardRadius = 16
    cardGap = 16
    outerPadding = 12
    cardPadding = 16
    cardTextHeight = 132

    def __init__(
        self,
        parent: wx.Window,
        thumbnailProvider: Callable[[Path], wx.Bitmap | None],
    ):
        super().__init__(parent, style=wx.VSCROLL | wx.FULL_REPAINT_ON_RESIZE)
        self.thumbnailProvider = thumbnailProvider
        self.entries: list[FolderImageEntry] = []
        self.emptyMessage = "No images to display."
        self.thumbnailWidth = self.defaultThumbnailWidth
        self.thumbnailHeight = self.defaultThumbnailHeight
        self._updateCardMetrics()
        self.columnCount = 1
        self._tooltipImagePath: Path | None = None

        styleScrolledWindow(self)
        self.SetBackgroundStyle(wx.BG_STYLE_PAINT)
        self.SetDoubleBuffered(True)
        self.SetScrollRate(12, 12)

        baseFont = self.GetFont()
        self.bodyFont = wx.Font(baseFont)
        self.matchFont = wx.Font(baseFont)
        self.matchFont.SetWeight(wx.FONTWEIGHT_BOLD)
        self.placeholderFont = wx.Font(baseFont)

        self.Bind(wx.EVT_PAINT, self._onPaint)
        self.Bind(wx.EVT_SIZE, self._onSize)
        self.Bind(wx.EVT_MOTION, self._onMouseMove)
        self.Bind(wx.EVT_LEAVE_WINDOW, self._onMouseLeave)

    def setThumbnailDimensions(self, thumbnailWidth: int, thumbnailHeight: int) -> None:
        self.thumbnailWidth = thumbnailWidth
        self.thumbnailHeight = thumbnailHeight
        self._updateCardMetrics()
        self._updateVirtualSize()
        self.Refresh()

    def _updateCardMetrics(self) -> None:
        self.cardWidth = self.thumbnailWidth + self.cardPadding * 2
        self.cardHeight = self.thumbnailHeight + self.cardTextHeight
        self.rowStride = self.cardHeight + self.cardGap

    def setEntries(
        self,
        entries: list[FolderImageEntry],
        emptyMessage: str,
        resetScroll: bool = False,
    ) -> None:
        self.entries = list(entries)
        self.emptyMessage = emptyMessage
        self._updateVirtualSize()
        if resetScroll:
            self.Scroll(0, 0)
        self.Refresh()

    def _onPaint(self, event: wx.PaintEvent) -> None:
        paintDc = wx.AutoBufferedPaintDC(self)
        self.DoPrepareDC(paintDc)
        paintDc.SetBackground(wx.Brush(self.GetBackgroundColour()))
        paintDc.Clear()

        viewportX, viewportY = self.CalcUnscrolledPosition(0, 0)
        viewportRect = wx.Rect(
            viewportX,
            viewportY,
            self.GetClientSize().width,
            self.GetClientSize().height,
        )

        if not self.entries:
            self._drawEmptyState(paintDc, viewportRect)
            return

        startIndex, endIndex = self._visibleIndexRange(viewportRect)
        for imageIndex in range(startIndex, endIndex):
            imageEntry = self.entries[imageIndex]
            cardRect = self._cardRectForIndex(imageIndex)
            self._drawImageCard(paintDc, imageEntry, cardRect)

    def _onSize(self, event: wx.SizeEvent) -> None:
        self._updateVirtualSize()
        self.Refresh()
        event.Skip()

    def _onMouseMove(self, event: wx.MouseEvent) -> None:
        imageEntry = self._entryAtPosition(event.GetPosition())
        imagePath = imageEntry.imagePath if imageEntry is not None else None
        if imagePath == self._tooltipImagePath:
            event.Skip()
            return

        self._tooltipImagePath = imagePath
        self.SetToolTip("" if imagePath is None else str(imagePath))
        event.Skip()

    def _onMouseLeave(self, event: wx.MouseEvent) -> None:
        self._tooltipImagePath = None
        self.UnsetToolTip()
        event.Skip()

    def _updateVirtualSize(self) -> None:
        clientWidth = max(1, self.GetClientSize().width)
        availableWidth = max(1, clientWidth - (self.outerPadding * 2))
        self.columnCount = max(
            1,
            (availableWidth + self.cardGap) // (self.cardWidth + self.cardGap),
        )

        rowCount = max(1, (len(self.entries) + self.columnCount - 1) // self.columnCount)
        contentWidth = (
            self.outerPadding * 2
            + (self.columnCount * self.cardWidth)
            + max(0, self.columnCount - 1) * self.cardGap
        )
        contentHeight = (
            self.outerPadding * 2
            + (rowCount * self.cardHeight)
            + max(0, rowCount - 1) * self.cardGap
        )
        self.SetVirtualSize((max(clientWidth, contentWidth), max(self.GetClientSize().height, contentHeight)))

    def _drawEmptyState(self, paintDc: wx.DC, viewportRect: wx.Rect) -> None:
        paintDc.SetFont(self.placeholderFont)
        paintDc.SetTextForeground(TEXT_MUTED)
        paintDc.DrawLabel(
            self.emptyMessage,
            viewportRect,
            alignment=wx.ALIGN_CENTER_HORIZONTAL | wx.ALIGN_CENTER_VERTICAL,
        )

    def _visibleIndexRange(self, viewportRect: wx.Rect) -> tuple[int, int]:
        startRow = max(0, (viewportRect.y - self.outerPadding) // self.rowStride)
        endRow = max(0, (viewportRect.y + viewportRect.height - self.outerPadding) // self.rowStride + 1)
        startIndex = startRow * self.columnCount
        endIndex = min(len(self.entries), (endRow + 1) * self.columnCount)
        return startIndex, endIndex

    def _cardRectForIndex(self, imageIndex: int) -> wx.Rect:
        rowIndex, columnIndex = divmod(imageIndex, self.columnCount)
        x = self.outerPadding + columnIndex * (self.cardWidth + self.cardGap)
        y = self.outerPadding + rowIndex * self.rowStride
        return wx.Rect(x, y, self.cardWidth, self.cardHeight)

    def _drawImageCard(
        self,
        paintDc: wx.DC,
        imageEntry: FolderImageEntry,
        cardRect: wx.Rect,
    ) -> None:
        borderColour = ACCENT if imageEntry.matchedPersonNames else BORDER
        paintDc.SetPen(wx.Pen(borderColour, 1))
        paintDc.SetBrush(wx.Brush(SURFACE_BG_ALT))
        paintDc.DrawRoundedRectangle(
            cardRect.x,
            cardRect.y,
            cardRect.width - 1,
            cardRect.height - 1,
            self.cardRadius,
        )

        thumbnailBitmap = self.thumbnailProvider(imageEntry.imagePath)
        thumbnailArea = wx.Rect(
            cardRect.x + self.cardPadding,
            cardRect.y + self.cardPadding,
            self.thumbnailWidth,
            self.thumbnailHeight,
        )
        if thumbnailBitmap is None:
            self._drawThumbnailPlaceholder(paintDc, thumbnailArea)
        else:
            bitmapX = thumbnailArea.x + max(0, (thumbnailArea.width - thumbnailBitmap.GetWidth()) // 2)
            bitmapY = thumbnailArea.y + max(0, (thumbnailArea.height - thumbnailBitmap.GetHeight()) // 2)
            paintDc.DrawBitmap(thumbnailBitmap, bitmapX, bitmapY, True)

        textX = cardRect.x + self.cardPadding
        textWidth = self.cardWidth - (self.cardPadding * 2)
        textY = thumbnailArea.y + thumbnailArea.height + 12
        textY = self._drawTextBlock(
            paintDc,
            imageEntry.imagePath.name,
            x=textX,
            y=textY,
            maxWidth=textWidth,
            font=self.bodyFont,
            colour=TEXT_PRIMARY,
            maxLines=2,
        )

        if imageEntry.matchedPersonNames:
            self._drawTextBlock(
                paintDc,
                f"Matches: {', '.join(imageEntry.matchedPersonNames)}",
                x=textX,
                y=textY + 4,
                maxWidth=textWidth,
                font=self.matchFont,
                colour=SUCCESS,
                maxLines=2,
            )

    def _drawThumbnailPlaceholder(self, paintDc: wx.DC, thumbnailArea: wx.Rect) -> None:
        paintDc.SetPen(wx.Pen(BORDER, 1))
        paintDc.SetBrush(wx.TRANSPARENT_BRUSH)
        paintDc.DrawRoundedRectangle(
            thumbnailArea.x,
            thumbnailArea.y,
            thumbnailArea.width,
            thumbnailArea.height,
            12,
        )
        paintDc.SetFont(self.placeholderFont)
        paintDc.SetTextForeground(TEXT_MUTED)
        paintDc.DrawLabel(
            "Preview unavailable",
            thumbnailArea,
            alignment=wx.ALIGN_CENTER_HORIZONTAL | wx.ALIGN_CENTER_VERTICAL,
        )

    def _drawTextBlock(
        self,
        paintDc: wx.DC,
        text: str,
        x: int,
        y: int,
        maxWidth: int,
        font: wx.Font,
        colour: wx.Colour,
        maxLines: int,
    ) -> int:
        paintDc.SetFont(font)
        paintDc.SetTextForeground(colour)
        lines = self._wrapTextLines(paintDc, text, maxWidth, maxLines)
        lineHeight = paintDc.GetTextExtent("Ag")[1] + 2
        currentY = y
        for line in lines:
            paintDc.DrawText(line, x, currentY)
            currentY += lineHeight

        return currentY

    def _wrapTextLines(
        self,
        paintDc: wx.DC,
        text: str,
        maxWidth: int,
        maxLines: int,
    ) -> list[str]:
        wrappedText = wordwrap(text, maxWidth, paintDc, breakLongWords=True)
        allLines = [line for line in wrappedText.splitlines() if line]
        if not allLines:
            return []

        wasTruncated = len(allLines) > maxLines
        lines = allLines[:maxLines]
        if wasTruncated:
            lines[-1] = self._ellipsizeText(paintDc, f"{lines[-1].rstrip('.')}...", maxWidth)
        else:
            lines[-1] = self._ellipsizeText(paintDc, lines[-1], maxWidth)

        return lines

    def _ellipsizeText(self, paintDc: wx.DC, text: str, maxWidth: int) -> str:
        if paintDc.GetTextExtent(text)[0] <= maxWidth:
            return text

        ellipsis = "..."
        if paintDc.GetTextExtent(ellipsis)[0] >= maxWidth:
            return ellipsis

        baseText = text.rstrip(".")
        while baseText and paintDc.GetTextExtent(f"{baseText}{ellipsis}")[0] > maxWidth:
            baseText = baseText[:-1].rstrip()

        return f"{baseText}{ellipsis}" if baseText else ellipsis

    def _entryAtPosition(self, position: wx.Point) -> FolderImageEntry | None:
        if not self.entries:
            return None

        logicalX, logicalY = self.CalcUnscrolledPosition(position.x, position.y)
        contentX = logicalX - self.outerPadding
        contentY = logicalY - self.outerPadding
        if contentX < 0 or contentY < 0:
            return None

        columnSpan = self.cardWidth + self.cardGap
        rowSpan = self.rowStride
        columnIndex = contentX // columnSpan
        rowIndex = contentY // rowSpan
        if columnIndex >= self.columnCount:
            return None

        if contentX % columnSpan >= self.cardWidth or contentY % rowSpan >= self.cardHeight:
            return None

        imageIndex = rowIndex * self.columnCount + columnIndex
        if imageIndex >= len(self.entries):
            return None

        return self.entries[imageIndex]


class FolderWorkspace(wx.Panel):
    thumbnailLoadBatchSize = 1
    thumbnailWorkerCount = max(1, (os.cpu_count() or 1) - 2)
    thumbnailSizeChoices = (120, 140, 160, 180, 200, 220, 240, 260, 280)
    thumbnailCacheRebuildBatchSize = 24

    def __init__(
        self,
        parent: wx.Window,
        personRepository: PersonRepository,
        faceRecognitionService: FaceRecognitionService,
        onRequestShowPeople: Callable[[], None] | None = None,
        onRequestOpenFolder: Callable[[], None] | None = None,
    ):
        super().__init__(parent)
        stylePanel(self)
        self.personRepository = personRepository
        self.faceRecognitionService = faceRecognitionService
        self.onRequestShowPeople = onRequestShowPeople
        self.onRequestOpenFolder = onRequestOpenFolder
        self.folderPath: Path | None = None
        self.allImageEntries: list[FolderImageEntry] = []
        self.visibleImageEntries: list[FolderImageEntry] = []
        self.filteredPersonNames: list[str] = []
        self.thumbnailWidth = FolderImageGallery.defaultThumbnailWidth
        self.thumbnailHeight = self._thumbnailHeightForWidth(self.thumbnailWidth)
        self.thumbnailSourceCache: dict[Path, PreparedBitmapData | None] = {}
        self.thumbnailCache: dict[Path, wx.Bitmap | None] = {}
        self.isLoadingFolder = False
        self.isSearching = False
        self._loadVersion = 0
        self._loadThumbnailCount = 0
        self._loadThumbnailTotal = 0
        self._loadWorkerFinished = False
        self._loadedImagePaths: list[Path] = []
        self._loadCancelEvent: threading.Event | None = None
        self._thumbnailCacheRebuildVersion = 0
        self._thumbnailCacheRebuildPaths: list[Path] = []
        self._thumbnailCacheRebuildIndex = 0
        self._searchVersion = 0
        self._searchCancelEvent: threading.Event | None = None
        self._isDestroyed = False

        self.headerCard = CardPanel(self, background=ACCENT)
        titleEyebrow = wx.StaticText(self.headerCard, label="Search")
        titleLabel = wx.StaticText(self.headerCard, label="Folder Search")
        self.folderLabel = wx.StaticText(self.headerCard, label="No folder imported.")
        self.statusLabel = wx.StaticText(
            self.headerCard,
            label="Use File -> Open Folder to load supported images and search for saved people.",
        )
        self.progressGauge = wx.Gauge(self.headerCard, range=1, style=wx.GA_HORIZONTAL | wx.GA_SMOOTH)
        self.progressGauge.Hide()

        styleText(titleEyebrow, "eyebrow")
        styleText(titleLabel, "app_title")
        styleText(self.folderLabel, "body")
        styleText(self.statusLabel, "muted")
        titleEyebrow.SetForegroundColour(wx.Colour(12, 17, 25))
        titleLabel.SetForegroundColour(wx.Colour(12, 17, 25))
        self.folderLabel.SetForegroundColour(wx.Colour(12, 17, 25))
        self.statusLabel.SetForegroundColour(wx.Colour(20, 28, 40))
        self.folderLabel.Wrap(920)
        self.statusLabel.Wrap(920)
        self.progressGauge.SetMinSize((280, 12))

        self.openFolderButton = wx.Button(self.headerCard, label="Open Folder")
        self.searchPeopleButton = wx.Button(self.headerCard, label="Search People")
        self.showAllButton = wx.Button(self.headerCard, label="Show All Images")
        self.cancelSearchButton = wx.Button(self.headerCard, label="Cancel Search")
        self.cancelSearchButton.Hide()

        self.openFolderButton.Bind(wx.EVT_BUTTON, self._onOpenFolder)
        self.searchPeopleButton.Bind(wx.EVT_BUTTON, self._onSearchPeople)
        self.showAllButton.Bind(wx.EVT_BUTTON, self._onShowAllImages)
        self.cancelSearchButton.Bind(wx.EVT_BUTTON, self._onCancelSearch)
        styleButton(self.openFolderButton, "primary")
        styleButton(self.searchPeopleButton, "secondary")
        styleButton(self.showAllButton, "subtle")
        styleButton(self.cancelSearchButton, "subtle")

        actionButtonSizer = wx.BoxSizer(wx.HORIZONTAL)
        actionButtonSizer.Add(self.openFolderButton, 0, wx.RIGHT, 8)
        actionButtonSizer.Add(self.searchPeopleButton, 0, wx.RIGHT, 8)
        actionButtonSizer.Add(self.showAllButton, 0, wx.RIGHT, 8)
        actionButtonSizer.Add(self.cancelSearchButton, 0)
        headerNavSizer = buildWorkspaceNavigationSizer(
            self.headerCard,
            activeView="folder",
            onRequestShowPeople=self.onRequestShowPeople,
            onRequestShowFolder=self._showCurrentWorkspace,
        )

        headerTextSizer = wx.BoxSizer(wx.VERTICAL)
        headerTextSizer.Add(titleEyebrow, 0, wx.BOTTOM, 6)
        headerTextSizer.Add(titleLabel, 0, wx.BOTTOM, 6)
        headerTextSizer.Add(self.folderLabel, 0, wx.BOTTOM, 6)
        headerTextSizer.Add(self.statusLabel, 0)
        headerTextSizer.Add(self.progressGauge, 0, wx.TOP | wx.EXPAND, 12)

        headerButtonSizer = wx.BoxSizer(wx.VERTICAL)
        headerButtonSizer.Add(headerNavSizer, 0, wx.BOTTOM | wx.ALIGN_RIGHT, 10)
        headerButtonSizer.Add(actionButtonSizer, 0, wx.ALIGN_RIGHT)

        headerSizer = wx.BoxSizer(wx.HORIZONTAL)
        headerSizer.Add(headerTextSizer, 1, wx.EXPAND)
        headerSizer.Add(headerButtonSizer, 0, wx.ALIGN_CENTER_VERTICAL)
        headerOuterSizer = wx.BoxSizer(wx.VERTICAL)
        headerOuterSizer.Add(headerSizer, 1, wx.ALL | wx.EXPAND, 22)
        self.headerCard.SetSizer(headerOuterSizer)

        galleryCard = CardPanel(self)
        galleryEyebrow = wx.StaticText(galleryCard, label="Gallery")
        galleryTitle = wx.StaticText(galleryCard, label="Imported Images")
        galleryCaption = wx.StaticText(
            galleryCard,
            label="Browse imported thumbnails and narrow the set down to matching people.",
        )
        styleText(galleryEyebrow, "eyebrow")
        styleText(galleryTitle, "section_title")
        styleText(galleryCaption, "muted")
        galleryCaption.Wrap(1000)

        self.imageGallery = FolderImageGallery(
            galleryCard,
            thumbnailProvider=self._getThumbnail,
        )
        self.imageGallery.setThumbnailDimensions(self.thumbnailWidth, self.thumbnailHeight)
        self.imageGallery.SetMinSize((0, 480))

        galleryToolbar = wx.Panel(galleryCard)
        stylePanel(galleryToolbar, background=SURFACE_BG_SUBTLE)
        galleryToolbar.SetMinSize((-1, 40))
        self.thumbnailSizeSlider = ThumbnailSizeSlider(
            galleryToolbar,
            values=self.thumbnailSizeChoices,
            initialValue=self.thumbnailWidth,
            onValueChanged=self._onThumbnailSizeChanged,
        )

        toolbarSizer = wx.BoxSizer(wx.HORIZONTAL)
        toolbarSizer.AddStretchSpacer()
        toolbarSizer.Add(self.thumbnailSizeSlider, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 4)
        galleryToolbar.SetSizer(toolbarSizer)

        gallerySizer = wx.BoxSizer(wx.VERTICAL)
        gallerySizer.Add(galleryEyebrow, 0, wx.BOTTOM, 6)
        gallerySizer.Add(galleryTitle, 0, wx.BOTTOM, 4)
        gallerySizer.Add(galleryCaption, 0, wx.BOTTOM, 14)
        gallerySizer.Add(self.imageGallery, 1, wx.EXPAND)
        gallerySizer.Add(galleryToolbar, 0, wx.TOP | wx.EXPAND, 12)
        galleryOuterSizer = wx.BoxSizer(wx.VERTICAL)
        galleryOuterSizer.Add(gallerySizer, 1, wx.ALL | wx.EXPAND, 22)
        galleryCard.SetSizer(galleryOuterSizer)

        rootSizer = wx.BoxSizer(wx.VERTICAL)
        rootSizer.Add(self.headerCard, 0, wx.ALL | wx.EXPAND, 4)
        rootSizer.Add(galleryCard, 1, wx.TOP | wx.BOTTOM | wx.EXPAND, 18)
        self.SetSizer(rootSizer)

        self.Bind(wx.EVT_WINDOW_DESTROY, self._onDestroy)

        self._updateButtons()
        self._renderVisibleImages(resetScroll=True)

    def loadFolder(self, folderPath: str | Path) -> None:
        folder = Path(folderPath)
        self._cancelLoad()
        self._cancelSearch(invalidate=True)
        self._cancelThumbnailCacheRebuild()
        self._loadVersion += 1
        loadVersion = self._loadVersion

        self.folderPath = folder
        self.isLoadingFolder = True
        self.filteredPersonNames.clear()
        self.allImageEntries = []
        self.visibleImageEntries = []
        self.thumbnailSourceCache.clear()
        self.thumbnailCache.clear()

        self.folderLabel.SetLabel(f"Folder: {folder}")
        self.folderLabel.Wrap(920)
        self._loadThumbnailCount = 0
        self._loadThumbnailTotal = 0
        self._loadWorkerFinished = False
        self._loadedImagePaths = []
        self._loadCancelEvent = threading.Event()
        self._showLoadProgress(totalImages=1)
        self._setStatus(f"Finding supported images in {folder}...")
        self._updateButtons()
        self._renderVisibleImages(resetScroll=True)
        self.Layout()

        threading.Thread(
            target=self._loadFolderWorker,
            args=(loadVersion, folder, self._loadCancelEvent),
            daemon=True,
        ).start()

    def showAllImages(self) -> None:
        if self.isLoadingFolder or self.isSearching:
            return

        self.filteredPersonNames.clear()
        self.visibleImageEntries = list(self.allImageEntries)
        self._updateStatusForCurrentView()
        self._updateButtons()
        self._renderVisibleImages(resetScroll=True)

    def searchForPeople(self) -> None:
        if self.isLoadingFolder:
            wx.MessageBox("Wait for the folder to finish loading before searching.", "Folder Search")
            return

        if self.isSearching:
            return

        if not self.allImageEntries:
            wx.MessageBox("Import a folder with supported images before searching.", "Folder Search")
            return

        people = self.personRepository.listPeopleSummaries()
        if not people:
            wx.MessageBox("No saved people are available for searching yet.", "Folder Search")
            return

        selectedPeople = self._promptForPeople(people)
        if not selectedPeople:
            return

        personEmbeddingsById = self._loadPeopleEmbeddings(selectedPeople)
        searchablePeople = [
            person
            for person in selectedPeople
            if person.id in personEmbeddingsById
        ]
        if not searchablePeople:
            wx.MessageBox("The selected people do not have any saved face embeddings yet.", "Folder Search")
            return

        skippedPersonNames = [
            person.name
            for person in selectedPeople
            if person.id not in personEmbeddingsById
        ]
        if skippedPersonNames:
            wx.MessageBox(
                "Some selected people were skipped because they do not have any saved face embeddings yet.\n\n"
                f"{', '.join(skippedPersonNames)}",
                "Folder Search",
            )

        self.isSearching = True
        self._searchVersion += 1
        searchVersion = self._searchVersion
        selectedPersonNames = [person.name for person in searchablePeople]
        selectedPersonNamesById = {person.id: person.name for person in searchablePeople}
        imagePaths = [imageEntry.imagePath for imageEntry in self.allImageEntries]
        self._searchCancelEvent = threading.Event()

        self._showSearchProgress(totalImages=len(imagePaths))
        self._setStatus("Preparing recognition model and scanning imported images...")
        self._updateButtons()

        threading.Thread(
            target=self._searchWorker,
            args=(
                searchVersion,
                self._searchCancelEvent,
                imagePaths,
                personEmbeddingsById,
                selectedPersonNames,
                selectedPersonNamesById,
            ),
            daemon=True,
        ).start()

    def _loadFolderWorker(
        self,
        loadVersion: int,
        folder: Path,
        cancelEvent: threading.Event,
    ) -> None:
        try:
            imagePaths = listSupportedImagesInFolder(folder)
            if cancelEvent.is_set():
                return
        except Exception as error:
            if cancelEvent.is_set():
                return
            wx.CallAfter(self._finishLoadFolder, loadVersion, None, str(error))
            return

        totalImages = len(imagePaths)
        wx.CallAfter(self._beginLoadThumbnailGeneration, loadVersion, totalImages)
        if cancelEvent.is_set():
            return
        if not imagePaths:
            if not cancelEvent.is_set():
                wx.CallAfter(self._markLoadWorkerFinished, loadVersion, imagePaths)
            return

        preparedBatch: list[tuple[Path, PreparedBitmapData | None]] = []
        processedCount = 0
        executor = ThreadPoolExecutor(
            max_workers=self.thumbnailWorkerCount,
            thread_name_prefix="thumbnailLoad",
        )
        futures: set[Future[tuple[Path, PreparedBitmapData | None]]] = set()
        try:
            for imagePath in imagePaths:
                if cancelEvent.is_set():
                    return

                futures.add(
                    executor.submit(
                        prepareThumbnailData,
                        imagePath,
                        self.thumbnailSizeChoices[-1],
                        self._thumbnailHeightForWidth(self.thumbnailSizeChoices[-1]),
                        cancelEvent,
                    )
                )

            while futures:
                if cancelEvent.is_set():
                    return

                completedFutures, pendingFutures = wait(
                    futures,
                    timeout=0.1,
                    return_when=FIRST_COMPLETED,
                )
                if not completedFutures:
                    futures = set(pendingFutures)
                    continue

                futures = set(pendingFutures)
                for completedFuture in completedFutures:
                    imagePath, preparedBitmap = completedFuture.result()
                    processedCount += 1
                    preparedBatch.append((imagePath, preparedBitmap))
                    if (
                        len(preparedBatch) >= self.thumbnailLoadBatchSize
                        or processedCount == totalImages
                    ):
                        wx.CallAfter(
                            self._appendLoadedThumbnailBatch,
                            loadVersion,
                            preparedBatch.copy(),
                            processedCount,
                            totalImages,
                        )
                        preparedBatch.clear()
        except Exception as error:
            if cancelEvent.is_set():
                return
            wx.CallAfter(self._finishLoadFolder, loadVersion, None, str(error))
            return
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

        if cancelEvent.is_set():
            return

        wx.CallAfter(self._markLoadWorkerFinished, loadVersion, imagePaths)

    def _beginLoadThumbnailGeneration(self, loadVersion: int, totalImages: int) -> None:
        if self._isDestroyed or loadVersion != self._loadVersion or not self.isLoadingFolder:
            return

        self._loadThumbnailCount = 0
        self._loadThumbnailTotal = totalImages
        self._loadWorkerFinished = False
        self._loadedImagePaths = []
        self.progressGauge.SetRange(max(1, totalImages))
        self.progressGauge.SetValue(0)
        if totalImages:
            self._setStatus(f"Preparing thumbnails for {totalImages} image(s)...")

    def _appendLoadedThumbnailBatch(
        self,
        loadVersion: int,
        preparedBatch: list[tuple[Path, PreparedBitmapData | None]],
        processedCount: int,
        totalImages: int,
    ) -> None:
        if self._isDestroyed or loadVersion != self._loadVersion or not self.isLoadingFolder:
            return

        for imagePath, preparedBitmap in preparedBatch:
            self.thumbnailSourceCache[imagePath] = preparedBitmap
            self.thumbnailCache[imagePath] = self._bitmapFromPreparedThumbnail(preparedBitmap)

        self._loadThumbnailCount = max(self._loadThumbnailCount, processedCount)
        self.progressGauge.SetRange(max(1, totalImages))
        self.progressGauge.SetValue(self._loadThumbnailCount)
        self._setStatus(
            f"Preparing thumbnails ({self._loadThumbnailCount}/{max(1, totalImages)})..."
        )
        self._tryCompleteLoad(loadVersion)

    def _markLoadWorkerFinished(self, loadVersion: int, imagePaths: list[Path]) -> None:
        if self._isDestroyed or loadVersion != self._loadVersion or not self.isLoadingFolder:
            return

        self._loadedImagePaths = list(imagePaths)
        self._loadWorkerFinished = True
        self._tryCompleteLoad(loadVersion)

    def _tryCompleteLoad(self, loadVersion: int) -> None:
        if self._isDestroyed or loadVersion != self._loadVersion or not self.isLoadingFolder:
            return

        if not self._loadWorkerFinished:
            return

        if self._loadThumbnailCount < self._loadThumbnailTotal:
            return

        imagePaths = list(self._loadedImagePaths)
        self._loadedImagePaths = []
        self._loadWorkerFinished = False
        self._finishLoadFolder(loadVersion, imagePaths, None)

    def _finishLoadFolder(
        self,
        loadVersion: int,
        imagePaths: list[Path] | None,
        errorMessage: str | None,
    ) -> None:
        if self._isDestroyed or loadVersion != self._loadVersion:
            return

        self.isLoadingFolder = False
        self._loadThumbnailCount = 0
        self._loadThumbnailTotal = 0
        self._loadWorkerFinished = False
        self._loadedImagePaths = []
        self._loadCancelEvent = None
        self._hideLoadProgress()
        if errorMessage is not None:
            self.allImageEntries = []
            self.visibleImageEntries = []
            self._setStatus("Could not load supported images from the selected folder.")
            self._updateButtons()
            self._renderVisibleImages(resetScroll=True)
            wx.MessageBox(f"Could not load the selected folder.\n\n{errorMessage}", "Folder Search")
            return

        self.allImageEntries = [
            FolderImageEntry(imagePath=imagePath)
            for imagePath in imagePaths or []
        ]
        self.visibleImageEntries = list(self.allImageEntries)
        self._updateStatusForCurrentView()
        self._updateButtons()
        self._renderVisibleImages(resetScroll=True)
        self.Layout()

    def _searchWorker(
        self,
        searchVersion: int,
        cancelEvent: threading.Event,
        imagePaths: list[Path],
        personEmbeddingsById: dict[int, list[np.ndarray]],
        selectedPersonNames: list[str],
        selectedPersonNamesById: dict[int, str],
    ) -> None:
        try:
            workerFaceRecognitionService = FaceRecognitionService(
                modelName=self.faceRecognitionService.modelName
            )
            matchingImagePaths: list[Path] = []
            matchedPersonIdsByPath: dict[Path, set[int]] = {}
            matchedPersonNamesByPath: dict[Path, tuple[str, ...]] = {}
            failureCount = 0
            lastProgressUpdateAt = 0.0

            for index, imagePath in enumerate(imagePaths, start=1):
                if cancelEvent.is_set():
                    wx.CallAfter(self._finishSearchCancelled, searchVersion)
                    return

                try:
                    imageBgr = workerFaceRecognitionService.readImage(imagePath)
                    matchedPersonIds = workerFaceRecognitionService.findMatchingPeople(
                        imageBgr=imageBgr,
                        personEmbeddingsById=personEmbeddingsById,
                        candidateEmbeddingsNormalized=True,
                    )
                except Exception:
                    failureCount += 1
                    matchedPersonIds = set()

                matchedPersonIdsByPath[imagePath] = matchedPersonIds
                matchedPersonNamesByPath[imagePath] = tuple(
                    selectedPersonNamesById[personId]
                    for personId in sorted(matchedPersonIds)
                    if personId in selectedPersonNamesById
                )
                if matchedPersonIds:
                    matchingImagePaths.append(imagePath)

                now = time.monotonic()
                if index == len(imagePaths) or now - lastProgressUpdateAt >= 0.08:
                    wx.CallAfter(
                        self._updateSearchProgress,
                        searchVersion,
                        index,
                        len(imagePaths),
                        imagePath.name,
                    )
                    lastProgressUpdateAt = now

            if cancelEvent.is_set():
                wx.CallAfter(self._finishSearchCancelled, searchVersion)
                return

            wx.CallAfter(
                self._finishSearch,
                searchVersion,
                FolderSearchResult(
                    filteredPersonNames=selectedPersonNames,
                    matchingImagePaths=matchingImagePaths,
                    matchedPersonIdsByPath=matchedPersonIdsByPath,
                    matchedPersonNamesByPath=matchedPersonNamesByPath,
                    failureCount=failureCount,
                ),
                None,
            )
        except Exception as error:
            wx.CallAfter(self._finishSearch, searchVersion, None, str(error))

    def _updateSearchProgress(
        self,
        searchVersion: int,
        currentIndex: int,
        totalImages: int,
        imageName: str,
    ) -> None:
        if self._isDestroyed or searchVersion != self._searchVersion or not self.isSearching:
            return

        self.progressGauge.SetRange(max(1, totalImages))
        self.progressGauge.SetValue(currentIndex)
        self._setStatus(f"Scanning {imageName} ({currentIndex}/{totalImages})")

    def _finishSearchCancelled(self, searchVersion: int) -> None:
        if self._isDestroyed or searchVersion != self._searchVersion:
            return

        self.isSearching = False
        self._searchCancelEvent = None
        self._hideSearchProgress()
        self._updateStatusForCurrentView()
        self._updateButtons()

    def _finishSearch(
        self,
        searchVersion: int,
        searchResult: FolderSearchResult | None,
        errorMessage: str | None,
    ) -> None:
        if self._isDestroyed or searchVersion != self._searchVersion:
            return

        self.isSearching = False
        self._searchCancelEvent = None
        self._hideSearchProgress()

        if errorMessage is not None:
            self._updateStatusForCurrentView()
            self._updateButtons()
            wx.MessageBox(f"The folder search failed.\n\n{errorMessage}", "Folder Search")
            return

        if searchResult is None:
            self._updateStatusForCurrentView()
            self._updateButtons()
            return

        entriesByPath = {
            imageEntry.imagePath: imageEntry
            for imageEntry in self.allImageEntries
        }
        for imageEntry in self.allImageEntries:
            imageEntry.matchedPersonIds = set(
                searchResult.matchedPersonIdsByPath.get(imageEntry.imagePath, set())
            )
            imageEntry.matchedPersonNames = searchResult.matchedPersonNamesByPath.get(
                imageEntry.imagePath,
                (),
            )

        self.filteredPersonNames = list(searchResult.filteredPersonNames)
        self.visibleImageEntries = [
            entriesByPath[imagePath]
            for imagePath in searchResult.matchingImagePaths
            if imagePath in entriesByPath
        ]
        self._updateStatusForCurrentView()
        self._updateButtons()
        self._renderVisibleImages(resetScroll=True)

        if searchResult.failureCount:
            wx.MessageBox(
                f"{searchResult.failureCount} image(s) could not be processed and were skipped.",
                "Folder Search",
            )

    def _showSearchProgress(self, totalImages: int) -> None:
        if self._isDestroyed:
            return

        self.progressGauge.SetRange(max(1, totalImages))
        self.progressGauge.SetValue(0)
        self.progressGauge.Show()
        self.cancelSearchButton.Enable(True)
        self.cancelSearchButton.Show()
        self.headerCard.Layout()
        self.Layout()

    def _hideSearchProgress(self) -> None:
        if self._isDestroyed:
            return

        self.progressGauge.Hide()
        self.cancelSearchButton.Hide()
        self.progressGauge.SetValue(0)
        self.cancelSearchButton.Enable(True)
        self.headerCard.Layout()
        self.Layout()

    def _showLoadProgress(self, totalImages: int) -> None:
        if self._isDestroyed:
            return

        self.progressGauge.SetRange(max(1, totalImages))
        self.progressGauge.SetValue(0)
        self.progressGauge.Show()
        self.cancelSearchButton.Hide()
        self.headerCard.Layout()
        self.Layout()

    def _hideLoadProgress(self) -> None:
        if self._isDestroyed:
            return

        self.progressGauge.Hide()
        self.progressGauge.SetValue(0)
        self.headerCard.Layout()
        self.Layout()

    def _cancelLoad(self) -> None:
        if self._loadCancelEvent is None:
            return

        self._loadCancelEvent.set()
        self._loadCancelEvent = None

    def _cancelSearch(self, invalidate: bool = False) -> None:
        if self._searchCancelEvent is not None:
            self._searchCancelEvent.set()

        if not invalidate:
            return

        self._searchVersion += 1
        self.isSearching = False
        self._searchCancelEvent = None
        self._hideSearchProgress()

    def _promptForPeople(self, people: list[PersonSummary]) -> list[PersonSummary]:
        dialog = wx.MultiChoiceDialog(
            self,
            "Choose one or more saved people to search for in the imported images.",
            "Search People",
            [person.name for person in people],
        )
        try:
            if dialog.ShowModal() != wx.ID_OK:
                return []

            selectedIndices = list(dialog.GetSelections())
            return [people[index] for index in selectedIndices]
        finally:
            dialog.Destroy()

    def _loadPeopleEmbeddings(
        self,
        selectedPeople: list[PersonSummary],
    ) -> dict[int, list[np.ndarray]]:
        personEmbeddingsById: dict[int, list[np.ndarray]] = {}

        for person in selectedPeople:
            embeddings: list[np.ndarray] = []
            for faceSample in self.personRepository.listFaceSamplesForPerson(person.id):
                embedding = self.faceRecognitionService.deserializeEmbedding(
                    faceSample.embedding,
                    faceSample.embeddingLength,
                )
                if embedding is None:
                    continue

                embeddings.append(self.faceRecognitionService.normalizeEmbedding(embedding))

            if embeddings:
                personEmbeddingsById[person.id] = embeddings

        return personEmbeddingsById

    def _thumbnailHeightForWidth(self, thumbnailWidth: int) -> int:
        return max(
            1,
            int(
                round(
                    thumbnailWidth
                    * FolderImageGallery.defaultThumbnailHeight
                    / FolderImageGallery.defaultThumbnailWidth
                )
            ),
        )

    def _bitmapFromPreparedThumbnail(
        self,
        preparedBitmap: PreparedBitmapData | None,
    ) -> wx.Bitmap | None:
        if preparedBitmap is None:
            return None

        return bitmapFromPreparedBitmapSized(
            preparedBitmap,
            maxWidth=self.thumbnailWidth,
            maxHeight=self.thumbnailHeight,
        )

    def _cancelThumbnailCacheRebuild(self) -> None:
        self._thumbnailCacheRebuildVersion += 1
        self._thumbnailCacheRebuildPaths = []
        self._thumbnailCacheRebuildIndex = 0

    def _startThumbnailCacheRebuild(self) -> None:
        if self._isDestroyed or self.isLoadingFolder or not self.thumbnailSourceCache:
            return

        self._cancelThumbnailCacheRebuild()
        self.thumbnailCache.clear()
        self._thumbnailCacheRebuildPaths = self._orderedThumbnailCacheRebuildPaths()
        if not self._thumbnailCacheRebuildPaths:
            self.imageGallery.Refresh()
            return

        rebuildVersion = self._thumbnailCacheRebuildVersion
        wx.CallAfter(self._continueThumbnailCacheRebuild, rebuildVersion)

    def _orderedThumbnailCacheRebuildPaths(self) -> list[Path]:
        visiblePaths = [
            imageEntry.imagePath
            for imageEntry in self.visibleImageEntries
            if imageEntry.imagePath in self.thumbnailSourceCache
        ]
        visiblePathSet = set(visiblePaths)
        remainingPaths = [
            imageEntry.imagePath
            for imageEntry in self.allImageEntries
            if (
                imageEntry.imagePath in self.thumbnailSourceCache
                and imageEntry.imagePath not in visiblePathSet
            )
        ]
        return visiblePaths + remainingPaths

    def _continueThumbnailCacheRebuild(self, rebuildVersion: int) -> None:
        if self._isDestroyed or rebuildVersion != self._thumbnailCacheRebuildVersion:
            return

        startIndex = self._thumbnailCacheRebuildIndex
        if startIndex >= len(self._thumbnailCacheRebuildPaths):
            return

        endIndex = min(
            startIndex + self.thumbnailCacheRebuildBatchSize,
            len(self._thumbnailCacheRebuildPaths),
        )
        batchPaths = self._thumbnailCacheRebuildPaths[startIndex:endIndex]
        visiblePathSet = {
            imageEntry.imagePath
            for imageEntry in self.visibleImageEntries
        }
        needsRefresh = False
        for imagePath in batchPaths:
            if imagePath in self.thumbnailCache:
                continue

            self.thumbnailCache[imagePath] = self._bitmapFromPreparedThumbnail(
                self.thumbnailSourceCache.get(imagePath)
            )
            if imagePath in visiblePathSet:
                needsRefresh = True

        self._thumbnailCacheRebuildIndex = endIndex
        if needsRefresh:
            self.imageGallery.Refresh()

        if endIndex < len(self._thumbnailCacheRebuildPaths):
            wx.CallAfter(self._continueThumbnailCacheRebuild, rebuildVersion)

    def _onThumbnailSizeChanged(self, thumbnailWidth: int) -> None:
        if thumbnailWidth == self.thumbnailWidth:
            return

        self.thumbnailWidth = thumbnailWidth
        self.thumbnailHeight = self._thumbnailHeightForWidth(thumbnailWidth)
        self.imageGallery.setThumbnailDimensions(self.thumbnailWidth, self.thumbnailHeight)
        self._startThumbnailCacheRebuild()
        self.imageGallery.Refresh()
        self.Layout()

    def _renderVisibleImages(self, resetScroll: bool = False) -> None:
        self.imageGallery.setEntries(
            self.visibleImageEntries,
            self._emptyGalleryMessage(),
            resetScroll=resetScroll,
        )

    def _emptyGalleryMessage(self) -> str:
        if self.isLoadingFolder:
            return "Loading supported images..."

        if self.folderPath is None:
            return "No folder imported."

        if not self.allImageEntries:
            return "No supported images found in the imported folder."

        if self.filteredPersonNames:
            return "No imported images matched the selected people."

        return "No images to display."

    def _getThumbnail(self, imagePath: Path) -> wx.Bitmap | None:
        if imagePath in self.thumbnailCache:
            return self.thumbnailCache[imagePath]

        if imagePath not in self.thumbnailSourceCache:
            return None

        bitmap = self._bitmapFromPreparedThumbnail(self.thumbnailSourceCache[imagePath])
        self.thumbnailCache[imagePath] = bitmap
        return bitmap

    def _updateButtons(self) -> None:
        hasImages = bool(self.allImageEntries)
        self.openFolderButton.Enable(not self.isLoadingFolder)
        self.searchPeopleButton.Enable(hasImages and not self.isLoadingFolder and not self.isSearching)
        self.showAllButton.Enable(
            hasImages
            and bool(self.filteredPersonNames)
            and not self.isLoadingFolder
            and not self.isSearching
        )
        self.thumbnailSizeSlider.Enable(not self.isLoadingFolder)

    def _updateStatusForCurrentView(self) -> None:
        if self.folderPath is None:
            self._setStatus("Use File -> Open Folder to load supported images and search for saved people.")
            return

        if not self.allImageEntries:
            self._setStatus("No supported image files were found in the imported folder.")
            return

        if self.filteredPersonNames:
            self._setStatus(
                "Showing "
                f"{len(self.visibleImageEntries)} of {len(self.allImageEntries)} image(s) containing: "
                f"{', '.join(self.filteredPersonNames)}"
            )
            return

        self._setStatus(
            f"Showing all {len(self.visibleImageEntries)} supported image(s) from the imported folder."
        )

    def _setStatus(self, text: str) -> None:
        self.statusLabel.SetLabel(text)
        self.statusLabel.Wrap(920)
        self.headerCard.Layout()

    def _showCurrentWorkspace(self) -> None:
        return

    def _onOpenFolder(self, event: wx.CommandEvent) -> None:
        if self.onRequestOpenFolder is not None:
            self.onRequestOpenFolder()

    def _onSearchPeople(self, event: wx.CommandEvent) -> None:
        self.searchForPeople()

    def _onShowAllImages(self, event: wx.CommandEvent) -> None:
        self.showAllImages()

    def _onCancelSearch(self, event: wx.CommandEvent) -> None:
        if self._searchCancelEvent is None or self._searchCancelEvent.is_set():
            return

        self._searchCancelEvent.set()
        self.cancelSearchButton.Enable(False)
        self._setStatus("Cancelling search...")

    def _onDestroy(self, event: wx.WindowDestroyEvent) -> None:
        if event.GetEventObject() is self:
            self._isDestroyed = True
            self._cancelThumbnailCacheRebuild()
            self._cancelLoad()
            self._loadVersion += 1
            self._cancelSearch(invalidate=True)

        event.Skip()
