from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Callable

import wx

from src.app.imagefiles import SUPPORTED_IMAGE_EXTENSIONS
from src.app.face.facerecognition import FaceRecognitionService
from src.data.store import FaceSample, Person, PersonRepository, PersonSummary
from src.ui.image_utils import bitmapFromEncodedImage
from src.ui.people_dialogs import OriginalImageFrame
from src.ui.people_workflow import collectFaceSamples, promptForImagePaths
from src.ui.theme import (
    ACCENT,
    BORDER,
    SURFACE_BG_ALT,
    CardPanel,
    applyAlternatingRowColour,
    styleButton,
    styleListBox,
    styleListCtrl,
    stylePanel,
    styleScrolledWindow,
    styleText,
    styleTextCtrl,
)


def showRepositoryWriteError(parent: wx.Window, action: str, error: Exception) -> None:
    wx.MessageBox(f"Could not {action}.\n\n{error}", "People")


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


class PeopleWorkspace(wx.Panel):
    def __init__(
        self,
        parent: wx.Window,
        personRepository: PersonRepository,
        faceRecognitionService: FaceRecognitionService,
        onRequestShowFolder: Callable[[], None] | None = None,
    ):
        super().__init__(parent)
        stylePanel(self)
        self.personRepository = personRepository
        self.faceRecognitionService = faceRecognitionService
        self.pageBook = wx.Simplebook(self)
        self.allPeoplePanel = AllPeoplePanel(
            self.pageBook,
            personRepository=self.personRepository,
            faceRecognitionService=self.faceRecognitionService,
            onRequestShowPeople=self.showAllPeople,
            onRequestShowFolder=onRequestShowFolder,
            onRequestAddPerson=self.showAddPerson,
        )
        self.addPersonPanel = AddPersonPanel(
            self.pageBook,
            personRepository=self.personRepository,
            faceRecognitionService=self.faceRecognitionService,
            onRequestShowPeople=self.showAllPeople,
            onRequestShowFolder=onRequestShowFolder,
            onPersonSaved=self._onPersonSaved,
        )

        self.pageBook.AddPage(self.allPeoplePanel, "All People")
        self.pageBook.AddPage(self.addPersonPanel, "Add new Person")
        self.pageBook.SetSelection(0)
        self.pageBook.SetBackgroundColour(self.GetBackgroundColour())

        rootSizer = wx.BoxSizer(wx.VERTICAL)
        rootSizer.Add(self.pageBook, 1, wx.EXPAND)
        self.SetSizer(rootSizer)

    def showAllPeople(self, selectedPersonId: int | None = None) -> None:
        self.allPeoplePanel.refreshPeople(selectedPersonId)
        self.pageBook.SetSelection(0)

    def showAddPerson(self) -> None:
        self.pageBook.SetSelection(1)

    def _onPersonSaved(self, person: Person) -> None:
        self.showAllPeople(person.id)


class AllPeoplePanel(wx.Panel):
    def __init__(
        self,
        parent: wx.Window,
        personRepository: PersonRepository,
        faceRecognitionService: FaceRecognitionService,
        onRequestShowPeople: Callable[[], None] | None = None,
        onRequestShowFolder: Callable[[], None] | None = None,
        onRequestAddPerson: Callable[[], None] | None = None,
    ):
        super().__init__(parent)
        stylePanel(self)
        self.personRepository = personRepository
        self.faceRecognitionService = faceRecognitionService
        self.onRequestAddPerson = onRequestAddPerson
        self.peopleById: dict[int, PersonSummary] = {}
        self.rowPersonIds: list[int] = []
        self.originalImageFrames: list[OriginalImageFrame] = []

        headerCard = CardPanel(self, background=ACCENT)
        headerEyebrow = wx.StaticText(headerCard, label="People")
        headerTitle = wx.StaticText(headerCard, label="People Library")
        headerCaption = wx.StaticText(
            headerCard,
            label="Curate saved identities, review stored face crops, and keep every profile current.",
        )
        styleText(headerEyebrow, "eyebrow")
        styleText(headerTitle, "app_title")
        styleText(headerCaption, "muted")
        headerEyebrow.SetForegroundColour(wx.Colour(12, 17, 25))
        headerTitle.SetForegroundColour(wx.Colour(12, 17, 25))
        headerCaption.SetForegroundColour(wx.Colour(20, 28, 40))
        headerCaption.Wrap(820)
        headerTextSizer = wx.BoxSizer(wx.VERTICAL)
        headerTextSizer.Add(headerEyebrow, 0, wx.BOTTOM, 6)
        headerTextSizer.Add(headerTitle, 0, wx.BOTTOM, 6)
        headerTextSizer.Add(headerCaption, 0)
        headerNavSizer = buildWorkspaceNavigationSizer(
            headerCard,
            activeView="people",
            onRequestShowPeople=onRequestShowPeople,
            onRequestShowFolder=onRequestShowFolder,
        )
        headerSizer = wx.BoxSizer(wx.HORIZONTAL)
        headerSizer.Add(headerTextSizer, 1, wx.EXPAND)
        headerSizer.Add(headerNavSizer, 0, wx.ALIGN_CENTER_VERTICAL)
        headerOuterSizer = wx.BoxSizer(wx.VERTICAL)
        headerOuterSizer.Add(headerSizer, 1, wx.ALL | wx.EXPAND, 22)
        headerCard.SetSizer(headerOuterSizer)

        libraryCard = CardPanel(self)
        self.peopleList = wx.ListCtrl(
            libraryCard,
            style=wx.LC_REPORT | wx.BORDER_SUNKEN | wx.LC_SINGLE_SEL,
        )
        self.peopleList.InsertColumn(0, "Name", width=220)
        self.peopleList.InsertColumn(1, "Samples", width=90)
        self.peopleList.InsertColumn(2, "Updated", width=180)
        self.peopleList.Bind(wx.EVT_LIST_ITEM_SELECTED, self._onPersonSelected)
        self.peopleList.SetMinSize((430, 460))
        styleListCtrl(self.peopleList)

        newPersonButton = wx.Button(libraryCard, label="New Person")
        refreshButton = wx.Button(libraryCard, label="Refresh")
        self.addImagesButton = wx.Button(libraryCard, label="Add Images")
        self.addImagesButton.Enable(False)

        newPersonButton.Bind(wx.EVT_BUTTON, self._onNewPerson)
        refreshButton.Bind(wx.EVT_BUTTON, self._onRefresh)
        self.addImagesButton.Bind(wx.EVT_BUTTON, self._onAddImages)
        styleButton(newPersonButton, "primary")
        styleButton(refreshButton, "secondary")
        styleButton(self.addImagesButton, "secondary")

        buttonSizer = wx.BoxSizer(wx.HORIZONTAL)
        buttonSizer.Add(newPersonButton, 0, wx.RIGHT, 8)
        buttonSizer.Add(refreshButton, 0, wx.RIGHT, 8)
        buttonSizer.Add(self.addImagesButton, 0)

        libraryEyebrow = wx.StaticText(libraryCard, label="Directory")
        libraryTitle = wx.StaticText(libraryCard, label="Saved People")
        libraryCaption = wx.StaticText(
            libraryCard,
            label="Select a profile to inspect saved samples or add more images to it.",
        )
        styleText(libraryEyebrow, "eyebrow")
        styleText(libraryTitle, "section_title")
        styleText(libraryCaption, "muted")
        libraryCaption.Wrap(360)
        librarySizer = wx.BoxSizer(wx.VERTICAL)
        librarySizer.Add(libraryEyebrow, 0, wx.BOTTOM, 6)
        librarySizer.Add(libraryTitle, 0, wx.BOTTOM, 4)
        librarySizer.Add(libraryCaption, 0, wx.BOTTOM, 14)
        librarySizer.Add(self.peopleList, 1, wx.BOTTOM | wx.EXPAND, 12)
        librarySizer.Add(buttonSizer, 0)
        libraryOuterSizer = wx.BoxSizer(wx.VERTICAL)
        libraryOuterSizer.Add(librarySizer, 1, wx.ALL | wx.EXPAND, 22)
        libraryCard.SetSizer(libraryOuterSizer)

        detailsCard = CardPanel(self)
        detailsEyebrow = wx.StaticText(detailsCard, label="Profile")
        detailsTitle = wx.StaticText(detailsCard, label="Saved Face Samples")
        self.personNameLabel = wx.StaticText(detailsCard, label="Select a person")
        self.personSummaryLabel = wx.StaticText(
            detailsCard,
            label="The saved face cutouts for the selected person will appear here.",
        )
        styleText(detailsEyebrow, "eyebrow")
        styleText(detailsTitle, "section_title")
        styleText(self.personNameLabel, "page_title")
        styleText(self.personSummaryLabel, "muted")
        self.personSummaryLabel.Wrap(520)

        self.thumbnailScroller = wx.ScrolledWindow(detailsCard, style=wx.VSCROLL)
        self.thumbnailScroller.SetScrollRate(12, 12)
        self.thumbnailSizer = wx.WrapSizer(wx.HORIZONTAL)
        self.thumbnailScroller.SetSizer(self.thumbnailSizer)
        self.thumbnailScroller.SetMinSize((620, 500))
        styleScrolledWindow(self.thumbnailScroller)

        detailsSizer = wx.BoxSizer(wx.VERTICAL)
        detailsSizer.Add(detailsEyebrow, 0, wx.BOTTOM, 6)
        detailsSizer.Add(detailsTitle, 0, wx.BOTTOM, 10)
        detailsSizer.Add(self.personNameLabel, 0, wx.BOTTOM, 6)
        detailsSizer.Add(self.personSummaryLabel, 0, wx.BOTTOM, 12)
        detailsSizer.Add(self.thumbnailScroller, 1, wx.EXPAND)
        detailsOuterSizer = wx.BoxSizer(wx.VERTICAL)
        detailsOuterSizer.Add(detailsSizer, 1, wx.ALL | wx.EXPAND, 22)
        detailsCard.SetSizer(detailsOuterSizer)

        contentSizer = wx.BoxSizer(wx.HORIZONTAL)
        contentSizer.Add(libraryCard, 0, wx.RIGHT | wx.EXPAND, 18)
        contentSizer.Add(detailsCard, 1, wx.EXPAND)

        rootSizer = wx.BoxSizer(wx.VERTICAL)
        rootSizer.Add(headerCard, 0, wx.ALL | wx.EXPAND, 4)
        rootSizer.Add(contentSizer, 1, wx.TOP | wx.BOTTOM | wx.EXPAND, 18)
        self.SetSizer(rootSizer)

        self.refreshPeople()

    def refreshPeople(self, selectedPersonId: int | None = None) -> None:
        people = self.personRepository.listPeopleSummaries()
        self.peopleById = {person.id: person for person in people}
        currentPersonId = selectedPersonId if selectedPersonId is not None else self.getSelectedPersonId()
        self.peopleList.DeleteAllItems()
        self.rowPersonIds.clear()

        for person in people:
            rowIndex = self.peopleList.InsertItem(self.peopleList.GetItemCount(), person.name)
            self.peopleList.SetItem(rowIndex, 1, str(person.sampleCount))
            self.peopleList.SetItem(rowIndex, 2, person.updatedAt)
            applyAlternatingRowColour(self.peopleList, rowIndex)
            self.rowPersonIds.append(person.id)

        if not people:
            self._showEmptyState()
            return

        if currentPersonId is not None and self._selectPersonById(currentPersonId):
            return

        self._selectPersonById(people[0].id)

    def getSelectedPersonId(self) -> int | None:
        selectedIndex = self.peopleList.GetFirstSelected()
        if selectedIndex == -1:
            return None

        return self.rowPersonIds[selectedIndex]

    def _onPersonSelected(self, event: wx.ListEvent) -> None:
        personId = self.getSelectedPersonId()
        if personId is None:
            self._showEmptyState()
            return

        person = self.peopleById[personId]
        self._showPersonDetails(person)

    def _onNewPerson(self, event: wx.CommandEvent) -> None:
        if self.onRequestAddPerson is not None:
            self.onRequestAddPerson()

    def _onRefresh(self, event: wx.CommandEvent) -> None:
        self.refreshPeople()

    def _onAddImages(self, event: wx.CommandEvent) -> None:
        personId = self.getSelectedPersonId()
        if personId is None:
            wx.MessageBox("Select a person before adding images.", "People")
            return

        imagePaths = promptForImagePaths(
            self,
            message="Choose one or more images to add to this person",
        )
        if imagePaths is None:
            return

        faceSamples = collectFaceSamples(self, self.faceRecognitionService, imagePaths)
        if faceSamples is None:
            return

        if not faceSamples:
            wx.MessageBox("No face samples were selected, so nothing was saved.", "People")
            return

        try:
            addedCount = self.personRepository.addFaceSamples(personId, faceSamples)
        except (sqlite3.Error, ValueError) as error:
            showRepositoryWriteError(self, "save the new face samples", error)
            return

        self.refreshPeople(personId)
        wx.MessageBox(f"Saved {addedCount} new face sample(s).", "People")

    def _showEmptyState(self) -> None:
        self.personNameLabel.SetLabel("No people saved yet")
        self.personSummaryLabel.SetLabel("Create a person from the People menu or the New Person button.")
        styleText(self.personNameLabel, "page_title")
        styleText(self.personSummaryLabel, "muted")
        self.addImagesButton.Enable(False)
        self._renderFaceSamples([])

    def _showPersonDetails(self, person: PersonSummary) -> None:
        self.personNameLabel.SetLabel(person.name)
        self.personSummaryLabel.SetLabel(
            f"{person.sampleCount} saved sample(s). Updated: {person.updatedAt}"
        )
        styleText(self.personNameLabel, "page_title")
        styleText(self.personSummaryLabel, "muted")
        self.addImagesButton.Enable(True)
        faceSamples = self.personRepository.listFaceSamplesForPerson(person.id)
        self._renderFaceSamples(faceSamples)
        self.Layout()

    def _renderFaceSamples(self, faceSamples: list[FaceSample]) -> None:
        self.thumbnailSizer.Clear(delete_windows=True)

        if not faceSamples:
            emptyLabel = wx.StaticText(
                self.thumbnailScroller,
                label="No saved face images for this person yet.",
            )
            styleText(emptyLabel, "muted")
            self.thumbnailSizer.Add(emptyLabel, 0, wx.ALL, 12)
        else:
            for faceSample in faceSamples:
                thumbnailPanel = self._buildThumbnailPanel(faceSample)
                self.thumbnailSizer.Add(thumbnailPanel, 0, wx.ALL, 8)

        self.thumbnailScroller.Layout()
        self.thumbnailScroller.FitInside()

    def _buildThumbnailPanel(self, faceSample: FaceSample) -> wx.Panel:
        panel = CardPanel(self.thumbnailScroller, background=SURFACE_BG_ALT, borderColour=BORDER)
        panel.SetMinSize((200, 236))
        panelSizer = wx.BoxSizer(wx.VERTICAL)
        contextMenuWidgets: list[wx.Window] = [panel]

        thumbnailBitmap = bitmapFromEncodedImage(faceSample.faceImage, maxWidth=160, maxHeight=160)
        if thumbnailBitmap is not None:
            imageWidget = wx.StaticBitmap(panel, bitmap=thumbnailBitmap)
            imageWidget.SetToolTip(faceSample.imagePath)
            panelSizer.Add(imageWidget, 0, wx.BOTTOM | wx.ALIGN_CENTER_HORIZONTAL, 8)
            contextMenuWidgets.append(imageWidget)
        else:
            placeholderLabel = wx.StaticText(panel, label="Image preview unavailable")
            styleText(placeholderLabel, "muted")
            panelSizer.Add(
                placeholderLabel,
                0,
                wx.BOTTOM | wx.ALIGN_CENTER_HORIZONTAL,
                8,
            )
            contextMenuWidgets.append(placeholderLabel)

        sourceLabel = wx.StaticText(panel, label=Path(faceSample.imagePath).name)
        createdLabel = wx.StaticText(panel, label=faceSample.createdAt)
        styleText(sourceLabel, "body")
        styleText(createdLabel, "muted")
        sourceLabel.Wrap(160)
        createdLabel.Wrap(160)
        sourceLabel.SetToolTip(faceSample.imagePath)
        panelSizer.Add(sourceLabel, 0, wx.BOTTOM, 4)
        panelSizer.Add(createdLabel, 0)
        panelOuterSizer = wx.BoxSizer(wx.VERTICAL)
        panelOuterSizer.Add(panelSizer, 1, wx.ALL | wx.EXPAND, 16)
        panel.SetSizer(panelOuterSizer)
        contextMenuWidgets.extend((sourceLabel, createdLabel))
        self._bindFaceSampleContextMenu(contextMenuWidgets, faceSample)
        return panel

    def _bindFaceSampleContextMenu(
        self,
        widgets: list[wx.Window],
        faceSample: FaceSample,
    ) -> None:
        for widget in widgets:
            widget.Bind(
                wx.EVT_CONTEXT_MENU,
                lambda event, sample=faceSample: self._showFaceSampleContextMenu(event, sample),
            )

    def _showFaceSampleContextMenu(
        self,
        event: wx.ContextMenuEvent,
        faceSample: FaceSample,
    ) -> None:
        menu = wx.Menu()
        viewOriginalItem = menu.Append(wx.ID_ANY, "View Original Image")
        canViewOriginal = (
            faceSample.hasOriginalImage
            or Path(faceSample.imagePath).is_file()
        )
        viewOriginalItem.Enable(canViewOriginal)
        if canViewOriginal:
            menu.Bind(
                wx.EVT_MENU,
                lambda _: self._showOriginalImage(faceSample),
                viewOriginalItem,
            )

        eventWindow = event.GetEventObject()
        if isinstance(eventWindow, wx.Window):
            eventWindow.PopupMenu(menu)
        else:
            self.PopupMenu(menu)
        menu.Destroy()

    def _showOriginalImage(self, faceSample: FaceSample) -> None:
        originalImageBytes = (
            self.personRepository.getStoredOriginalImageForFaceSample(faceSample.id)
            if faceSample.hasOriginalImage
            else None
        )
        if originalImageBytes is None and not Path(faceSample.imagePath).is_file():
            wx.MessageBox(
                "The original image is not available for this sample.",
                "People",
            )
            return

        frame = OriginalImageFrame(
            self.GetTopLevelParent(),
            imagePath=faceSample.imagePath,
            imageBytes=originalImageBytes,
        )
        frame.Bind(wx.EVT_CLOSE, self._onOriginalImageFrameClosed)
        self.originalImageFrames.append(frame)
        frame.Show()

    def _onOriginalImageFrameClosed(self, event: wx.CloseEvent) -> None:
        eventWindow = event.GetEventObject()
        if isinstance(eventWindow, OriginalImageFrame):
            self.originalImageFrames = [
                frame
                for frame in self.originalImageFrames
                if frame is not eventWindow
            ]

        event.Skip()

    def _selectPersonById(self, personId: int) -> bool:
        for rowIndex, rowPersonId in enumerate(self.rowPersonIds):
            if rowPersonId != personId:
                continue

            self.peopleList.SetItemState(
                rowIndex,
                wx.LIST_STATE_SELECTED | wx.LIST_STATE_FOCUSED,
                wx.LIST_STATE_SELECTED | wx.LIST_STATE_FOCUSED,
            )
            self.peopleList.EnsureVisible(rowIndex)
            self._showPersonDetails(self.peopleById[personId])
            return True

        return False


class AddPersonPanel(wx.Panel):
    def __init__(
        self,
        parent: wx.Window,
        personRepository: PersonRepository,
        faceRecognitionService: FaceRecognitionService,
        onRequestShowPeople: Callable[[], None] | None = None,
        onRequestShowFolder: Callable[[], None] | None = None,
        onPersonSaved: Callable[[Person], None] | None = None,
    ):
        super().__init__(parent)
        stylePanel(self)
        self.personRepository = personRepository
        self.faceRecognitionService = faceRecognitionService
        self.onPersonSaved = onPersonSaved
        self.selectedImagePaths: list[str] = []

        headerCard = CardPanel(self, background=ACCENT)
        titleEyebrow = wx.StaticText(headerCard, label="People")
        titleLabel = wx.StaticText(headerCard, label="Create a New Person")
        helpLabel = wx.StaticText(
            headerCard,
            label=(
                "Choose images where the person is visible. For each image, you will confirm "
                "or draw the face rectangle before the sample is saved."
            ),
        )
        styleText(titleEyebrow, "eyebrow")
        styleText(titleLabel, "app_title")
        styleText(helpLabel, "muted")
        titleEyebrow.SetForegroundColour(wx.Colour(12, 17, 25))
        titleLabel.SetForegroundColour(wx.Colour(12, 17, 25))
        helpLabel.SetForegroundColour(wx.Colour(20, 28, 40))
        helpLabel.Wrap(880)
        headerTextSizer = wx.BoxSizer(wx.VERTICAL)
        headerTextSizer.Add(titleEyebrow, 0, wx.BOTTOM, 6)
        headerTextSizer.Add(titleLabel, 0, wx.BOTTOM, 6)
        headerTextSizer.Add(helpLabel, 0)
        headerNavSizer = buildWorkspaceNavigationSizer(
            headerCard,
            activeView="people",
            onRequestShowPeople=onRequestShowPeople,
            onRequestShowFolder=onRequestShowFolder,
        )
        headerSizer = wx.BoxSizer(wx.HORIZONTAL)
        headerSizer.Add(headerTextSizer, 1, wx.EXPAND)
        headerSizer.Add(headerNavSizer, 0, wx.ALIGN_CENTER_VERTICAL)
        headerOuterSizer = wx.BoxSizer(wx.VERTICAL)
        headerOuterSizer.Add(headerSizer, 1, wx.ALL | wx.EXPAND, 22)
        headerCard.SetSizer(headerOuterSizer)

        formCard = CardPanel(self)
        nameLabel = wx.StaticText(formCard, label="Person Name")
        self.nameCtrl = wx.TextCtrl(formCard)
        queueTitle = wx.StaticText(formCard, label="Queued Images")
        self.imageCountLabel = wx.StaticText(formCard, label="No images selected.")
        self.imageList = wx.ListBox(formCard, style=wx.LB_EXTENDED)
        self.imageList.SetMinSize((640, 380))
        styleText(nameLabel, "section_title")
        styleText(queueTitle, "section_title")
        styleText(self.imageCountLabel, "muted")
        styleTextCtrl(self.nameCtrl)
        styleListBox(self.imageList)

        addImagesButton = wx.Button(formCard, label="Add Images")
        removeImagesButton = wx.Button(formCard, label="Remove Selected")
        clearImagesButton = wx.Button(formCard, label="Clear List")
        savePersonButton = wx.Button(formCard, label="Save Person")

        addImagesButton.Bind(wx.EVT_BUTTON, self._onAddImages)
        removeImagesButton.Bind(wx.EVT_BUTTON, self._onRemoveSelectedImages)
        clearImagesButton.Bind(wx.EVT_BUTTON, self._onClearImages)
        savePersonButton.Bind(wx.EVT_BUTTON, self._onSavePerson)
        styleButton(addImagesButton, "secondary")
        styleButton(removeImagesButton, "subtle")
        styleButton(clearImagesButton, "subtle")
        styleButton(savePersonButton, "primary")

        buttonSizer = wx.BoxSizer(wx.HORIZONTAL)
        buttonSizer.Add(addImagesButton, 0, wx.RIGHT, 8)
        buttonSizer.Add(removeImagesButton, 0, wx.RIGHT, 8)
        buttonSizer.Add(clearImagesButton, 0, wx.RIGHT, 8)
        buttonSizer.Add(savePersonButton, 0)

        formSizer = wx.BoxSizer(wx.VERTICAL)
        formSizer.Add(nameLabel, 0, wx.BOTTOM, 8)
        formSizer.Add(self.nameCtrl, 0, wx.BOTTOM | wx.EXPAND, 18)
        formSizer.Add(queueTitle, 0, wx.BOTTOM, 8)
        formSizer.Add(self.imageCountLabel, 0, wx.BOTTOM, 10)
        formSizer.Add(self.imageList, 1, wx.BOTTOM | wx.EXPAND, 14)
        formSizer.Add(buttonSizer, 0)
        formOuterSizer = wx.BoxSizer(wx.VERTICAL)
        formOuterSizer.Add(formSizer, 1, wx.ALL | wx.EXPAND, 22)
        formCard.SetSizer(formOuterSizer)

        guideCard = CardPanel(self)
        guideEyebrow = wx.StaticText(guideCard, label="Workflow")
        guideTitle = wx.StaticText(guideCard, label="How Person Creation Works")
        stepsLabel = wx.StaticText(
            guideCard,
            label=(
                "1. Add one or more supported images.\n"
                "2. Enter the person name.\n"
                "3. Confirm or draw the face rectangle for each image.\n"
                "4. Save the profile and return to the library."
            ),
        )
        formatLabel = wx.StaticText(
            guideCard,
            label=f"Supported formats: {', '.join(SUPPORTED_IMAGE_EXTENSIONS)}",
        )
        hintLabel = wx.StaticText(
            guideCard,
            label="Use a few images from different angles or lighting conditions for better matching later.",
        )
        styleText(guideEyebrow, "eyebrow")
        styleText(guideTitle, "section_title")
        styleText(stepsLabel, "body")
        styleText(formatLabel, "success")
        styleText(hintLabel, "muted")
        stepsLabel.Wrap(320)
        formatLabel.Wrap(320)
        hintLabel.Wrap(320)
        guideSizer = wx.BoxSizer(wx.VERTICAL)
        guideSizer.Add(guideEyebrow, 0, wx.BOTTOM, 6)
        guideSizer.Add(guideTitle, 0, wx.BOTTOM, 12)
        guideSizer.Add(stepsLabel, 0, wx.BOTTOM, 14)
        guideSizer.Add(formatLabel, 0, wx.BOTTOM, 10)
        guideSizer.Add(hintLabel, 0)
        guideOuterSizer = wx.BoxSizer(wx.VERTICAL)
        guideOuterSizer.Add(guideSizer, 1, wx.ALL | wx.EXPAND, 22)
        guideCard.SetSizer(guideOuterSizer)

        contentSizer = wx.BoxSizer(wx.HORIZONTAL)
        contentSizer.Add(formCard, 1, wx.RIGHT | wx.EXPAND, 18)
        contentSizer.Add(guideCard, 0, wx.EXPAND)

        rootSizer = wx.BoxSizer(wx.VERTICAL)
        rootSizer.Add(headerCard, 0, wx.ALL | wx.EXPAND, 4)
        rootSizer.Add(contentSizer, 1, wx.TOP | wx.BOTTOM | wx.EXPAND, 18)
        self.SetSizer(rootSizer)

    def _onAddImages(self, event: wx.CommandEvent) -> None:
        imagePaths = promptForImagePaths(
            self,
            message="Choose one or more images for this person",
        )
        if imagePaths is None:
            return

        existingPaths = set(self.selectedImagePaths)
        for imagePath in imagePaths:
            if imagePath not in existingPaths:
                self.selectedImagePaths.append(imagePath)
                existingPaths.add(imagePath)

        self._refreshImageList()

    def _onRemoveSelectedImages(self, event: wx.CommandEvent) -> None:
        selectedIndices = list(self.imageList.GetSelections())
        if not selectedIndices:
            return

        for selectedIndex in sorted(selectedIndices, reverse=True):
            del self.selectedImagePaths[selectedIndex]

        self._refreshImageList()

    def _onClearImages(self, event: wx.CommandEvent) -> None:
        self.selectedImagePaths.clear()
        self._refreshImageList()

    def _onSavePerson(self, event: wx.CommandEvent) -> None:
        personName = self.nameCtrl.GetValue().strip()
        if not personName:
            wx.MessageBox("Enter a person name before saving.", "People")
            return

        if not self.selectedImagePaths:
            wx.MessageBox("Add at least one image before saving.", "People")
            return

        faceSamples = collectFaceSamples(self, self.faceRecognitionService, self.selectedImagePaths)
        if faceSamples is None:
            return

        if not faceSamples:
            wx.MessageBox("No face samples were selected, so nothing was saved.", "People")
            return

        try:
            person = self.personRepository.createPersonWithSamples(personName, faceSamples)
        except sqlite3.IntegrityError:
            wx.MessageBox(f'A person named "{personName}" already exists.', "People")
            return
        except (sqlite3.Error, ValueError) as error:
            showRepositoryWriteError(self, "save the person", error)
            return

        self.nameCtrl.SetValue("")
        self.selectedImagePaths.clear()
        self._refreshImageList()
        wx.MessageBox(
            f'Saved "{person.name}" with {len(faceSamples)} face sample(s).',
            "People",
        )

        if self.onPersonSaved is not None:
            self.onPersonSaved(person)

    def _refreshImageList(self) -> None:
        self.imageList.Set(self.selectedImagePaths)
        if not self.selectedImagePaths:
            self.imageCountLabel.SetLabel("No images selected.")
            styleText(self.imageCountLabel, "muted")
            return

        self.imageCountLabel.SetLabel(f"{len(self.selectedImagePaths)} image(s) queued for this person.")
        styleText(self.imageCountLabel, "muted")
