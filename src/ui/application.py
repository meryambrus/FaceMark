import wx

from src.app.face.facerecognition import FaceRecognitionService
from src.data.store import PersonRepository
from src.ui.folder_workspace import FolderWorkspace
from src.ui.menubar import MenuBarBuilder
from src.ui.people_workspace import PeopleWorkspace
from src.ui.theme import styleFrame, stylePanel


class Application:
    def __init__(self, title="FaceMark", size=(1600, 900)):
        self.app = wx.App()
        self.window = wx.Frame(None, title=title, size=size)
        self.panel = wx.Panel(self.window)
        self.personRepository = PersonRepository()
        self.faceRecognitionService = FaceRecognitionService()
        self.mainBook = wx.Simplebook(self.panel)
        self.peopleWorkspace = PeopleWorkspace(
            self.mainBook,
            personRepository=self.personRepository,
            faceRecognitionService=self.faceRecognitionService,
            onRequestShowFolder=self.showFolderWorkspace,
        )
        self.folderWorkspace = FolderWorkspace(
            self.mainBook,
            personRepository=self.personRepository,
            faceRecognitionService=self.faceRecognitionService,
            onRequestShowPeople=self.showAllPeople,
            onRequestOpenFolder=self.openFolder,
        )

        self.configureTheme()
        self.buildLayout()
        self.buildMenuBar()

        self.window.Show(True)
        self.app.MainLoop()

    def configureTheme(self) -> None:
        styleFrame(self.window)
        stylePanel(self.panel)
        self.mainBook.SetBackgroundColour(self.panel.GetBackgroundColour())

    def buildLayout(self) -> None:
        self.mainBook.AddPage(self.peopleWorkspace, "People")
        self.mainBook.AddPage(self.folderWorkspace, "Folder")
        self.mainBook.SetSelection(0)

        rootSizer = wx.BoxSizer(wx.VERTICAL)
        rootSizer.Add(self.mainBook, 1, wx.ALL | wx.EXPAND, 18)
        self.panel.SetSizer(rootSizer)
        frameSizer = wx.BoxSizer(wx.VERTICAL)
        frameSizer.Add(self.panel, 1, wx.EXPAND)
        self.window.SetSizer(frameSizer)

    def buildMenuBar(self) -> None:
        menuBar = MenuBarBuilder(
            self.window,
            openFolder=self.openFolder,
            exitApplication=self.exitApplication,
            showAllPeople=self.showAllPeople,
            showAddPerson=self.showAddPerson,
        ).build()
        self.window.SetMenuBar(menuBar)

    def showAllPeople(self) -> None:
        self.mainBook.SetSelection(0)
        self.peopleWorkspace.showAllPeople()

    def showAddPerson(self) -> None:
        self.mainBook.SetSelection(0)
        self.peopleWorkspace.showAddPerson()

    def openFolder(self) -> None:
        with wx.DirDialog(self.window, "Choose a folder") as dialog:
            if dialog.ShowModal() != wx.ID_OK:
                return

            folderPath = dialog.GetPath()

        self.folderWorkspace.loadFolder(folderPath)
        self.mainBook.SetSelection(1)

    def exitApplication(self) -> None:
        self.window.Close(True)

    def showFolderWorkspace(self) -> None:
        self.mainBook.SetSelection(1)
