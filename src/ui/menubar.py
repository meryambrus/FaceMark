from typing import Callable

import wx

from src.ui.menu_items import CallbackMenuItem


class MenuBarBuilder:
    def __init__(
        self,
        window: wx.Frame,
        openFolder: Callable[[], None],
        exitApplication: Callable[[], None],
        showAllPeople: Callable[[], None],
        showAddPerson: Callable[[], None],
    ):
        self.window = window
        self.openFolder = openFolder
        self.exitApplication = exitApplication
        self.showAllPeople = showAllPeople
        self.showAddPerson = showAddPerson

    def build(self) -> wx.MenuBar:
        menuBar = wx.MenuBar()

        fileMenu = wx.Menu()
        fileMenuItems = [
            CallbackMenuItem(
                self.window,
                self.openFolder,
                "Open Folder",
                itemId=wx.ID_OPEN,
            ),
            CallbackMenuItem(
                self.window,
                self.exitApplication,
                "Exit",
                itemId=wx.ID_EXIT,
            ),
        ]
        for index, menuItem in enumerate(fileMenuItems):
            if index == 1:
                fileMenu.AppendSeparator()

            menuItem.addToMenu(fileMenu)

        peopleMenu = wx.Menu()
        peopleMenuItems = [
            CallbackMenuItem(self.window, self.showAllPeople, "All People"),
            CallbackMenuItem(self.window, self.showAddPerson, "Add new Person"),
        ]
        for menuItem in peopleMenuItems:
            menuItem.addToMenu(peopleMenu)

        menuBar.Append(fileMenu, "&File")
        menuBar.Append(peopleMenu, "&People")
        return menuBar
