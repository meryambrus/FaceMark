from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import wx


class MenuItemAction(ABC):
    item_id = wx.ID_ANY
    item_label = ""

    def __init__(self, window: wx.Frame):
        self.window = window

    def addToMenu(self, menu: wx.Menu) -> wx.MenuItem:
        menuItem = menu.Append(self.item_id, self.item_label)
        self.window.Bind(wx.EVT_MENU, self.handle, menuItem)
        return menuItem

    @abstractmethod
    def handle(self, event: wx.CommandEvent) -> None:
        pass


class CallbackMenuItem(MenuItemAction):
    def __init__(
        self,
        window: wx.Frame,
        callback: Callable[[], None],
        itemLabel: str,
        itemId: int = wx.ID_ANY,
    ):
        super().__init__(window)
        self.callback = callback
        self.item_label = itemLabel
        self.item_id = itemId

    def handle(self, event: wx.CommandEvent) -> None:
        self.callback()
