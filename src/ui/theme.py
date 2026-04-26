from __future__ import annotations

import wx


WINDOW_BG = wx.Colour(14, 18, 27)
SURFACE_BG = wx.Colour(24, 30, 41)
SURFACE_BG_ALT = wx.Colour(31, 38, 52)
SURFACE_BG_SUBTLE = wx.Colour(19, 24, 34)
BORDER = wx.Colour(54, 64, 82)
TEXT_PRIMARY = wx.Colour(241, 245, 252)
TEXT_MUTED = wx.Colour(153, 164, 184)
ACCENT = wx.Colour(96, 165, 250)
SUCCESS = wx.Colour(74, 222, 128)
WARNING = wx.Colour(250, 204, 21)


class CardPanel(wx.Panel):
    def __init__(
        self,
        parent: wx.Window,
        background: wx.Colour = SURFACE_BG,
        borderColour: wx.Colour = BORDER,
        radius: int = 16,
    ):
        super().__init__(
            parent,
            style=wx.TAB_TRAVERSAL | wx.CLIP_CHILDREN | wx.FULL_REPAINT_ON_RESIZE,
        )
        self.background = background
        self.borderColour = borderColour
        self.radius = radius
        self.SetBackgroundStyle(wx.BG_STYLE_PAINT)
        self.SetDoubleBuffered(True)
        self.SetBackgroundColour(background)
        self.Bind(wx.EVT_PAINT, self._onPaint)
        self.Bind(wx.EVT_SIZE, self._onSize)

    def _onPaint(self, event: wx.PaintEvent) -> None:
        paintDc = wx.AutoBufferedPaintDC(self)
        paintDc.SetBackground(wx.Brush(self.GetParent().GetBackgroundColour()))
        paintDc.Clear()

        rect = self.GetClientRect()
        if rect.width <= 0 or rect.height <= 0:
            return

        paintDc.SetPen(wx.Pen(self.borderColour, 1))
        paintDc.SetBrush(wx.Brush(self.background))
        paintDc.DrawRoundedRectangle(rect.x, rect.y, rect.width - 1, rect.height - 1, self.radius)

    def _onSize(self, event: wx.SizeEvent) -> None:
        self.Refresh()
        event.Skip()


def styleFrame(frame: wx.Frame) -> None:
    frame.SetDoubleBuffered(True)
    frame.SetBackgroundColour(WINDOW_BG)
    frame.SetForegroundColour(TEXT_PRIMARY)


def stylePanel(window: wx.Window, background: wx.Colour = WINDOW_BG) -> None:
    window.SetBackgroundStyle(wx.BG_STYLE_SYSTEM)
    window.SetDoubleBuffered(True)
    window.SetBackgroundColour(background)
    window.SetForegroundColour(TEXT_PRIMARY)


def styleDialog(dialog: wx.Dialog) -> None:
    dialog.SetDoubleBuffered(True)
    dialog.SetBackgroundColour(WINDOW_BG)
    dialog.SetForegroundColour(TEXT_PRIMARY)


def styleText(label: wx.StaticText, role: str = "body") -> None:
    font = label.GetFont()

    if role == "eyebrow":
        font.SetPointSize(9)
        font.SetWeight(wx.FONTWEIGHT_BOLD)
        label.SetForegroundColour(ACCENT)
    elif role == "app_title":
        font.SetPointSize(18)
        font.SetWeight(wx.FONTWEIGHT_BOLD)
        label.SetForegroundColour(TEXT_PRIMARY)
    elif role == "page_title":
        font.SetPointSize(15)
        font.SetWeight(wx.FONTWEIGHT_BOLD)
        label.SetForegroundColour(TEXT_PRIMARY)
    elif role == "section_title":
        font.SetPointSize(12)
        font.SetWeight(wx.FONTWEIGHT_BOLD)
        label.SetForegroundColour(TEXT_PRIMARY)
    elif role == "muted":
        label.SetForegroundColour(TEXT_MUTED)
    elif role == "success":
        font.SetWeight(wx.FONTWEIGHT_BOLD)
        label.SetForegroundColour(SUCCESS)
    elif role == "warning":
        font.SetWeight(wx.FONTWEIGHT_BOLD)
        label.SetForegroundColour(WARNING)
    else:
        label.SetForegroundColour(TEXT_PRIMARY)

    label.SetFont(font)


def styleButton(button: wx.Button, variant: str = "secondary") -> None:
    font = button.GetFont()
    font.SetWeight(wx.FONTWEIGHT_BOLD)
    button.SetFont(font)
    button.SetForegroundColour(TEXT_PRIMARY)
    button.SetCursor(wx.Cursor(wx.CURSOR_HAND))
    button.SetMinSize((132, 38))

    if variant == "primary":
        button.SetBackgroundColour(ACCENT)
        button.SetForegroundColour(wx.Colour(12, 17, 25))
    elif variant == "nav_active":
        button.SetBackgroundColour(SURFACE_BG)
        button.SetForegroundColour(ACCENT)
    elif variant == "nav":
        button.SetBackgroundColour(SURFACE_BG_SUBTLE)
        button.SetForegroundColour(TEXT_MUTED)
    elif variant == "subtle":
        button.SetBackgroundColour(SURFACE_BG_SUBTLE)
        button.SetForegroundColour(TEXT_MUTED)
    else:
        button.SetBackgroundColour(SURFACE_BG_ALT)
        button.SetForegroundColour(TEXT_PRIMARY)


def styleTextCtrl(textCtrl: wx.TextCtrl) -> None:
    textCtrl.SetBackgroundColour(SURFACE_BG_SUBTLE)
    textCtrl.SetForegroundColour(TEXT_PRIMARY)


def styleListBox(listBox: wx.ListBox) -> None:
    listBox.SetBackgroundColour(SURFACE_BG_SUBTLE)
    listBox.SetForegroundColour(TEXT_PRIMARY)


def styleListCtrl(listCtrl: wx.ListCtrl) -> None:
    listCtrl.SetBackgroundColour(SURFACE_BG_SUBTLE)
    listCtrl.SetForegroundColour(TEXT_PRIMARY)
    listCtrl.SetTextColour(TEXT_PRIMARY)


def styleScrolledWindow(
    scrolledWindow: wx.ScrolledWindow,
    background: wx.Colour = SURFACE_BG_SUBTLE,
) -> None:
    scrolledWindow.SetBackgroundStyle(wx.BG_STYLE_SYSTEM)
    scrolledWindow.SetDoubleBuffered(True)
    scrolledWindow.SetBackgroundColour(background)
    scrolledWindow.SetForegroundColour(TEXT_PRIMARY)


def applyAlternatingRowColour(listCtrl: wx.ListCtrl, rowIndex: int) -> None:
    rowColour = SURFACE_BG_SUBTLE if rowIndex % 2 == 0 else SURFACE_BG
    listCtrl.SetItemBackgroundColour(rowIndex, rowColour)
    listCtrl.SetItemTextColour(rowIndex, TEXT_PRIMARY)
