from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import wx


@dataclass(slots=True)
class PreparedBitmapData:
    width: int
    height: int
    rgbBytes: bytes


def _targetBitmapSize(
    imageWidth: int,
    imageHeight: int,
    maxWidth: int,
    maxHeight: int,
) -> tuple[int, int]:
    scale = min(maxWidth / imageWidth, maxHeight / imageHeight, 1.0)
    targetWidth = max(1, int(imageWidth * scale))
    targetHeight = max(1, int(imageHeight * scale))
    return targetWidth, targetHeight


def bitmapFromEncodedImage(
    imageBytes: bytes | None,
    maxWidth: int,
    maxHeight: int,
) -> wx.Bitmap | None:
    if not imageBytes:
        return None

    imageBuffer = np.frombuffer(imageBytes, dtype=np.uint8)
    imageBgr = cv2.imdecode(imageBuffer, cv2.IMREAD_COLOR)
    if imageBgr is None:
        return None

    return bitmapFromBgr(imageBgr, maxWidth=maxWidth, maxHeight=maxHeight)


def bitmapFromImagePath(
    imagePath: str | Path,
    maxWidth: int,
    maxHeight: int,
) -> wx.Bitmap | None:
    preparedBitmap = preparedBitmapFromImagePath(
        imagePath,
        maxWidth=maxWidth,
        maxHeight=maxHeight,
    )
    if preparedBitmap is None:
        return None

    return bitmapFromPreparedBitmap(preparedBitmap)


def preparedBitmapFromImagePath(
    imagePath: str | Path,
    maxWidth: int,
    maxHeight: int,
) -> PreparedBitmapData | None:
    imageBgr = cv2.imread(str(imagePath))
    if imageBgr is None:
        return None

    return preparedBitmapFromBgr(imageBgr, maxWidth=maxWidth, maxHeight=maxHeight)


def bitmapFromBgr(
    imageBgr: np.ndarray,
    maxWidth: int,
    maxHeight: int,
) -> wx.Bitmap:
    preparedBitmap = preparedBitmapFromBgr(
        imageBgr,
        maxWidth=maxWidth,
        maxHeight=maxHeight,
    )
    return bitmapFromPreparedBitmap(preparedBitmap)


def preparedBitmapFromBgr(
    imageBgr: np.ndarray,
    maxWidth: int,
    maxHeight: int,
) -> PreparedBitmapData:
    imageHeight, imageWidth = imageBgr.shape[:2]
    targetWidth, targetHeight = _targetBitmapSize(
        imageWidth,
        imageHeight,
        maxWidth,
        maxHeight,
    )
    resizedImageBgr = imageBgr
    if targetWidth != imageWidth or targetHeight != imageHeight:
        resizedImageBgr = cv2.resize(
            imageBgr,
            (targetWidth, targetHeight),
            interpolation=cv2.INTER_AREA,
        )

    imageRgb = cv2.cvtColor(resizedImageBgr, cv2.COLOR_BGR2RGB)
    return PreparedBitmapData(
        width=targetWidth,
        height=targetHeight,
        rgbBytes=imageRgb.tobytes(),
    )


def bitmapFromPreparedBitmap(preparedBitmap: PreparedBitmapData) -> wx.Bitmap:
    wxImage = wx.Image(preparedBitmap.width, preparedBitmap.height)
    wxImage.SetData(preparedBitmap.rgbBytes)
    return wx.Bitmap(wxImage)


def scalePreparedBitmap(
    preparedBitmap: PreparedBitmapData,
    maxWidth: int,
    maxHeight: int,
) -> PreparedBitmapData:
    targetWidth, targetHeight = _targetBitmapSize(
        preparedBitmap.width,
        preparedBitmap.height,
        maxWidth,
        maxHeight,
    )
    if targetWidth == preparedBitmap.width and targetHeight == preparedBitmap.height:
        return preparedBitmap

    imageRgb = np.frombuffer(preparedBitmap.rgbBytes, dtype=np.uint8).reshape(
        (preparedBitmap.height, preparedBitmap.width, 3)
    )
    resizedImageRgb = cv2.resize(
        imageRgb,
        (targetWidth, targetHeight),
        interpolation=cv2.INTER_AREA,
    )
    return PreparedBitmapData(
        width=targetWidth,
        height=targetHeight,
        rgbBytes=resizedImageRgb.tobytes(),
    )


def bitmapFromPreparedBitmapSized(
    preparedBitmap: PreparedBitmapData,
    maxWidth: int,
    maxHeight: int,
) -> wx.Bitmap:
    return bitmapFromPreparedBitmap(
        scalePreparedBitmap(
            preparedBitmap,
            maxWidth=maxWidth,
            maxHeight=maxHeight,
        )
    )
