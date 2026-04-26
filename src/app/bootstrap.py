from __future__ import annotations

import argparse
import ctypes
import os
import sys

from src.ui.ui_registry import (
    createApplication,
    getAvailableUiDefinitions,
    getDefaultUiId,
)


def configureWindowsDpiAwareness() -> None:
    if not sys.platform.startswith("win"):
        return

    try:
        shcore = ctypes.windll.shcore
        if shcore.SetProcessDpiAwareness(2) == 0:
            return
    except (AttributeError, OSError):
        pass

    try:
        user32 = ctypes.windll.user32
        if user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-3)):
            return
    except (AttributeError, OSError):
        pass

    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except (AttributeError, OSError):
        pass


def buildArgumentParser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch FaceMark.")
    parser.add_argument(
        "--ui",
        dest="uiId",
        help=(
            "UI implementation to launch. "
            f"Defaults to FACEMARK_UI or {getDefaultUiId()}."
        ),
    )
    parser.add_argument(
        "--list-uis",
        action="store_true",
        help="List available UI implementations and exit.",
    )
    return parser


def resolveRequestedUiId(argumentUiId: str | None) -> str:
    if argumentUiId:
        return argumentUiId

    return os.environ.get("FACEMARK_UI") or getDefaultUiId()


def printAvailableUis() -> None:
    print("Available UIs:")
    for uiDefinition in getAvailableUiDefinitions():
        aliasSuffix = ""
        if uiDefinition.aliases:
            aliasSuffix = f" (aliases: {', '.join(uiDefinition.aliases)})"

        print(f"- {uiDefinition.uiId}: {uiDefinition.description}{aliasSuffix}")


def main(argv: list[str] | None = None) -> int:
    configureWindowsDpiAwareness()
    arguments = buildArgumentParser().parse_args(sys.argv[1:] if argv is None else argv)
    if arguments.list_uis:
        printAvailableUis()
        return 0

    requestedUiId = resolveRequestedUiId(arguments.uiId)
    try:
        createApplication(
            uiId=requestedUiId,
            title="FaceMark",
        )
    except ValueError as error:
        print(error, file=sys.stderr)
        return 2

    return 0
