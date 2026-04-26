from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from src.ui.application import Application


type WindowSize = tuple[int, int]


class UiFactory(Protocol):
    def __call__(
        self,
        title: str = "FaceMark",
        size: WindowSize = (1600, 900),
    ) -> object: ...


@dataclass(frozen=True, slots=True)
class UiDefinition:
    uiId: str
    label: str
    description: str
    applicationFactory: UiFactory
    aliases: tuple[str, ...] = ()


uiDefinitions = (
    UiDefinition(
        uiId="legacy",
        label="Legacy UI",
        description="The current FaceMark interface.",
        applicationFactory=Application,
        aliases=("current", "classic"),
    ),
)

uiDefinitionsByName = {
    uiName.casefold(): uiDefinition
    for uiDefinition in uiDefinitions
    for uiName in (uiDefinition.uiId, *uiDefinition.aliases)
}


def getAvailableUiDefinitions() -> tuple[UiDefinition, ...]:
    return uiDefinitions


def getDefaultUiId() -> str:
    return uiDefinitions[0].uiId


def getUiDefinition(uiId: str | None) -> UiDefinition:
    requestedUiId = getDefaultUiId() if uiId is None else uiId.strip()
    if not requestedUiId:
        requestedUiId = getDefaultUiId()

    uiDefinition = uiDefinitionsByName.get(requestedUiId.casefold())
    if uiDefinition is not None:
        return uiDefinition

    availableUiIds = ", ".join(
        uiDefinition.uiId
        for uiDefinition in uiDefinitions
    )
    raise ValueError(
        f'Unknown UI "{requestedUiId}". Available UIs: {availableUiIds}.'
    )


def createApplication(
    uiId: str | None = None,
    title: str = "FaceMark",
    size: WindowSize = (1600, 900),
) -> object:
    uiDefinition = getUiDefinition(uiId)
    return uiDefinition.applicationFactory(title=title, size=size)
