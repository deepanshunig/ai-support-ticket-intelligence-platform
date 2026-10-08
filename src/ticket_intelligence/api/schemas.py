"""Bounded requests for ticket creation and authenticated human review."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class TicketInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=2000, strict=True)

    @field_validator("text", mode="before")
    @classmethod
    def trim_text(cls, value):
        return value.strip() if isinstance(value, str) else value


class ReviewInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected_version: int = Field(ge=1, strict=True)
    category: str = Field(min_length=1, max_length=100, strict=True)
    status: Literal["assigned", "resolved"]
    note: str = Field(min_length=1, max_length=1000, strict=True)

    @field_validator("category", "note", mode="before")
    @classmethod
    def trim_fields(cls, value):
        return value.strip() if isinstance(value, str) else value
