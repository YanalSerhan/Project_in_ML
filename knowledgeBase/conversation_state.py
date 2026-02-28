from pydantic import BaseModel, Field
from typing import Optional, List


class ConversationState(BaseModel):
    """
    Tracks ALL mentioned entities.
    """
    courses: List[str] = Field(default_factory=list)
    lecturers: List[str] = Field(default_factory=list)
    years: List[int] = Field(default_factory=list)
    semesters: List[str] = Field(default_factory=list)

    def update(self, extracted: dict):
        """
        Slot filling logic for multi-entity:
        - If the extractor outputs a list, append NEW values.
        - Avoid duplicates.
        """
        if extracted.get("course"):
            for c in extracted["course"]:
                if c not in self.courses:
                    self.courses.append(c)

        if extracted.get("lecturer"):
            for l in extracted["lecturer"]:
                if l not in self.lecturers:
                    self.lecturers.append(l)

        if extracted.get("years"):
            for y in extracted["years"]:
                if y not in self.years:
                    self.years.append(y)

        if extracted.get("semesters"):
            for s in extracted["semesters"]:
                if s not in self.semesters:
                    self.semesters.append(s)

    def most_recent_course(self):
        return self.courses[-1] if self.courses else None

    def most_recent_lecturer(self):
        return self.lecturers[-1] if self.lecturers else None

    def to_prompt_str(self):
        return (
            f"Courses mentioned in conversation: {self.courses}\n"
            f"Lecturers mentioned in conversation: {self.lecturers}\n"
            #f"Years mentioned: {self.years}\n"
            #f"Semesters mentioned: {self.semesters}\n"
            f"Most recent course (used for vague references): {self.most_recent_course()}\n"
            f"Most recent lecturer: {self.most_recent_lecturer()}\n"
        )
