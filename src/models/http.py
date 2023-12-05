from typing import Optional
from pydantic import BaseModel


class StandardResponse(BaseModel):
  message: str
  error: Optional[str] = None

  def to_json(self):
    return self.__dict__