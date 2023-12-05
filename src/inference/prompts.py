import enum
import jinja2

from typing import Callable, List, Optional

from pydantic import BaseModel

class PromptTemplateName(enum.Enum):
    default = 0
    chatml = 1

class PromptMessage(BaseModel):
    role: str
    content: Optional[str]

class PromptTemplate:
    def __init__(self, tmpl: Callable[[List[PromptMessage]], str], stop: str) -> None:
        self.template = tmpl
        self.stop = stop if len(stop) > 0 else ""

    def get_prompt(self, messages: List[PromptMessage]) -> str:
        return self.template(messages)


def get_default_tmpl() -> PromptTemplate:
    def fn(messages: List[PromptMessage]) -> str:
        # # Create a variable to store the prompt
        # prompt = ""

        # # Add a system message if provided
        # if len(system_message) > 0:
        #     prompt += f"System:\n{system_message}\n"

        # # Add a user message
        # prompt += f"User:\n{user_prompt}"
        # return prompt
        return ""

    return PromptTemplate(fn, "")


# Prompt template for chatml
# TODO: Load jinja2 template from file
def get_chatml_tmpl() -> PromptTemplate:
    environment = jinja2.Environment()
    chatml_tmpl: str = """{% for msg in messages %}<|im_start|>{{ msg.role }}
{{ msg.content }}<|im_end|>
{% endfor %}<|im_start|>assistant"""
    template = environment.from_string(chatml_tmpl)
    def fn(messages: List[PromptMessage]) -> str:
        prompt = template.render(messages=messages)
        print("================")
        print("Generated prompt:", prompt)
        print("================")
        return prompt

    return PromptTemplate(fn, "<|im_end|>")

def get_tmpl(tmpl: str) -> PromptTemplate:
    match tmpl:
        case "chatml":
            print("Loading chatml prompt template")
            return get_chatml_tmpl()
        case _:
            print("Loading default prompt template")
            return get_default_tmpl()