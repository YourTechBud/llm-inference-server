import enum
from typing import Callable

class PromptTemplateName(enum.Enum):
    default = 0
    chatml = 1

class PromptTemplate:
    def __init__(self, tmpl: Callable[[str, str], str], stop: str) -> None:
        self.template = tmpl
        self.stop = stop if len(stop) > 0 else ""

    def get_prompt(self, system_message: str, user_prompt: str) -> str:
        return self.template(system_message, user_prompt)


def get_default_tmpl() -> PromptTemplate:
    def fn(system_message, user_prompt) -> str:
        # Create a variable to store the prompt
        prompt = ""

        # Add a system message if provided
        if len(system_message) > 0:
            prompt += f"System:\n{system_message}\n"

        # Add a user message
        prompt += f"User:\n{user_prompt}"
        return prompt

    return PromptTemplate(fn, "")


# Prompt template for chatml
def get_chatml_tmpl() -> PromptTemplate:
    def fn(system_message, user_prompt) -> str:
        if system_message == "":
            system_message = "You are a helpul AI assistant. Answer the user's prompt step by step."
        prompt = f"""<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{user_prompt}<|im_end|>
<|im_start|>assistant"""
        print("Generated prompt:", prompt)
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