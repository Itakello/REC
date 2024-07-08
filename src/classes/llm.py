from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator, Iterator, Mapping

import httpx
import ollama
from itakello_logging import ItakelloLogging
from tqdm import tqdm

from ..interfaces.base_class import BaseClass

logger = ItakelloLogging().get_logger(__name__)


@dataclass
class LLM(BaseClass):
    base_model: str
    system_prompt_path: Path
    model_name: str = field(init=False)
    system_prompt: str = field(init=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.model_name = f"custom_{self.base_model}"
        self.system_prompt = self.load_system_prompt()
        self.create_custom_model()

    def load_system_prompt(self) -> str:
        with open(self.system_prompt_path, "r") as file:
            return file.read().strip()

    def create_custom_model(self) -> None:
        try:
            logger.info(f"Creating custom model {self.model_name}")
            modelfile = f"""
            FROM {self.base_model}
            SYSTEM "{self.system_prompt}"
            """
            try:
                ollama.create(model=self.model_name, modelfile=modelfile)
                logger.confirmation(
                    f"Custom model {self.model_name} created successfully."
                )
            except ollama.ResponseError as e:
                logger.error(f"Error creating custom model: {e}")
                raise
        except httpx.ConnectError:
            logger.error("Ollama is not currently running. Please start it.")
            raise

    async def _show_async_progress_tqdm(self, iterator: AsyncIterator) -> None:
        pbar = None
        async for update in iterator:
            if "total" in update and pbar is None:
                pbar = tqdm(
                    total=update["total"] / 1e9,
                    bar_format="{l_bar}{bar}| {n:.3f}/{total:.3f} {unit} [elapsed: {elapsed}]",
                    unit="GB",
                    colour="green",
                )
            if pbar is not None and "completed" in update:
                pbar.n = update["completed"] / 1e9
                pbar.refresh()

        if pbar is not None:
            pbar.close()

    async def _download_base_model(self) -> None:
        try:
            ollama.show(self.base_model)
            logger.info(f"Model {self.base_model} already exists.")
        except httpx.ConnectError:
            logger.error("Ollama is not currently running. Please start it.")
            raise
        except Exception:
            logger.warning(f"Downloading model [{self.base_model}]. Please wait...")
            iterator = await ollama.AsyncClient().pull(
                model=self.base_model, stream=True
            )
            if isinstance(iterator, Mapping):
                logger.error("The LLM download iterator is not an async iterator.")
                raise
            await self._show_async_progress_tqdm(iterator)
            logger.confirmation(
                f"Base model [{self.base_model}] downloaded successfully"
            )

    def generate(self, sentences: list[str]) -> str:
        try:
            prompt = ", ".join(sentences)
            response = ollama.generate(model=self.model_name, prompt=prompt)
            if isinstance(response, Iterator):
                logger.error("The LLM generated response is an iterator.")
                raise
            return response["response"]
        except httpx.ConnectError:
            logger.error("Ollama is not currently running. Please start it.")
            raise
        except Exception as e:
            logger.error(f"An error occurred during generation: {e}")
            raise


if __name__ == "__main__":
    llm = LLM(
        base_model="llama3",
        system_prompt_path=Path("prompts/referential-expression-prompt.txt"),
    )
    new_sentence = llm.generate(
        [
            "A blonde woman in a white shirt and long black skirt.",
            "There is one small girl wearing white top is touching the elephant",
        ]
    )
    print(new_sentence)
