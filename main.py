import sys
from dataclasses import dataclass
from os import getenv

import httpx
import tiktoken
from dotenv import load_dotenv
from openai import AzureOpenAI
from openai.types import CreateEmbeddingResponse
from openai.types.chat import ChatCompletion


def getenv_or_throw(name: str) -> str:
    value = getenv(name)
    if value is None:
        raise ValueError(f"Environment variable '{name}' is not set")
    return value


@dataclass
class AzureOpenAIConfig:
    api_key: str
    """API Key to use for the Azure OpenAI API."""

    api_version: str
    """
    API version to use for the Azure OpenAI API.
    E.g.: 2024-02-15-preview
    https://learn.microsoft.com/en-us/azure/ai-services/openai/reference
    """

    azure_endpoint: str
    """
    Azure OpenAI API base URL.
    E.g.: https://example.openai.azure.com/
    """

    completion_model_name: str
    """
    Model (a.k.a deployment) name to use for completions.
    This is dependent on your provisioned resources.
    E.g.: gpt-35-turbo-dev (but can be anything)
    """

    embedding_model_name: str
    """
    Model (a.k.a deployment) name to use for embeddings.
    This is dependent on your provisioned resources.
    E.g.: text-embedding-ada-002-dev (but can be anything)
    """


def create_azure_openai_client(config: AzureOpenAIConfig) -> AzureOpenAI:
    """
    Initializes the Azure OpenAI client with the API Key and other necessary environment variables
    """
    # disable SSL verification for the HTTP client so we can use an SSH-tunnel
    # to connect to the Azure OpenAI API since it's firewalled
    http_client = httpx.Client(verify=False)

    return AzureOpenAI(
        api_key=config.api_key,
        api_version=config.api_version,
        azure_endpoint=config.azure_endpoint,
        http_client=http_client,
    )


def count_tokens(text: str) -> int:
    encoding = tiktoken.encoding_for_model("gpt-35-turbo")
    tokens = encoding.encode(text)
    return len(tokens)


@dataclass
class AIPrompt:
    system: str
    user: str


@dataclass
class AICompletionResult:
    content: str


@dataclass
class AIEmbeddingResult:
    embeddings: list[float]


class OpenAIClient:
    def __init__(self, config: AzureOpenAIConfig):
        self.__config: AzureOpenAIConfig = config
        self.__client = create_azure_openai_client(self.__config)

    def complete(self, prompt: AIPrompt) -> AICompletionResult:
        """
        Completes the prompt with the Azure OpenAI API.
        """

        # https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/chatgpt?tabs=python-new
        completion: ChatCompletion = self.__client.chat.completions.create(
            model=self.__config.completion_model_name,
            temperature=None,
            frequency_penalty=None,
            seed=1,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": prompt.system,
                },
                {
                    "role": "user",
                    "content": prompt.user,
                },
            ],
        )

        return AICompletionResult(content=completion.choices[0].message.content)

    def create_embedding(self, text: str) -> AIEmbeddingResult:
        """
        Creates an embedding for the given text.
        """
        if count_tokens(text) > 8192:
            raise ValueError("Text is too long to embed")

        # https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/embeddings?tabs=python-new
        response: CreateEmbeddingResponse = self.__client.embeddings.create(
            model=self.__config.embedding_model_name,
            input=text,
        )

        return AIEmbeddingResult(embeddings=response.data[0].embedding)


def run_completion_task(openai_client: OpenAIClient):
    completion_result = openai_client.complete(
        prompt=AIPrompt(
            system=(
                "Your task is to select one of the following words that best summarize the user's input: "
                "'happy', 'sad', 'angry', 'excited', 'calm', 'bored', 'confused', 'surprised', 'disgusted', 'fearful', 'neutral'. "
                "Your responses must be formatted in valid JSON."
            ),
            user=(
                "Here is the user's input: 'I stubbed my toe and am miffed about it'."
            ),
        )
    )
    print(completion_result.content)


def run_embedding_task(openai_client: OpenAIClient):
    embedding_result = openai_client.create_embedding(
        text="The quick brown fox jumps over the lazy dog."
    )
    print(embedding_result.embeddings)


task_by_command_name = {
    "completion": run_completion_task,
    "embedding": run_embedding_task,
}


def main(argv: list[str]):
    openai_client = OpenAIClient(
        config=AzureOpenAIConfig(
            api_key=getenv_or_throw("AZURE_OPENAI_API_KEY"),
            api_version=getenv_or_throw("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=getenv_or_throw("AZURE_OPENAI_API_BASE_URL"),
            completion_model_name=getenv_or_throw("AZURE_OPENAI_COMPLETION_MODEL_NAME"),
            embedding_model_name=getenv_or_throw("AZURE_OPENAI_EMBEDDING_MODEL_NAME"),
        )
    )

    if len(argv) == 0:
        raise ValueError("Missing command")

    command = argv[0]
    task = task_by_command_name.get(command)
    if task is None:
        raise ValueError(f"Unknown command: {command}")

    task(openai_client)


if __name__ == "__main__":
    load_dotenv()
    try:
        main(sys.argv[1:])
    except Exception as e:
        print(e)
        sys.exit(1)
