import sys
from dataclasses import dataclass
from os import getenv

import httpx
from dotenv import load_dotenv
from openai import AzureOpenAI
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

    model_name: str
    """
    Model (a.k.a deployment) name to use for completions.
    This is dependent on your provisioned resources.
    E.g.: gpt-35-turbo-dev (but can be anything)
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


@dataclass
class AIPrompt:
    system: str
    user: str


@dataclass
class AICompletionResult:
    content: str


class OpenAIClient:
    def __init__(self, config: AzureOpenAIConfig):
        self.__config: AzureOpenAIConfig = config
        self.__client = create_azure_openai_client(self.__config)

    def complete(self, prompt: AIPrompt) -> AICompletionResult:
        completion: ChatCompletion = self.__client.chat.completions.create(
            model=self.__config.model_name,
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


def main(argv: list[str]):
    openai_client = OpenAIClient(
        config=AzureOpenAIConfig(
            api_key=getenv_or_throw("AZURE_OPENAI_API_KEY"),
            api_version=getenv_or_throw("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=getenv_or_throw("AZURE_OPENAI_API_BASE_URL"),
            model_name=getenv_or_throw("AZURE_OPENAI_MODEL_NAME"),
        )
    )

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


if __name__ == "__main__":
    load_dotenv()
    try:
        main(sys.argv[1:])
    except Exception as e:
        print(e)
        sys.exit(1)
