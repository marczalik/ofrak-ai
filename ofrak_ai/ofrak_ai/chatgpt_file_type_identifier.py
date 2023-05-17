import logging

from dataclasses import dataclass

from ofrak import Resource
from ofrak.component.analyzer import Analyzer
from ofrak.core.program import Program
from ofrak_ai.chatgpt import ChatGPTAnalysis, ChatGPTConfig, get_chatgpt_response

LOGGER = logging.getLogger(__name__)


@dataclass
class ChatGPTFunctionAnalyzerConfig(ChatGPTConfig):
    pass


class ChatGPTFunctionAnalyzer(Analyzer[ChatGPTFunctionAnalyzerConfig, ChatGPTAnalysis]):
    targets = (Program,)
    outputs = (ChatGPTAnalysis,)

    async def analyze(
        self, resource: Resource, config: ChatGPTFunctionAnalyzerConfig
    ) -> ChatGPTAnalysis:
        if not config:
            config = ChatGPTFunctionAnalyzerConfig()
        assert await resource.view_as(Program)
        data = await resource.get_data()
        x = 0
        y = 1000
        chunks = []
        while y <= len(data):
            chunks.append(data[x:y])
            x += 1000
            y += 1000
        history = []
        for chunk in chunks:
            history.append(
                {
                    "role": "user",
                    "content": f"This is a chunk of bytecode from a program: {chunk!r}",
                }
            )
        history.append(
            {
                "role": "user",
                "content": f"Please explain what the program does and rewrite it in C for me.",
            }
        )
        response = await get_chatgpt_response(
            history=history, max_tokens=1000, config=config
        )
        return ChatGPTAnalysis(response.choices[0].message.content)
