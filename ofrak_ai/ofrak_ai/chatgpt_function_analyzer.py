import logging

from dataclasses import dataclass
from openai.error import OpenAIError
from tiktoken import Encoding, encoding_for_model

from ofrak import Resource
from ofrak.component.analyzer import Analyzer
from ofrak.core.complex_block import ComplexBlock
from ofrak_ai.chatgpt import ChatGPTAnalysis, ChatGPTConfig, get_chatgpt_response

LOGGER = logging.getLogger(__name__)


@dataclass
class ChatGPTFunctionAnalyzerConfig(ChatGPTConfig):
    encoding: Encoding = encoding_for_model(ChatGPTConfig.model)


class ChatGPTFunctionAnalyzer(Analyzer[ChatGPTFunctionAnalyzerConfig, ChatGPTAnalysis]):
    targets = (ComplexBlock,)
    outputs = (ChatGPTAnalysis,)

    async def analyze(
        self, resource: Resource, config: ChatGPTFunctionAnalyzerConfig
    ) -> ChatGPTAnalysis:
        if not config:
            config = ChatGPTFunctionAnalyzerConfig()
        cb = await resource.view_as(ComplexBlock)
        assembly = await cb.get_assembly()
        history = [
            {
                "role": "user",
                "content": f"You are a security researcher capable of reverse engineering code from assembly code.\n\
                            Below is the assembly code and associated symbol name from a binary that I am trying to reverse engineer.\n\
                            {cb.Symbol}:\n\
                            {assembly}\n\
                            Please do the following:\n\
                            \tSummarize this assembly code and explain your reasoning as much as possible.\n\
                            \tWITHOUT USING INLINE ASSEMBLY OR THE __asm__ FUNCTION, decompile it into equivalent pseudo-C code with high-level comments.\n\
                            \tGuess the closest matching library function based on function signature.\n\
                            \tAnd finally, suggest suitable names for the function in question.",
            }
        ]
        encoding_length = len(config.encoding.encode(history[0]["content"]))
        if encoding_length >= 4000:
            return
        try:
            response = await get_chatgpt_response(
                history=history, max_tokens=4096 - encoding_length, config=config
            )

        except OpenAIError as e:
            # openai's error messages are rather unhelpful. Log traceback for additional details
            LOGGER.exception(f"Exception occurred, skipped {cb.Symbol}")

        return ChatGPTAnalysis(response.choices[0].message.content)
