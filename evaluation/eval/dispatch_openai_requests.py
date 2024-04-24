'''
This file is copied and modified from https://gist.github.com/neubig/80de662fb3e225c18172ec218be4917a.
Thanks to Graham Neubig for sharing the original code.
'''

import openai
import asyncio
from typing import Any, List, Dict

async def dispatch_openai_chat_requesets(
    messages_list: List[List[Dict[str,Any]]],
    model: str,
    **completion_kwargs: Any,
) -> List[str]:
    """Dispatches requests to OpenAI chat completion API asynchronously.
    
    Args:
        messages_list: List of messages to be sent to OpenAI chat completion API.
        model: OpenAI model to use.
        completion_kwargs: Keyword arguments to be passed to OpenAI ChatCompletion API. See https://platform.openai.com/docs/api-reference/chat for details.
    Returns:
        List of responses from OpenAI API.
    """
    async_responses = [
        openai.ChatCompletion.acreate(
            engine=model,
            messages=x,
            **completion_kwargs,
        )
        for x in messages_list
    ]
    return await asyncio.gather(*async_responses)


async def dispatch_openai_prompt_requesets(
    prompt_list: List[str],
    model: str,
    **completion_kwargs: Any,
) -> List[str]:
    """Dispatches requests to OpenAI text completion API asynchronously.
    
    Args:
        prompt_list: List of prompts to be sent to OpenAI text completion API.
        model: OpenAI model to use.
        completion_kwargs: Keyword arguments to be passed to OpenAI text completion API. See https://platform.openai.com/docs/api-reference/completions for details.
    Returns:
        List of responses from OpenAI API.
    """
    async_responses = [
        openai.Completion.acreate(
            engine=model,
            prompt=x,
            **completion_kwargs,
        )
        for x in prompt_list
    ]
    return await asyncio.gather(*async_responses)


if __name__ == "__main__":
     # chat_completion_responses = openai.ChatCompletion.create(
    #     engine="code-davinci-002", # The deployment name you chose when you deployed the GPT-35-Turbo or GPT-4 model.
    #     messages=[
    #         {"role": "system", "content": "Assistant is a large language model trained by OpenAI."},
    #         {"role": "user", "content": "Who were the founders of Microsoft?"}
    #     ]
    # )
    
    import openai
    # openai.api_key = "7cf72d256d55479383ab6db31cda2fae"
    # openai.api_base =  "https://pnlpopenai2.openai.azure.com/" 
    openai.api_type = 'azure'
    openai.api_version = '2023-05-15' # this may change in the future
    openai.api_key = "050fd3ed1d8740bfbd07334dfbc6a614" 
    openai.api_base = "https://pnlpopenai3.openai.azure.com/"
    
    
    # chat_completion_responses = openai.ChatCompletion.create(
    #     engine="gpt-4", # The deployment name you chose when you deployed the GPT-35-Turbo or GPT-4 model.
    #     messages=[
    #         {"role": "system", "content": "Assistant is a large language model trained by OpenAI."},
    #         {"role": "user", "content": "Who were the founders of Microsoft?"}
    #     ]
    # )
    
    chat_completion_responses = asyncio.run(
        dispatch_openai_chat_requesets(
            messages_list=[
                [{"role": "user", "content": "Write a poem about asynchronous execution."}],
                [{"role": "user", "content": "Write a poem about asynchronous pirates."}],
            ],
            model="gpt-4",
            temperature=0.3,
            max_tokens=200,
            top_p=1.0,

        )
    )

    # for i, x in enumerate(chat_completion_responses):
    #     print(f"Chat completion response {i}:\n{x['choices'][0]['message']['content']}\n\n")

    # prompt_completion_responses = asyncio.run(
    #     dispatch_openai_prompt_requesets(
    #         prompt_list=[
    #             "Write a poem about asynchronous execution.\n",
    #             "Write a poem about asynchronous pirates.\n",
    #         ],
    #         model="text-davinci-003",
    #         temperature=0.3,
    #         max_tokens=200,
    #         top_p=1.0,
    #     )
    # )

    # for i, x in enumerate(prompt_completion_responses):
    #     print(f"Prompt completion response {i}:\n{x['choices'][0]['text']}\n\n")