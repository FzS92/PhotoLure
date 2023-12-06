import openai


def chat_gpt(prompt):
    openai_key = "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"
    openai.api_key = openai_key

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {
                "role": "user",
                "content": (
                    f"In one paragraph, describe the {prompt}.cinematic, nature,"
                    " hyperrealistic, 8K"
                ),
            }
        ],
    )

    response = response.choices[0].message.content
    return response


if __name__ == "__main__":
    prompt = (
        "A luxury huge private yacht, sailing in the bahamas with palm trees in the"
        " background and hardwood deck on the yacht,"
    )
    response = chat_gpt(prompt)
    print(type(response))
    print(response)
