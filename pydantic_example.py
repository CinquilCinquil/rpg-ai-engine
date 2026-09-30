from ollama import chat
from pydantic import BaseModel


class WeatherArgs(BaseModel):
    city: str


def get_weather(city: str) -> str:
    # Replace with a real API call.
    return f"The weather in {city} is 28°C and sunny."


tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": WeatherArgs.model_json_schema(),
        },
    }
]

messages = [
    {"role": "user", "content": "What's the weather in Natal?"}
]

response = chat(
    model="qwen3:4b",
    messages=messages,
    tools=tools,
)

# Let the model request a tool.
messages.append(response.message)

for call in response.message.tool_calls or []:
    args = WeatherArgs.model_validate(call.function.arguments)

    print(call)

    """
    if call.function.name == "get_weather":
        result = get_weather(args.city)

        messages.append({
            "role": "tool",
            "tool_name": call.function.name,
            "content": result,
        })
    """

"""
# Give the tool result back to the model.
final = chat(
    model="qwen3:4b",
    messages=messages,
)

print(final.message.content)
"""

# import json

# with open("tools.json") as f:
#    tools = json.load(f)