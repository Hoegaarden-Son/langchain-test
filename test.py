from openai import OpenAI

client = OpenAI(
  api_key="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"
)

completion = client.chat.completions.create(
  model="gpt-4o-mini",
  store=True,
  messages=[
    {"role": "user", "content": "hello? do you like pizza?"}
  ]
)

print(completion.choices[0].message);
