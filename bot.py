import os
from dotenv import load_dotenv
from huggingface_hub import hf_hub_url
import torch
import transformers
import discord

load_dotenv()

token = os.getenv("DISCORD_TOKEN")

bot = discord.Client()

model_cache = {}

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    if message.content.startswith("!chat"):
        prompt = message.content.replace("!chat", "")
        response = await get_response(prompt)
        await message.channel.send(f"{message.author.mention} {response.strip()}")

    elif message.content.startswith("!model"):
        model_name = message.content.replace("!model", "").strip()
        model_cache[message.author] = model_name
        await message.channel.send(f"{message.author.mention}, model set to {model_name}")

async def get_response(prompt):
    author = message.author
    model_name = model_cache.get(author, "gpt2")
    if model_name not in model_cache:
        model_cache[author] = model_name

    if model_name not in model_cache.values():
        model_cache[author] = model_name

    if model_name not in model_cache:
        response = "Error: Model not found. Please choose a model from the list or set your own model using the !model command."
    else:
        model = transformers.pipeline("text-generation", model=model_name)
        response = model(prompt, max_length=30, num_return_sequences=1, do_sample=True, top_k=50, top_p=0.95)[0]["generated_text"]

    return response

bot.run(token)
