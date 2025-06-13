import os
import time
import logging
import discord
import requests
from dotenv import load_dotenv
import google.generativeai as genai
from collections import defaultdict
from datetime import datetime, timedelta, timezone

# --- 環境変数のチェックと読み込み ---
load_dotenv()

# 必要な環境変数のリスト
REQUIRED_VARS = [
    'DISCORD_TOKEN',
    'GEMINI_API_KEY',
    'TARGET_CHANNEL_ID',
    'GOOGLE_SEARCH_API_KEY',
    'GOOGLE_SEARCH_ENGINE_ID'
]

# 環境変数を読み込み
DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
TARGET_CHANNEL_ID = int(os.getenv('TARGET_CHANNEL_ID'))
GOOGLE_SEARCH_API_KEY = os.getenv('GOOGLE_SEARCH_API_KEY')
GOOGLE_SEARCH_ENGINE_ID = os.getenv('GOOGLE_SEARCH_ENGINE_ID')

# --- Gemini APIの設定 ---
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel(model_name="gemini-1.5-flash")

class ConversationManager:
    def __init__(self, max_age_minutes=60, max_messages=30):
        self.conversations = defaultdict(list)
        self.max_age = timedelta(minutes=max_age_minutes)
        self.max_messages = max_messages
        self.token_limit = 7168

    def add_message(self, channel_id: int, role: str, content: str):
        role_for_gemini = "model" if role == "assistant" else "user"
        self.conversations[channel_id].append({
            'role': role_for_gemini,
            'content': content,
            'timestamp': datetime.now(timezone.utc)
        })
        self._cleanup_conversation(channel_id)

    def clear_conversation(self, channel_id: int = None):
        if channel_id is None:
            self.conversations.clear()
        elif channel_id in self.conversations:
            self.conversations[channel_id].clear()

    def get_formatted_history(self, channel_id: int) -> list:
        return [{'role': msg['role'], 'parts': [msg['content']]} for msg in self.conversations[channel_id]]

    def _cleanup_conversation(self, channel_id: int):
        current_time = datetime.now(timezone.utc)
        self.conversations[channel_id] = [
            msg for msg in self.conversations[channel_id]
            if current_time - msg['timestamp'] <= self.max_age
        ]
        if len(self.conversations[channel_id]) > self.max_messages:
            self.conversations[channel_id] = self.conversations[channel_id][-self.max_messages:]
        while self._calculate_total_tokens(channel_id) > self.token_limit:
            if not self.conversations[channel_id]: break
            self.conversations[channel_id].pop(0)

    def _calculate_total_tokens(self, channel_id: int) -> int:
        history_content = [msg['content'] for msg in self.conversations[channel_id]]
        if not history_content: return 0
        try:
            return model.count_tokens(history_content).total_tokens
        except Exception as e:
            logging.warning(f"トークン数の計算中にエラー: {e}")
            return sum(len(content) for content in history_content)

def google_search(query: str, num_results: int = 3) -> str:
    url = f"https://www.googleapis.com/customsearch/v1?q={query}&key={GOOGLE_SEARCH_API_KEY}&cx={GOOGLE_SEARCH_ENGINE_ID}&num={num_results}&gl=jp&hl=ja"
    try:
        response = requests.get(url)
        response.raise_for_status()
        search_results = response.json()
        snippets = [f"- {item.get('title', '')}: {item.get('snippet', '')}" for item in search_results.get("items", [])]
        return "\n".join(snippets) if snippets else "関連する情報が見つかりませんでした。"
    except requests.exceptions.RequestException as e:
        logging.error(f"Google Search API Error: {e}")
        return "検索中にエラーが発生しました。"

async def get_contextual_history(channel: discord.TextChannel, limit: int = 5) -> list:
    history = []
    async for message in channel.history(limit=limit):
        content = message.clean_content.strip()
        if not content: continue
        role = "model" if message.author.bot else "user"
        history.append({'role': role, 'parts': [content]})
    return list(reversed(history))

intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)
conversation_manager = ConversationManager(max_age_minutes=10, max_messages=20)

@client.event
async def on_ready():
    print(f'{client.user} としてログインしました')
    target_channel = client.get_channel(TARGET_CHANNEL_ID)
    if target_channel:
        print(f'監視対象チャンネル: #{target_channel.name} (ID: {TARGET_CHANNEL_ID})')
    else:
        print(f'警告: チャンネルID {TARGET_CHANNEL_ID} が見つかりません。')

@client.event
async def on_message(message):
    if message.author == client.user or message.channel.id != TARGET_CHANNEL_ID:
        return

    user_input = message.clean_content.strip()
    if not user_input: return

    if user_input == "!reset":
        conversation_manager.clear_conversation(message.channel.id)
        await message.channel.send("短期会話履歴をリセットしました。")
        return

    async with message.channel.typing():
        try:
            if user_input.lower().startswith("!search") or user_input.lower().startswith("!research"):
                query = user_input.split(maxsplit=1)[1]
                search_context = google_search(query)
                prompt = f"以下の検索結果を参考にして、ユーザーの質問に日本語で分かりやすく要約して答えてください。\n\n# 参考情報\n{search_context}\n\n# ユーザーの質問\n{query}"
                response = model.generate_content(prompt)
                await message.reply(response.text)
                return

            discord_history = await get_contextual_history(message.channel, limit=5)
            short_term_history = conversation_manager.get_formatted_history(message.channel.id)
            full_history = short_term_history + discord_history
            
            chat_session = model.start_chat(history=full_history)
            response = await chat_session.send_message_async(user_input)
            response_text = response.text.strip()
            
            conversation_manager.add_message(message.channel.id, "user", user_input)
            conversation_manager.add_message(message.channel.id, "assistant", response_text)
            
            if response_text:
                await message.reply(response_text)
            else:
                await message.reply("うーん…")

        except Exception as e:
            logging.error(f"メッセージ処理エラー: {e}", exc_info=True)
            await message.reply(f"ごめんなさい、エラーが発生しました。\n`{e}`")


if __name__ == "__main__":
    try:
        client.run(DISCORD_TOKEN, log_level=logging.INFO)
    except Exception as e:
        print(f"ボットの起動に失敗しました: {e}")
