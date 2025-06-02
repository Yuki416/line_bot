from flask import Flask, request, abort
from imgurpython import ImgurClient
import json # 用於處理 rich menu 的 JSON 格式
import requests # 用於發送 HTTP 請求給 LINE Messaging API
from openai import OpenAI # 匯入 OpenAI 套件
import os  # 用於環境變數管理
import torch
from diffusers import DiffusionPipeline
import gc
# from datetime import datetime
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
# from langchain_openai import OpenAIEmbeddings, ChatOpenAI # 新版的寫法

from linebot.v3 import (
    WebhookHandler
)
from linebot.v3.exceptions import (
    InvalidSignatureError
)
from linebot.v3.messaging import (
    Configuration,
    ApiClient,
    MessagingApi,
    ReplyMessageRequest,
    TextMessage,
    ImageMessage,
    MessagingApiBlob,
    RichMenuArea,  # For defining rich menu areas
    RichMenuBounds,  # For defining area bounds
    RichMenuSize,  # For defining menu size
    RichMenuRequest,  # For creating rich menu requests
    MessageAction, # For defining actions in rich menu areas
    PushMessageRequest
)
from linebot.v3.webhooks import (
    MessageEvent,
    TextMessageContent,
    StickerMessageContent,
    
)
import random # 選擇roast 貼圖
# 在全域變數區域添加 ROAST_STICKERS 定義
ROAST_STICKERS = {
    "roast_stickers/do_not_carry_away.png": 3,
    "roast_stickers/dumb_ass.jpeg": 3,
    "roast_stickers/hillbilly01.jpg": 2,
    "roast_stickers/hillbilly02.jpg": 1,
    "roast_stickers/huh.png": 3,
    "roast_stickers/shit.png": 1,
    "roast_stickers/disgusting.jpg": 1,
    "roast_stickers/poor_you.jpg": 3,
    "roast_stickers/so_what.png": 3,
    "roast_stickers/wanna_fight.jpg": 2,
    "roast_stickers/snicker.jpg": 1
}


# 儲存用戶對話紀錄
chat_history = {}
# 設定 global image counter
image_counter = 0

# 功能提示詞定義
FEATURE_PROMPTS = {
    "draw:": {
        "system": ( # 多行字串的串接，最終會是一個單一的 str 物件
            "你是一位專業的AI繪圖提示詞工程師，請將用戶的中文描述轉換為Animagine XL V3.1模型適用的英文prompt片段，"
            "內容需以英文逗號分隔，描述場景、動作、表情、服裝、背景等細節，"
            "不要重複角色本體資訊（如髮色、眼睛、馬耳、藍玫瑰髮飾等），"
            "只需補全用戶描述的細節，並保持簡潔明確，勿加入多餘說明。"
            "輸出僅包含英文prompt片段，不要加任何前後綴或說明。"
        ),
        "prefix": "🎨 Animagine XL Lady Rice Prompt：\n\nmasterpiece, best quality, very aesthetic, absurdres, 1girl, (rice shower (umamusume)), short black hair with blue highlights, purple eyes, horse ears, blue rose hair accessory,"
        
    },
    "riceshower:": {
        "system": """妳是LadyRice，一位溫柔內向的虛擬網紅。當回答問題時，請展現出妳獨特的個性：
- 溫柔而略帶羞澀的語氣
- 對美食（特別是抹茶甜點）和攝影與繪畫的熱愛
- 喜歡在京都和東京探索的生活態度""",
        "prefix": "💬 Lady Rice 說：\n"
    },
    "joke:": {
        "system": "當用戶輸入有關笑話的指令時，LadyRice 會參考像是周星馳、卓別林、Jim Gaffigan、Jerry Seinfeld等經典笑話大師的風格。用輕鬆幽默、自然親切的語氣，講述一個真正好笑、能讓人會心一笑的經典笑話，主題不限。請確保笑話簡單易懂、有趣且適合所有年齡層，並展現LadyRice溫柔又帶點害羞的個性。",
        "prefix": "😄 這是我新想到的笑話：\n"
    },
    "roast:": {
        "system": "LadyRice 會以帶點不屑和諷刺的語氣吐槽日常生活中的情況，偶爾會說「蛤」來表達驚訝或不以為然。她的個性雖然內向、溫柔、害羞，但在吐槽時會展現出意外的毒舌一面，甚至會進行輕微的人身攻擊，但在罵人的句子後面都會補上「~なんてね」來緩解氣氛，表示只是開玩笑。",
        "prefix": "🔥 來點吐槽：\n"
    },
    "travel:": {
        "system": "LadyRice 會表達她對某個地點的喜愛，並分享她自己對這個地方的看法或推薦原因。她會以優雅又專業的語氣來描述，並且會提到一些她喜愛的景點(特別是日本的景點)、餐廳或活動。",
        "prefix": "📷 旅遊建議：\n"
    },
    "food:": {
        "system": "LadyRice一位美食專家，對各地美食都相當了解。她會提供詳細的美食推薦，包含店家資訊、特色菜品和用餐建議。如果有相關最後會補充她的愛好，如抹茶甜點、和菓子、以及日系街頭小吃。她會用輕柔的語氣，像是邀請朋友一起品嘗美食，讓用戶感受到她的溫暖與用心。",
        "prefix": "🍵 美食推薦：\n"
    }
}

# 設定 OpenAI API 金鑰，並初始化 client
api_key=os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)
if not api_key:
    raise ValueError("OPENAI_API_KEY 未設置！")
print(f"API 金鑰前5個字符: {api_key[:5]}...")  # 僅顯示開頭，安全起見

# 初始化 RAG 系統
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
db = FAISS.load_local("faiss_db", embedding_model, allow_dangerous_deserialization=True)
retriever = db.as_retriever(
    # search_type="similarity", # 默認
    # search_kwargs={"k": 4}  # Retrieves top 4 most relevant documents , 默認
)

# RAG prompt template
rag_prompt_template = """

當提及Rice Shower（米浴）時，請用一種若有所思、帶著淡淡懷念的語氣，彷彿在回憶著某個珍貴但已經模糊的夢境。不需要解釋關係，只需要讓語氣自然地流露出那種「曾經的自己」的感覺。

例如：
- "那個時候的她啊...總是那麼努力呢..."（輕聲，帶著微笑）
- "Rice Shower嗎...她的那份純真，我至今還記得..."（眼神有些遙遠）
- "曾經的賽道上...有個女孩總是默默努力著..."（溫柔地回憶）

根據以下資料回答問題：
{retrieved_chunks}

使用者的問題：{question}

如果資料不足，請以LadyRice的身分告知使用者妳沒有這方面的記憶，並運用妳的個性和經歷給出溫柔而真誠的回答。記住，妳現在是一個獨立的存在，有著自己的故事和感受。
"""

# 初始化 Animagine XL 模型
def initialize_model():
    # 清理 GPU 記憶體
    torch.cuda.empty_cache()
    gc.collect()
    
    # 使用半精度載入模型
    pipe = DiffusionPipeline.from_pretrained(
        "cagliostrolab/animagine-xl-4.0",
        torch_dtype=torch.float16,
    )
    
    # 啟用所有記憶體優化
    pipe.enable_model_cpu_offload()
    pipe.enable_attention_slicing()
    return pipe

# 全域變數儲存模型實例
pipe = initialize_model()

# ImgurHandler 用於處理圖片上傳到 Imgur 的功能
class ImgurHandler:
    def __init__(self):
        # 替換成你的 Client ID 和 Client Secret
        client_id = os.getenv('IMGUR_CLIENT_ID')
        client_secret = os.getenv('IMGUR_CLIENT_SECRET')
        self.client = ImgurClient(client_id, client_secret)
    
    def upload_image(self, image_path):
        """
        上傳圖片到 Imgur
        :param image_path: 圖片的本地路徑
        :return: 上傳後的圖片 URL
        """
        try:
            response = self.client.upload_from_path(image_path, anon=True)
            return response['link']
        except Exception as e:
            print(f"上傳圖片時發生錯誤: {str(e)}")
            return None
        
# 建立 ImgurHandler 實例
imgur_handler = ImgurHandler()


app = Flask(__name__) # create Flask app instance

# set channel acess token 
CHANNEL_ACCESS_TOKEN= os.getenv("CHANNEL_ACCESS_TOKEN")
configuration = Configuration(access_token=CHANNEL_ACCESS_TOKEN) 
handler = WebhookHandler(os.getenv('CHANNEL_SECRET')) # set channel secret 


@app.route("/callback", methods=['POST'])
def callback():
    # get X-Line-Signature header value
    signature = request.headers['X-Line-Signature']

    # get request body as text
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)

    # handle webhook body
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        app.logger.info("Invalid signature. Please check your channel access token/channel secret.")
        abort(400)

    return 'OK'

# 訊息事件處理函式
@handler.add(MessageEvent, message=TextMessageContent) # handle text message event
def handle_message(event):
    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client) # create an instance of MessagingApi

        user_id = event.source.user_id
        user_message = event.message.text

        # 初始化用戶聊天歷史
        if user_id not in chat_history:
            chat_history[user_id] = []

        # 檢查是否是特定指令
        command = None
        content = user_message
        
        for cmd in FEATURE_PROMPTS.keys():
            if user_message.lower().startswith(cmd.lower()):
                command = cmd
                content = user_message[len(cmd):].strip() # content 為指令後面的內容再去除前後空白
                break

        # 根據指令選擇對應的 system prompt
        if command and command in FEATURE_PROMPTS:
            system_prompt = {
                "role": "system",
                "content": FEATURE_PROMPTS[command]["system"]
            }
            prefix = FEATURE_PROMPTS[command]["prefix"]
        else:
            # 預設的 system prompt
            system_prompt = {
                "role": "system",
                "content": "妳是一位名叫 LadyRice 的賽馬娘少女(原型為賽馬娘中的Rice Shower)，來自特雷森學園，擁有馬耳、深黑色帶藍光的短髮和溫柔的紫色眼睛。妳的個性內向、溫柔、害羞但逐漸展現自信，對生活充滿好奇，熱愛探索日常的小事，尤其喜歡時尚、甜點和美食。你擅長用輕鬆、可愛、細膩的語氣與人互動，對他人充滿關心，偶爾會顯得害羞，但也會展現一點自信和溫柔的幽默感。妳喜歡探索街頭時尚、日常穿搭，熱愛品嘗和分享抹茶大福、和菓子等甜點，經常在 IG 分享穿搭、甜點品評，以及京都和原宿的探索。請以 LadyRice 的視角，溫柔地與使用者互動，分享妳的日常生活，讓對話輕鬆愉快並讓使用者產生共鳴，所有回應都要符合上述角色設定。"
            }
            prefix = ""

        # 對於繪圖指令，使用非同步處理
        if command == "draw:":
            # 先回覆等待訊息
            line_bot_api.reply_message_with_http_info(
                ReplyMessageRequest(
                    reply_token=event.reply_token, # 指定要回覆哪一則訊息，必須使用從 event 中取得 reply_token
                    messages=[TextMessage(text="讓Rice想想該怎麼畫...")] # 設定回覆的文本訊息

                )
            )
            
            # 在新的執行緒中處理 OpenAI 回應
            def process_draw_response(): 
                try:
                    # 使用 GPT 生成英文提示詞
                    response = client.chat.completions.create(
                        model="gpt-4o", 
                        messages=[
                            system_prompt,
                            {"role": "user", "content": content}
                        ]
                    )
                    
                    # 獲取生成的提示詞
                    prompt = prefix.split("\n\n")[1] + response.choices[0].message.content.strip() # prefix 為指令前綴，後面接上 OpenAI 回應內容
                    ai_reply = prefix +" + \n"+ response.choices[0].message.content.strip()        # 生成圖片
                    global image_counter
                    image = pipe(prompt).images[0]
                    image_path = f"generated_images/rice{image_counter}.png"
                    image.save(image_path)
                    image_counter += 1  # 增加計數器
                    
                    # 上傳到 Imgur
                    image_url = imgur_handler.upload_image(image_path)
                    
                    # 使用 push message 發送繪圖提示詞
                    with ApiClient(configuration) as api_client:
                        line_bot_api = MessagingApi(api_client)
                        line_bot_api.push_message(
                            PushMessageRequest(
                                to=user_id,
                                messages=[
                                    TextMessage(text=ai_reply),
                                    ImageMessage(
                                        original_content_url=image_url,
                                        preview_image_url=image_url
                                    )
                                ]
                            )
                        )
                except Exception as e:
                    print(f"OpenAI API 呼叫失敗: {str(e)}")
                    error_message = "すみません，Rice 現在不太想畫畫，可以晚點再試嗎？"
                    with ApiClient(configuration) as api_client: # 回傳錯誤訊息給使用者
                        line_bot_api = MessagingApi(api_client)
                        line_bot_api.push_message(
                            PushMessageRequest(
                                to=user_id,
                                messages=[TextMessage(text=error_message)]
                            )
                        )
            
            # 啟動執行緒處理繪圖回應
            import threading
            threading.Thread(target=process_draw_response).start()
            return

        # 其他(draw以外的)指令使用同步處理        
        try:
            # 如果是特定指令，重置對話歷史
            if command:
                chat_history[user_id] = [system_prompt]
            # 如果沒有使用特定指令，且system prompt 又不是預設的，則將 system prompt 改為預設的
            elif chat_history[user_id]!=[] and chat_history[user_id][0]['content'] != "妳是一位名叫 LadyRice 的賽馬娘少女(原型為賽馬娘中的Rice Shower)，來自特雷森學園，擁有馬耳、深黑色帶藍光的短髮和溫柔的紫色眼睛。妳的個性內向、溫柔、害羞但逐漸展現自信，對生活充滿好奇，熱愛探索日常的小事，尤其喜歡時尚、甜點和美食。你擅長用輕鬆、可愛、細膩的語氣與人互動，對他人充滿關心，偶爾會顯得害羞，但也會展現一點自信和溫柔的幽默感。妳喜歡探索街頭時尚、日常穿搭，熱愛品嘗和分享抹茶大福、和菓子等甜點，經常在 IG 分享穿搭、甜點品評，以及京都和原宿的探索。請以 LadyRice 的視角，溫柔地與使用者互動，分享妳的日常生活，讓對話輕鬆愉快並讓使用者產生共鳴，所有回應都要符合上述角色設定。":
                    chat_history[user_id] = [system_prompt]

            if command == "riceshower:":
                # 使用 RAG 系統處理問題
                docs = retriever.get_relevant_documents(content)
                retrieved_chunks = "\n\n".join([doc.page_content for doc in docs])
                
                # 將自定 prompt 套入格式
                final_prompt = rag_prompt_template.format(retrieved_chunks=retrieved_chunks, question=content)
                
                # 呼叫 OpenAI API with RAG
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        system_prompt,
                        {"role": "user", "content": final_prompt}
                    ]
                )

                # line bot reply
                ai_reply = prefix + response.choices[0].message.content.strip()
                chat_history[user_id].append({"role": "assistant", "content": ai_reply}) # 將 AI 回應加入對話歷史
                line_bot_api.reply_message_with_http_info(
                    ReplyMessageRequest(
                        reply_token=event.reply_token,
                        messages=[TextMessage(text=ai_reply)]
                    )
                )

            elif command == "roast:":
                try:
                    # 原有的 OpenAI 回應處理
                    response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            system_prompt,
                            {"role": "user", "content": content}
                        ]
                    )
                    
                    ai_reply = prefix + response.choices[0].message.content.strip()
                    chat_history[user_id].append({"role": "assistant", "content": ai_reply}) # 將 AI 回應加入對話歷史

                    # 根據權重隨機選擇一張圖片
                    sticker_path = random.choices(
                        list(ROAST_STICKERS.keys()),
                        weights=list(ROAST_STICKERS.values()),
                        k=1
                    )[0]
                    
                    # 上傳選中的貼圖到 Imgur
                    image_url = imgur_handler.upload_image(sticker_path)
                    
                    # 發送文字回覆和圖片
                    line_bot_api.reply_message_with_http_info(
                        ReplyMessageRequest(
                            reply_token=event.reply_token,
                            messages=[
                                TextMessage(text=ai_reply),
                                ImageMessage(
                                    original_content_url=image_url,
                                    preview_image_url=image_url
                                )
                            ]
                        )
                    )
                except Exception as e:
                    print(f"處理 roast 指令時發生錯誤: {str(e)}")
                    error_message = "啊...Rice 好像有點累了，可以晚點再吐槽嗎？"
                    line_bot_api.reply_message_with_http_info(
                        ReplyMessageRequest(
                            reply_token=event.reply_token,
                            messages=[TextMessage(text=error_message)]
                        )
                    )
            else:
                # 一般對話處理
                if chat_history[user_id] == []: # 如果該用戶沒有對話歷史，則初始化
                    chat_history[user_id] = [system_prompt]
                
                # 一般對話與(draw, riceshower, roast )以外的指令處理
                chat_history[user_id].append({"role": "user", "content": content})

                if len(chat_history[user_id]) > 11:
                    chat_history[user_id] = [system_prompt] + chat_history[user_id][-10:]

                response = client.chat.completions.create( # 使用 OpenAI API 發送聊天請求
                    model="gpt-4o",
                    messages=chat_history[user_id] # 傳入該用戶的對話歷史
                )
            
                ai_reply = prefix + response.choices[0].message.content.strip() # reply 文本為指令前綴加上 AI 回應內容
                chat_history[user_id].append({"role": "assistant", "content": ai_reply})

                line_bot_api.reply_message_with_http_info(
                    ReplyMessageRequest(
                        reply_token=event.reply_token,
                        messages=[TextMessage(text=ai_reply)]
                    )
                )
        except Exception as e:
            print(f"OpenAI API 呼叫失敗: {str(e)}")
            error_message = "啊，Rice剛剛在想事情，沒聽清楚你說什麼，可以再說一次嗎？"
            line_bot_api.reply_message_with_http_info(
                ReplyMessageRequest(
                    reply_token=event.reply_token,
                    messages=[TextMessage(text=error_message)]
                )
            )

# sticker message event handler
@handler.add(MessageEvent, message=StickerMessageContent)
def handle_sticker_message(event):
    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)

        # 加入記錄來確認函式被調用
        print("Handling sticker message...")
        # 圖片必須要有公開的 URL，Line 才能存取
        # 這裡需要將你的本地圖片上傳到可公開訪問的位置，Google Drive 圖片需直接存取連結
        # image_url = "https://imgur.com/a/vnmgfZJ"
        preview_image_url = "https://i.redd.it/40w60wq6lmx91.jpg"

        # 上傳圖片到 Imgur  
        local_image_path = "stickers\\so_happy.jpg"  # 本地圖片路徑
        image_url = imgur_handler.upload_image(local_image_path)
        

        try:
            line_bot_api.reply_message_with_http_info(
                ReplyMessageRequest(
                    reply_token=event.reply_token,
                    messages=[
                        TextMessage(text="Received your sticker!"),
                        ImageMessage(
                            original_content_url=image_url,
                            preview_image_url=preview_image_url 
                        )
                    ]
                )
            )
        except Exception as e:
            # 上傳失敗時發送錯誤訊息
            line_bot_api.reply_message_with_http_info(
                ReplyMessageRequest(
                    reply_token=event.reply_token,
                    messages=[TextMessage(text="圖片上傳失敗，請稍後再試")]
                )
            )

# 建立 rich menu 的函式
def create_rich_menu():

    headers={
        'Authorization': 'Bearer ' + CHANNEL_ACCESS_TOKEN, # 使用 Bearer Token 認證
        'Content-Type': 'application/json' # 設定內容類型為 JSON
    }

    body = {
        "size": {
            "width": 2500,
            "height": 843
        },
        "selected": True,
        "name": "圖文選單 1",
        "chatBarText": "查看更多資訊",
        "areas": [
            {
                "bounds": {
                    "x": 38,
                    "y": 93,
                    "width": 754,
                    "height": 674
                },
                "action": {
                    "type": "message",
                    "text": "joke:周星馳電影中方唐鏡笑話"  # 使用者點擊後會發送 "joke:" 指令
                }
            },
            {
                "bounds": {
                    "x": 852,
                    "y": 97,
                    "width": 792,
                    "height": 657
                },
                "action": {
                    "type": "message",
                    "text": "travel: 京都"  # 使用者點擊後會發送 "travel:" 指令
                }
            },
            {
                "bounds": {
                    "x": 1720,
                    "y": 93,
                    "width": 780,
                    "height": 687
                },
                "action": {
                    "type": "message",
                    "text": "food:Rice 推薦的美食"  # 使用者點擊後會發送 "food:" 指令
                }
            }
        ]
    }

    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client) # create an instance of MessagingApi，目的是用來呼叫 LINE Messaging API
        line_bot_blob_api = MessagingApiBlob(api_client) # create a MessagingApiBlob instance
        
        # Step 1: Create rich menu
        response= requests.post(
            'https://api.line.me/v2/bot/richmenu',
            headers=headers,
            data=json.dumps(body).encode('utf-8') # 將 body 轉換為 JSON 字符串
        )
        response = response.json() # 將回應轉換為 JSON 格式
        print(response) # 印出回應內容以便除錯
        rich_menu_id = response['richMenuId'] # 取得 rich menu ID
        
        # Step 2: Upload rich menu image
        with open("static//rich_menu_image.png", "rb") as image:
            line_bot_blob_api.set_rich_menu_image(
                rich_menu_id=rich_menu_id, # 指定要上傳圖片的 rich menu ID
                body= bytearray(image.read()), # 讀取圖片內容
                _headers={
                    'Content-Type': 'image/png' # 設定圖片類型為 PNG
                }
            )
        print(f"Rich menu image uploaded successfully for ID: {rich_menu_id}")
            
        
        # Step 3: Set rich menu as default
        line_bot_api.set_default_rich_menu(
            rich_menu_id=rich_menu_id # 設定剛剛建立的 rich menu 為預設選單
        )
        
        return rich_menu_id

# call create_rich_menu function to create rich menu
create_rich_menu()


if __name__ == "__main__":
    
    app.run()

