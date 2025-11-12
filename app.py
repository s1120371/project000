# app.py (方法四：批次辨識 + 整合式營養紀錄)
# 包含使用者後台、AI 問答、BMI、成就、YOLOv8辨識 (雙模型)、FatSecret營養查詢、營養歷史紀錄(整合)

from flask import Flask, request, jsonify, send_from_directory, render_template
import firebase_admin
from firebase_admin import credentials, auth, firestore
from firebase_admin.firestore import Query
import os
import requests
from llama_cpp import Llama
from datetime import datetime, timedelta
from requests_oauthlib import OAuth1
from deep_translator import GoogleTranslator
import re

# --- 辨識功能所需套件 ---
from ultralytics import YOLO
from PIL import Image
import io
import cv2
import numpy as np
import base64
import uuid

# --- ★ (新) 導入 LangChain 和 Google AI 套件 ---
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv  # 導入 dotenv

# --- ★★★ 關鍵修正：必須先「載入」才能「讀取」 ★★★ ---

# 1. 先呼叫 load_dotenv()，它會去讀取 .env 檔案
load_dotenv() 

# 2. 現在才使用 os.getenv() 從環境變數中讀取
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# 3. 執行檢查
if not GEMINI_API_KEY:
    print("="*50)
    print("錯誤：在 .env 檔案中找不到 'GEMINI_API_KEY'！")
    print("請執行以下檢查：")
    print("1. 確保 .env 檔案與 app.py 在同一個資料夾。")
    print("2. 確保檔案名稱是 .env (沒有 .txt 副檔名)。")
    print("3. 確保 .env 檔案內容是 GEMINI_API_KEY=... (沒有引號或空格)。")
    print("="*50)
    exit() # 找不到金鑰，直接停止程式
else:
    print("✅ 成功從 .env 載入 GEMINI_API_KEY。")
    # 只有在成功找到 Key 之後，才設定這行
    os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY
# -----------------------------------

# ------------------ 初始化設定 ------------------
app = Flask(__name__)

# --- 辨識功能資料夾設定 ---
UPLOAD_FOLDER = os.path.join("static", "uploads")
RESULT_FOLDER = os.path.join("static", "results")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
# ---------------------------------

# --- Firebase 初始化 ---
if not os.path.exists('serviceAccountKey.json'):
    print("錯誤：找不到 'serviceAccountKey.json' 檔案！")
    exit()
cred = credentials.Certificate('serviceAccountKey.json')
firebase_admin.initialize_app(cred)
db = firestore.client()

# --- GGUF AI 模型載入 ---
print("正在載入 GGUF AI 模型...")
model_filename = "Meta-Llama-3-8B-Instruct.Q4_K_M.gguf"
if not os.path.exists(model_filename):
    print(f"錯誤：找不到模型檔案 '{model_filename}'！")
    exit()
llm = Llama(model_path=model_filename, n_ctx=4096, n_gpu_layers=0, verbose=True)
print(f"GGUF AI 模型 '{model_filename}' 載入完成！")

# --- YOLOv8 辨識模型載入 (載入兩個模型) ---
print("正在載入 YOLOv8 辨識模型 A...")
model_A_path = os.path.join(os.getcwd(), "best_food_model_A.pt") # <--- 您的第一個模型
if not os.path.exists(model_A_path):
    print(f"錯誤：找不到模型檔案 'best_food_model_A.pt'！")
    exit()
model_A = YOLO(model_A_path)
print("YOLOv8 辨識模型 A 載入完成！")

print("正在載入 YOLOv8 辨識模型 B...")
model_B_path = os.path.join(os.getcwd(), "best_food_model_B.pt") # <--- 您的第二個模型
if not os.path.exists(model_B_path):
    print(f"錯誤：找不到模型檔案 'best_food_model_B.pt'！")
    exit()
model_B = YOLO(model_B_path)
print("YOLOv8 辨識模型 B 載入完成！")
# ---------------------------------

# --- API 金鑰設定 ---
FIREBASE_WEB_API_KEY = "AIzaSyCCNPhST7sScxFdSZJ6-NbxKgqrSYOzes4"
CONSUMER_KEY = "ba46d91448844c4ba3aa81ff09e605df"
CONSUMER_SECRET = "60302dd6c9c240d1a1118a75677e3967"
API_BASE = "https://platform.fatsecret.com/rest/server.api"
# ---------------------------------

# --- 食物中英文字典 (定義兩個字典) ---
# (字典內容省略，與您上一個版本相同)
item_translation_A = {
    'rice': '米飯',
    'fried cabbage': '炒高麗菜',
    'scrambled eggs with tomatoes': '番茄炒蛋',
    'stir fried water spinach': '炒空心菜',
    'dongpo pork': '東坡肉',
    'pan fried salmon': '煎鮭魚',
    'pumpkin scrambled eggs': '南瓜炒蛋',
    'braised bamboo shoots': '滷筍絲',
    'stir fried enoki mushrooms': '炒金針菇',
    'stir fried rapeseed': '炒油菜',
    'stir-fried rapeseed': '炒油菜', # 處理同義詞
    'Fried sausages': '煎香腸',
    'Stir-fried bean sprouts': '炒豆芽菜',
    'Stir fried bean sprouts': '炒豆芽菜', # 處理同義詞
    'Stir-fried carrots': '炒紅蘿蔔',
    'Stir fried carrots': '炒紅蘿蔔', # 處理同義詞
}
item_translation_B = {
    'rice': '米飯', 'eels on rice': '鰻魚飯', 'pilaf': '抓飯', 'chicken-\'n\'-egg on rice': '親子丼',
    'pork cutlet on rice': '豬排飯', 'beef curry': '牛肉咖哩', 'sushi': '壽司', 'chicken rice': '雞肉飯',
    'fried rice': '炒飯', 'tempura bowl': '天婦羅丼', 'bibimbap': '韓式拌飯', 'toast': '吐司',
    'croissant': '可頌', 'roll bread': '餐包', 'raisin bread': '葡萄乾麵包', 'chip butty': '薯條三明治',
    'hamburger': '漢堡', 'pizza': '披薩', 'sandwiches': '三明治', 'udon noodle': '烏龍麵',
    'tempura udon': '天婦羅烏龍麵', 'soba noodle': '蕎麥麵', 'ramen noodle': '拉麵',
    'beef noodle': '牛肉麵', 'tensin noodle': '天津麵', 'fried noodle': '炒麵', 'spaghetti': '義大利麵',
    'Japanese-style pancake': '日式大阪燒', 'takoyaki': '章魚燒', 'gratin': '焗烤',
    'sauteed vegetables': '炒蔬菜', 'croquette': '可樂餅', 'grilled eggplant': '烤茄子',
    'sauteed spinach': '炒菠菜', 'vegetable tempura': '蔬菜天婦羅', 'miso soup': '味噌湯',
    'potage': '法式濃湯', 'sausage': '香腸', 'oden': '關東煮', 'omelet': '歐姆蛋',
    'ganmodoki': '日式炸豆腐丸', 'jiaozi': '餃子', 'stew': '燉菜', 'teriyaki grilled fish': '照燒烤魚',
    'fried fish': '炸魚', 'grilled salmon': '烤鮭魚', 'salmon meuniere': '法式香煎鮭魚',
    'sashimi': '生魚片', 'grilled pacific saury': '烤秋刀魚', 'sukiyaki': '壽喜燒',
    'sweet and sour pork': '糖醋排骨', 'lightly roasted fish': '炙燒魚',
    'steamed egg hotchpotch': '茶碗蒸', 'tempura': '天婦羅', 'fried chicken': '炸雞',
    'sirloin cutlet': '沙朗牛排', 'nanbanzuke': '南蠻漬', 'boiled fish': '水煮魚',
    'seasoned beef with potatoes': '馬鈴薯燉牛肉', 'hambarg steak': '漢堡排', 'steak': '牛排',
    'dried fish': '魚乾', 'ginger pork saute': '薑汁燒肉', 'spicy chili-flavored tofu': '麻婆豆腐',
    'yakitori': '日式烤雞串', 'cabbage roll': '高麗菜捲', 'egg sunny-side up': '太陽蛋', 'natto': '納豆',
    'cold tofu': '冷豆腐', 'egg roll': '蛋捲', 'chilled noodle': '涼麵',
    'stir-fried beef and peppers': '青椒炒牛肉', 'simmered pork': '控肉',
    'boiled chicken and vegetables': '水煮雞肉蔬菜', 'sashimi bowl': '生魚片丼', 'sushi bowl': '壽司丼',
    'fish-shaped pancake with bean jam': '鯛魚燒', 'shrimp with chili sauce': '乾燒蝦仁',
    'roast chicken': '烤雞', 'steamed meat dumpling': '燒賣', 'omelet with fried rice': '蛋包飯',
    'cutlet curry': '炸豬排咖哩', 'spaghetti meat sauce': '義大利肉醬麵', 'fried shrimp': '炸蝦',
    'potato salad': '馬鈴薯沙拉', 'green salad': '生菜沙拉', 'macaroni salad': '通心粉沙拉',
    'Japanese tofu and vegetable chowder': '日式豆腐蔬菜雜燴', 'pork miso soup': '豬肉味噌湯',
    'chinese soup': '中式湯', 'beef bowl': '牛丼', 'kinpira-style sauteed burdock': '金平牛蒡',
    'rice ball': '飯糰', 'pizza toast': '披薩吐司', 'dipping noodles': '沾麵', 'hot dog': '熱狗',
    'french fries': '薯條', 'mixed rice': '什錦飯', 'goya chanpuru': '沖繩苦瓜炒什錦',
    'green curry': '綠咖哩', 'okinawa soba': '沖繩麵', 'mango pudding': '芒果布丁',
    'almond jelly': '杏仁豆腐', 'jjigae': '韓式鍋物', 'dak galbi': '辣炒雞排', 'dry curry': '乾咖哩',
    'kamameshi': '釜飯', 'rice vermicelli': '米粉', 'paella': '西班牙海鮮飯', 'tanmen': '湯麵',
    'kushikatu': '串炸', 'yellow curry': '黃咖哩', 'pancake': '鬆餅', 'champon': '強棒麵',
    'crepe': '可麗餅', 'tiramisu': '提拉米蘇', 'waffle': '鬆餅', 'rare cheese cake': '生乳酪蛋糕',
    'shortcake': '草莓蛋糕', 'chop suey': '炒雜碎', 'twice cooked pork': '回鍋肉',
    'mushroom risotto': '蘑菇燉飯', 'samul': '四物', 'zoni': '日式年糕湯', 'french toast': '法式吐司',
    'fine white noodles': '素麵', 'minestrone': '義大利蔬菜湯', 'pot au feu': '法式燉菜鍋',
    'chicken nugget': '雞塊', 'namero': '生魚たたき', 'french bread': '法國麵包', 'rice gruel': '粥',
    'broiled eel bowl': '鰻魚丼', 'clear soup': '清湯', 'yudofu': '湯豆腐', 'mozuku': '水雲',
    'inarizushi': '稻荷壽司', 'pork loin cutlet': '里肌豬排', 'pork fillet cutlet': '菲力豬排',
    'chicken cutlet': '雞排', 'ham cutlet': '火腿排', 'minced meat cutlet': '絞肉排',
    'thinly sliced raw horsemeat': '生馬肉', 'bagel': '貝果', 'scone': '司康', 'tortilla': '墨西哥薄餅',
    'tacos': '塔可', 'nachos': '墨西哥玉米片', 'meat loaf': '肉塊', 'scrambled egg': '炒蛋',
    'rice gratin': '焗烤飯', 'lasagna': '千層麵', 'Caesar salad': '凱薩沙拉', 'oatmeal': '燕麥片',
    'fried pork dumplings served in soup': '湯餃', 'oshiruko': '日式紅豆湯', 'muffin': '瑪芬',
    'popcorn': '爆米花', 'cream puff': '泡芙', 'doughnut': '甜甜圈', 'apple pie': '蘋果派',
    'parfait': '百匯', 'fried pork in scoop': '炸豬排', 'lamb kebabs': '羊肉串',
    'dish consisting of stir-fried potato, eggplant and green pepper': '地三鮮', 'roast duck': '烤鴨',
    'hot pot': '火鍋', 'pork belly': '五花肉', 'xiao long bao': '小籠包', 'moon cake': '月餅',
    'custard tart': '蛋塔', 'beef noodle soup': '牛肉麵', 'pork cutlet': '豬排',
    'minced pork rice': '滷肉飯', 'fish ball soup': '魚丸湯', 'oyster omelette': '蚵仔煎',
    'glutinous oil rice': '油飯', 'turnip pudding': '蘿蔔糕', 'stinky tofu': '臭豆腐',
    'lemon fig jelly': '檸檬無花果凍', 'khao soi': '泰北咖哩麵', 'Sour prawn soup': '泰式酸辣蝦湯',
    'Thai papaya salad': '涼拌青木瓜', 'boned, sliced Hainan-style chicken with marinated rice': '海南雞飯',
    'hot and sour, fish and vegetable ragout': '酸辣魚蔬菜燴', 'stir-fried mixed vegetables': '炒時蔬',
    'beef in oyster sauce': '蠔油牛肉', 'pork satay': '豬肉沙嗲', 'spicy chicken salad': '涼拌辣雞',
    'noodles with fish curry': '魚咖哩麵', 'Pork Sticky Noodles': '豬肉羹麵', 'Pork with lemon': '檸檬豬肉',
    'stewed pork leg': '燉豬腳', 'charcoal-boiled pork neck': '炭烤豬頸肉', 'fried mussel pancakes': '泰式淡菜煎',
    'Deep Fried Chicken Wing': '炸雞翅', 'Barbecued red pork in sauce with rice': '叉燒飯',
    'Rice with roast duck': '燒鴨飯', 'Rice crispy pork': '脆皮燒肉飯', 'Wonton soup': '雲吞湯',
    'Chicken Rice Curry With Coconut': '椰漿咖哩雞飯', 'Crispy Noodles': '廣州炒麵',
    'Egg Noodle In Chicken Yellow Curry': '黃咖哩雞肉麵', 'coconut milk soup': '椰奶湯',
    'pho': '越南河粉', 'Hue beef rice vermicelli soup': '順化牛肉米粉湯', 'Vermicelli noodles with snails': '螺螄粉',
    'Fried spring rolls': '炸春捲', 'Steamed rice roll': '腸粉', 'Shrimp patties': '蝦餅',
    'ball shaped bun with pork': '肉包', 'Coconut milk-flavored crepes with shrimp and beef': '越南煎餅',
    'Small steamed savory rice pancake': '越式碗粿', 'Glutinous Rice Balls': '湯圓',
    'loco moco': '夏威夷漢堡飯', 'haupia': '夏威夷椰子布丁', 'malasada': '夏威夷甜甜圈',
    'laulau': '勞勞', 'spam musubi': '午餐肉飯糰', 'oxtail soup': '牛尾湯', 'adobo': '菲律賓醋烹雞',
    'lumpia': '菲律賓春捲', 'brownie': '布朗尼', 'churro': '吉拿棒', 'jambalaya': '什錦飯',
    'nasi goreng': '印尼炒飯', 'ayam goreng': '印尼炸雞', 'ayam bakar': '印尼烤雞', 'bubur ayam': '印尼雞粥',
    'gulai': '古來', 'laksa': '叻沙', 'mie ayam': '雞肉麵', 'mie goreng': '印尼炒麵', 'nasi campur': '印尼什錦飯',
    'nasi padang': '巴東飯', 'nasi uduk': '椰漿飯', 'babi guling': '烤乳豬', 'kaya toast': '咖椰吐司',
    'bak kut teh': '肉骨茶', 'curry puff': '咖哩餃', 'chow mein': '炒麵', 'zha jiang mian': '炸醬麵',
    'kung pao chicken': '宮保雞丁', 'crullers': '油條', 'eggplant with garlic sauce': '魚香茄子',
    'three cup chicken': '三杯雞', 'bean curd family style': '家常豆腐',
    'salt & pepper fried shrimp with shell': '椒鹽蝦', 'baked salmon': '烤鮭魚',
    'braised pork meat ball with napa cabbage': '紅燒獅子頭', 'winter melon soup': '冬瓜湯',
    'steamed spareribs': '蒸排骨', 'chinese pumpkin pie': '南瓜餅', 'eight treasure rice': '八寶飯',
    'hot & sour soup': '酸辣湯'
}
# ---------------------------------


# ===============================================================
# 輔助函式 (保持不變)
# ===============================================================

# --- 輔助函式：驗證 Token ---
def verify_token(request):
    id_token = request.headers.get('Authorization', '').split('Bearer ')[-1]
    if not id_token:
        return None, (jsonify({'error': '缺少驗證資訊'}), 401)
    try:
        decoded_token = auth.verify_id_token(id_token)
        return decoded_token, None
    except Exception as e:
        return None, (jsonify({'error': f'Token 無效或過期: {e}'}), 401)


# --- 核心AI功能函式 (聊天用) ---
def generate_llama_advice(user_query, user_profile, history_messages=None):
    # (您原本的 chat AI 函式... 內容保持不變，省略)
    system_prompt = """你是一個專業又親切的「健康管家 AI」。
...
"""
    # (省略...)
    return response['choices'][0]['message']['content']


# --- ★ (新) 核心AI功能函式 (飲食評價用 - 改用 LangChain + Gemini) ---
def generate_gemini_evaluation(user_profile, diet_data):
    """
    根據使用者資料和單次飲食紀錄，使用 LangChain + Gemini 產生個人化評價。
    """
    print("生成 AI 飲食評價 (使用 Google Gemini)...")

    # (處理 user_profile 和 diet_data 的程式碼... 保持不變)
    # 1. 處理使用者資料
    goal_map = {
        'weight-loss': '減重', 'muscle-gain': '增肌',
        'control-sugar': '控制血糖', 'general-health': '維持一般健康'
    }
    diet_map = {
        'omnivore': '一般葷食', 'lacto-ovo': '蛋奶素', 'vegan': '全素'
    }
    profile_text = f"""
- 健康目標: {goal_map.get(user_profile.get('goal'), '未設定')}
- 飲食習慣: {diet_map.get(user_profile.get('diet'), '未設定')}
- 已知過敏原: {', '.join(user_profile.get('allergens', [])) or '無'}
"""
    # 2. 處理飲食數據
    diet_text = f"""
- 食物清單: {', '.join(diet_data.get('foods_list', []))}
- 總熱量: {diet_data.get('total_calories')} kcal
- 總蛋白質: {diet_data.get('total_protein')} g
- 總脂肪: {diet_data.get('total_fat')} g
- 總碳水: {diet_data.get('total_carbs')} g
"""

    # --- ★★★ 關鍵修正：暫時移除 Firebase 的環境變數 ★★★ ---
    original_creds = os.environ.pop('GOOGLE_APPLICATION_CREDENTIALS', None)
    
    try:
        # 組件 A: 語言模型
        # 我們強制傳入 API Key，並使用您在 hello_langchain.py 中
        # 已確認可以運作的模型名稱 "models/gemini-1.5-flash"
        llm = ChatGoogleGenerativeAI(
            model="models/gemini-2.5-flash", 
            google_api_key=GEMINI_API_KEY  # 確保您在檔案頂部定義了 GEMINI_API_KEY
        )
        
        # 組件 B: 提示模板 (PromptTemplate)
        system_prompt = """你是一個專業的「健康管家 AI」。你的任務是根據使用者的「個人資料」和他們「剛剛儲存的飲食紀錄」，提供一段簡潔、專業、個人化的評價與建議。

你必須嚴格遵守以下規則：
1.  **唯一語言：你的所有回答都必須使用繁體中文。**
2.  **格式限制：請輸出純文字。絕對禁止使用星號 (*) 或 Markdown 格式（例如 **粗體**）。請不要使用項目符號，僅使用數字編號 (1. 2. 3.) 或分段。**
3.  **核心任務：** 根據使用者的「健康目標」來評價這餐的營養（熱量、蛋白質等）是否合適。
4.  **過敏原檢查：** **務必**檢查「食物清單」中是否有任何食物 *可能* 觸發使用者的「已知過敏原」。如果有，必須提出明確警告。
5.  **語氣：** 保持溫暖、鼓勵，像一個專業的營養師。
6.  **結構：**
    * 先給予一點鼓勵（例如：很高興看到您記錄飲食！）。
    * 針對「健康目標」進行分析。
    * (如果需要) 提出「過敏原警告」。
    * 最後提供 1-2 個具體的改進建議。
"""
        
        prompt_template_str_with_vars = f"""{system_prompt}
---
**我的個人資料：**
{{profile}}
---
**我儲存的飲食紀錄：**
{{diet}}
---
請開始你的評價與建議：
"""
        
        prompt = ChatPromptTemplate.from_template(prompt_template_str_with_vars)
        output_parser = StrOutputParser()
        chain = prompt | llm | output_parser

        # 5. 執行「鏈」
        print("正在呼叫 Google Gemini API (強制使用 AI Studio Key)...")
        result = chain.invoke({
            "profile": profile_text,
            "diet": diet_text
        })
        
        # --- ★★★ 關鍵修正：恢復環境變數 ★★★ ---
        if original_creds:
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = original_creds
            
        return result

    except Exception as e:
        print(f"AI 評價生成失敗 (Gemini): {e}")
        # --- ★★★ 關鍵修正：恢復環境變數 (即使失敗也要恢復) ★★★ ---
        if original_creds:
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = original_creds
            
        if "API key not valid" in str(e):
            return "抱歉，AI 評價服務無法連線。請檢查伺服器上的 GEMINI_API_KEY 是否設定正確。"
        return f"抱歉，AI 評價服務目前暫時無法連線：{e}"

# --- 核心AI功能函式 ---
def generate_llama_advice(user_query, user_profile, history_messages=None):
    # (省略...)
    system_prompt = """你是一個專業又親切的「健康管家 AI」。

你必須嚴格遵守以下規則：
1.  **首要規則：你的唯一輸出語言是繁體中文。絕對禁止使用英文或其他任何語言作答。**
2.  **重要規則：對話紀錄的優先級最高。** 如果使用者在對話中提到了新的健康資訊（例如新的過敏原、改變的飲食目標等），你必須將這些新資訊視為對個人基本資料的「即時更新」。當基本資料和對話紀錄有衝突時，永遠以對話紀錄中的最新說法為準。
3.  你的任務是根據使用者的個人健康資料和他們提出的問題，提供準確、個人化且安全的健康飲食建議。
4.  你的回答應該要自然地**整合**使用者的個人資料，而不是分開條列。
5.  如果使用者的問題涉及到他的過敏原（無論是基本資料中的還是對話中提到的），請**務必**在回答中提出明確的安全警告。
6.  保持回覆簡潔、溫暖、易於理解。
"""
    goal_map = {
        'weight-loss': '減重', 'muscle-gain': '增肌',
        'control-sugar': '控制血糖', 'general-health': '維持一般健康'
    }
    diet_map = {
        'omnivore': '一般葷食', 'lacto-ovo': '蛋奶素', 'vegan': '全素'
    }
    profile_text = f"""
- 健康目標: {goal_map.get(user_profile.get('goal'), '未設定')}
- 飲食習慣: {diet_map.get(user_profile.get('diet'), '未設定')}
- 已知過敏原: {', '.join(user_profile.get('allergens', [])) or '無'}
"""
    messages = [{"role": "system", "content": system_prompt + "\n這是使用者的基本資料（請記住，對話紀錄優先級更高）：\n" + profile_text}]

    if history_messages:
        messages.extend(history_messages)

    messages.append({"role": "user", "content": user_query})

    response = llm.create_chat_completion(
        messages=messages, max_tokens=512, temperature=0.7
    )
    return response['choices'][0]['message']['content']

# --- 營養查詢輔助函式 ---
def translate_text(text, target='zh-TW'):
    if not text: return ""
    try: return GoogleTranslator(source='auto', target=target).translate(text)
    except Exception: return text

def parse_index(description):
    index = {}
    cal_match = re.search(r"Calories:\s*([\d.]+)kcal", description)
    fat_match = re.search(r"Fat:\s*([\d.]+)g", description)
    carb_match = re.search(r"Carbs:\s*([\d.]+)g", description)
    protein_match = re.search(r"Protein:\s*([\d.]+)g", description)

    if cal_match: index["熱量 (kcal)"] = f"{float(cal_match.group(1)):.2f}"
    if fat_match: index["脂肪 (g)"] = f"{float(fat_match.group(1)):.2f}"
    if carb_match: index["碳水化合物 (g)"] = f"{float(carb_match.group(1)):.2f}"
    if protein_match: index["蛋白質 (g)"] = f"{float(protein_match.group(1)):.2f}"
    return index

def search_food_index(food_name):
    auth = OAuth1(CONSUMER_KEY, CONSUMER_SECRET)
    params = {"method": "foods.search", "search_expression": food_name, "format": "json", "max_results": 1}
    try:
        res = requests.get(API_BASE, params=params, auth=auth)
        res.raise_for_status()
        data = res.json()
        if "foods" in data and "food" in data["foods"] and data["foods"]["food"]:
            food_item = data["foods"]["food"]
            if isinstance(food_item, list): food_item = food_item[0]
            
            # ★ 修正 1：啟用翻譯
            food_name_cn = translate_text(food_item.get("food_name"))
            
            desc_cn = translate_text(food_item.get("food_description"))
            index_data = parse_index(food_item.get("food_description", ""))
            
            # ★ 修正 2：回傳中文名稱
            return {"food_name": food_name_cn, "food_description": desc_cn, "index": index_data}
            
    except requests.exceptions.RequestException as e: print(f"查詢 API 時發生錯誤 ({food_name}): {e}")
    return None
# ---------------------------------


# ===============================================================
# 路由 (移除 /nutrition-history)
# ===============================================================

# --- 靜態網頁路由 ---
@app.route('/')
def index(): return render_template("index.html")
@app.route('/home')
def home(): return render_template("home.html")
@app.route('/login')
def login(): return render_template("login.html")
@app.route('/nutrition', methods=['GET'])
def nutrition(): return render_template("nutrition.html")
@app.route('/edit-profile')
def edit_profile(): return render_template("edit-profile.html")
@app.route('/bmi')
def bmi(): return render_template("bmi.html")
@app.route('/achievements')
def achievements(): return render_template("achievements.html")

# --- ★ (移除) /nutrition-history 路由已被移除 ---
# (此處無程式碼)

# --- YOLOv8 辨識與營養查詢路由 (多張圖片) ---
@app.route("/predict", methods=["POST"])
def predict():
    files = request.files.getlist('image')
    
    if not files or all(f.filename == '' for f in files):
        return render_template("nutrition.html", error="未上傳任何圖片", all_results=[])

    all_results = []

    try:
        for file in files:
            if file.filename == '':
                continue

            # --- 影像處理 ---
            image_bytes = file.read()
            pil_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            
            # --- 儲存原始圖片 ---
            filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
            img_path = os.path.join(UPLOAD_FOLDER, filename)
            pil_image.save(img_path) 
            print(f"✅ 上傳圖片儲存於: {img_path}")
            uploaded_image_web = img_path.replace("\\", "/")

            cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

            image_detected_foods = [] 
            image_food_infos = []     
            seen_foods_eng_names = set() 

            # --- 執行模型 A 預測 ---
            print("執行模型 A 預測...")
            results_A = model_A(pil_image)
            if results_A and results_A[0].boxes:
                print(f"模型 A 找到 {len(results_A[0].boxes)} 個潛在物件")
                for box in results_A[0].boxes:
                    conf = round(float(box.conf[0]), 2)
                    if conf < 0.5: continue

                    cls = int(box.cls[0])
                    # ★ 修正 1：永遠使用英文名稱
                    eng_name_A = model_A.names[cls].lower().replace("_", " ").replace("-", " ")
                    
                    print(f"  模型 A: {eng_name_A} (信心度: {conf})")
                    # ★ 修正 2：偵測清單使用英文
                    image_detected_foods.append({'name': eng_name_A, 'confidence': f"{conf:.2f}", 'source': 'A'})

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    # ★ 修正 3：畫框標籤使用英文
                    label_A = f"{eng_name_A} {conf} (A)" 
                    cv2.rectangle(cv_image, (x1, y1), (x2, y2), (255, 0, 0), 2) # 畫框 (藍色)

                    # 畫白框文字
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.6
                    pos = (x1, y1 - 10) 
                    border_color = (255, 255, 255) 
                    main_color = (255, 0, 0)
                    cv2.putText(cv_image, label_A, pos, font, font_scale, border_color, 5, cv2.LINE_AA)
                    cv2.putText(cv_image, label_A, pos, font, font_scale, main_color, 2, cv2.LINE_AA)

                    if eng_name_A not in seen_foods_eng_names:
                        seen_foods_eng_names.add(eng_name_A)
                        
                        # ★ 修正 4：使用英文名稱 (eng_name_A) 去搜尋 API
                        index_data_A = search_food_index(eng_name_A) 
                        
                        if index_data_A:
                            # ★ 修正 5：使用 API 傳回的中文名稱 (index_data_A['food_name'])
                            image_food_infos.append({
                                'food_name': index_data_A['food_name'], 'confidence': f"{conf:.2f}",
                                'food_description': index_data_A['food_description'],
                                'index': index_data_A['index'], 'source': 'A'
                            })
                        else:
                            # Fallback：如果 API 查不到，卡片也使用英文
                            image_food_infos.append({
                                'food_name': eng_name_A, 'confidence': f"{conf:.2f}",
                                'food_description': "查無此食物的詳細營養資訊。",
                                'index': None, 'source': 'A'
                            })

            # --- 執行模型 B 預測 ---
            print("執行模型 B 預測...")
            results_B = model_B(pil_image)
            if results_B and results_B[0].boxes:
                print(f"模型 B 找到 {len(results_B[0].boxes)} 個潛在物件")
                for box in results_B[0].boxes:
                    conf = round(float(box.conf[0]), 2)
                    if conf < 0.2: continue

                    cls = int(box.cls[0])
                    # ★ 修正 1：永遠使用英文名稱
                    eng_name_B = model_B.names[cls].lower().replace("_", " ").replace("-", " ")
                    
                    print(f"  模型 B: {eng_name_B} (信心度: {conf})")
                    # ★ 修正 2：偵測清單使用英文
                    image_detected_foods.append({'name': eng_name_B, 'confidence': f"{conf:.2f}", 'source': 'B'})

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    # ★ 修正 3：畫框標籤使用英文
                    label_B = f"{eng_name_B} {conf} (B)" 
                    text_y = y1 - 30 if y1 > 30 else y1 + 15
                    cv2.rectangle(cv_image, (x1, y1), (x2, y2), (255, 0, 0), 2) # 畫框 (藍色)

                    # 畫白框文字 
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.6
                    pos = (x1, text_y)
                    border_color = (255, 255, 255) 
                    main_color = (255, 0, 0) # 藍色
                    cv2.putText(cv_image, label_B, pos, font, font_scale, border_color, 5, cv2.LINE_AA)
                    cv2.putText(cv_image, label_B, pos, font, font_scale, main_color, 2, cv2.LINE_AA)

                    if eng_name_B not in seen_foods_eng_names:
                        seen_foods_eng_names.add(eng_name_B)
                        
                        # ★ 修正 4：使用英文名稱 (eng_name_B) 去搜尋 API
                        index_data_B = search_food_index(eng_name_B)
                        
                        if index_data_B:
                            # ★ 修正 5：使用 API 傳回的中文名稱 (index_data_B['food_name'])
                            image_food_infos.append({
                                'food_name': index_data_B['food_name'], 'confidence': f"{conf:.2f}",
                                'food_description': index_data_B['food_description'],
                                'index': index_data_B['index'], 'source': 'B'
                            })
                        else:
                            # Fallback：如果 API 查不到，卡片也使用英文
                            image_food_infos.append({
                                'food_name': eng_name_B, 'confidence': f"{conf:.2f}",
                                'food_description': "查無此食物的詳細營養資訊。",
                                'index': None, 'source': 'B'
                            })

            if not image_detected_foods:
                image_detected_foods = [{'name': '未偵測到任何食物', 'confidence': 'N/A', 'source': '-'}]

            result_filename = f"result_combined_{uuid.uuid4()}.jpg"
            result_img_path = os.path.join(RESULT_FOLDER, result_filename)
            cv2.imwrite(result_img_path, cv_image)
            print(f"✅ 合併偵測結果儲存於: {result_img_path}")
            result_image_web = result_img_path.replace("\\", "/")

            all_results.append({
                "uploaded_image": uploaded_image_web,
                "result_image": result_image_web,
                "detected_foods": image_detected_foods,
                "food_infos": image_food_infos
            })

        return render_template("nutrition.html", all_results=all_results)

    except Exception as e:
        print(f"辨識過程中發生錯誤: {e}")
        # *** 確保即使出錯，頁面也能載入歷史紀錄 ***
        return render_template("nutrition.html", error=f"處理過程中發生錯誤: {e}", all_results=[])
# ---------------------------------


# --- API 路由 (使用者後台) ---
# ( ... /api/register, /api/login, /ask, /api/chat-history, ... 保持不變 ... )
# ( ... /api/user-profile, /api/update-password, /api/bmi-records, ... 保持不變 ... )
# ( ... /api/achievement-goals, /api/achievement-records, ... 保持不變 ... )
# ( ... /api/nutrition-records (GET/POST), /api/nutrition-records/<record_id> (DELETE) 保持不變 ... )

# 複製所有您原本的 API 路由到這裡...
# (這裡我只保留幾個範例，請確保您複製了所有 API 路由)

@app.route('/api/register', methods=['POST'])
def api_register():
    data = request.json
    email, password = data.get('email'), data.get('password')
    if not email or not password: return jsonify({'error': '缺少 Email 或密碼'}), 400
    try:
        user = auth.create_user(email=email, password=password)
        user_info = {
            'username': data.get('username', ''), 'email': email, 'fullname': data.get('fullname', ''),
            'birthdate': data.get('birthdate', ''), 'gender': data.get('gender', ''),
            'allergens': data.get('allergens', []), 'diet': data.get('diet', ''), 'goal': data.get('goal', ''),
            'createdAt': firestore.SERVER_TIMESTAMP
        }
        db.collection('users').document(user.uid).set(user_info)
        return jsonify({'message': f'使用者 {user.email} 註冊成功', 'uid': user.uid}), 201
    except Exception as e: return jsonify({'error': f'註冊失敗: {str(e)}'}), 400

@app.route('/api/login', methods=['POST'])
def api_login():
    data = request.json
    email, password = data.get('email'), data.get('password')
    if not email or not password: return jsonify({'error': '缺少 Email 或密碼'}), 400
    try:
        rest_api_url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={FIREBASE_WEB_API_KEY}"
        payload = {'email': email, 'password': password, 'returnSecureToken': True}
        response = requests.post(rest_api_url, json=payload)
        response.raise_for_status()
        firebase_data = response.json()
        return jsonify({'idToken': firebase_data['idToken'], 'expiresIn': firebase_data['expiresIn']}), 200
    except requests.exceptions.HTTPError as err:
        error_message = err.response.json().get('error', {}).get('message', '未知錯誤')
        return jsonify({'error': f'登入失敗: {error_message}'}), 401
    except Exception as e: return jsonify({'error': f'伺服器發生錯誤: {str(e)}'}), 500

@app.route('/ask', methods=['POST'])
def ask():
    decoded_token, error = verify_token(request)
    if error: return error
    uid = decoded_token['uid']

    user_data = request.json
    user_question = user_data.get('question')
    if not user_question: return jsonify({'answer': '你沒有問問題喔！'})

    try:
        user_doc_ref = db.collection('users').document(uid)
        user_doc = user_doc_ref.get()
        if not user_doc.exists: return jsonify({'answer': '錯誤：找不到您的使用者設定檔。'})

        chat_history_ref = user_doc_ref.collection('chatHistory')
        docs = reversed(list(chat_history_ref.order_by("timestamp", direction=Query.DESCENDING).limit(10).stream()))

        history_messages = []
        for doc in docs:
            data = doc.to_dict()
            history_messages.append({"role": data['role'], "content": data['content']})

        answer = generate_llama_advice(user_question, user_doc.to_dict(), history_messages)

        chat_history_ref.add({
            'role': 'user',
            'content': user_question,
            'timestamp': firestore.SERVER_TIMESTAMP
        })
        chat_history_ref.add({
            'role': 'assistant',
            'content': answer,
            'timestamp': firestore.SERVER_TIMESTAMP
        })

        return jsonify({'answer': answer})
    except Exception as e:
        return jsonify({'answer': f'伺服器發生未預期的錯誤: {str(e)}'}), 500

@app.route('/api/chat-history', methods=['GET'])
def get_chat_history():
    decoded_token, error = verify_token(request)
    if error: return error
    uid = decoded_token['uid']

    try:
        chat_history_ref = db.collection('users').document(uid).collection('chatHistory')
        docs = chat_history_ref.order_by("timestamp").stream()

        records = [{'role': doc.to_dict()['role'], 'content': doc.to_dict()['content']} for doc in docs]
        return jsonify(records), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/user-profile', methods=['GET', 'POST'])
def user_profile():
    decoded_token, error = verify_token(request)
    if error: return error
    uid = decoded_token['uid']
    user_ref = db.collection('users').document(uid)

    if request.method == 'GET':
        user_doc = user_ref.get()
        profile_data = user_doc.to_dict() if user_doc.exists else {}
        profile_data['email'] = decoded_token.get('email', '')
        return jsonify(profile_data)

    if request.method == 'POST':
        update_data = request.json
        update_data.pop('email', None) 
        update_data['updatedAt'] = firestore.SERVER_TIMESTAMP
        user_ref.set(update_data, merge=True)
        return jsonify({'message': '資料更新成功'}), 200

@app.route('/api/update-password', methods=['POST'])
def update_password():
    decoded_token, error = verify_token(request)
    if error: return error
    uid = decoded_token['uid']

    new_password = request.json.get('password')
    if not new_password or len(new_password) < 6:
        return jsonify({'error': '密碼格式不符 (至少6位數)'}), 400
    try:
        auth.update_user(uid, password=new_password)
        return jsonify({'message': '密碼更新成功'}), 200
    except Exception as e: return jsonify({'error': f'密碼更新失敗: {str(e)}'}), 500

@app.route('/api/bmi-records', methods=['GET', 'POST'])
def bmi_records():
    decoded_token, error = verify_token(request)
    if error: return error
    uid = decoded_token['uid']

    records_ref = db.collection('users').document(uid).collection('bmiRecords')
    if request.method == 'GET':
        docs = records_ref.order_by("date", direction=Query.DESCENDING).order_by("time", direction=Query.DESCENDING).stream()
        records = [{'id': doc.id, **doc.to_dict()} for doc in docs]
        return jsonify(records), 200
    if request.method == 'POST':
        data = request.json
        try:
            height_m = float(data['height']) / 100
            weight_kg = float(data['weight'])
            bmi_value = round(weight_kg / (height_m ** 2), 2)
            data['bmi'] = bmi_value 
        except (ValueError, KeyError, ZeroDivisionError):
            data['bmi'] = None 
        data['timestamp'] = firestore.SERVER_TIMESTAMP 
        records_ref.add(data)
        return jsonify({'message': 'BMI 紀錄已儲存'}), 201
@app.route('/api/bmi-records/<record_id>', methods=['DELETE'])
def delete_bmi_record(record_id):
    decoded_token, error = verify_token(request)
    if error: return error
    uid = decoded_token['uid']

    try:
        db.collection('users').document(uid).collection('bmiRecords').document(record_id).delete()
        return jsonify({'message': '紀錄已刪除'}), 200
    except Exception as e: return jsonify({'error': f'刪除失敗: {str(e)}'}), 500

@app.route('/api/achievement-goals', methods=['GET', 'POST'])
def achievement_goals():
    decoded_token, error = verify_token(request)
    if error: return error
    uid = decoded_token['uid']

    user_ref = db.collection('users').document(uid)
    if request.method == 'GET':
        user_doc = user_ref.get()
        goals = {'goalWater': 2000, 'goalExercise': 30}
        if user_doc.exists:
            user_data = user_doc.to_dict()
            goals['goalWater'] = user_data.get('goalWater', 2000)
            goals['goalExercise'] = user_data.get('goalExercise', 30)
        return jsonify(goals), 200
    if request.method == 'POST':
        data = request.json
        user_ref.set({'goalWater': data.get('goalWater'), 'goalExercise': data.get('goalExercise')}, merge=True)
        return jsonify({'message': '目標已儲存'}), 200

@app.route('/api/achievement-records/<date_str>', methods=['GET', 'POST', 'DELETE'])
def achievement_record_by_date(date_str):
    decoded_token, error = verify_token(request)
    if error: return error
    uid = decoded_token['uid']

    record_ref = db.collection('users').document(uid).collection('achievementRecords').document(date_str)
    if request.method == 'GET':
        doc = record_ref.get()
        if doc.exists: return jsonify(doc.to_dict())
        default_record = {'date': date_str, 'waterMl': 0, 'exerciseMin': 0}
        record_ref.set(default_record) 
        return jsonify(default_record)
    if request.method == 'POST':
        update_data, current_doc = request.json, record_ref.get()
        if not current_doc.exists: 
             record_ref.set({'date': date_str, 'waterMl': 0, 'exerciseMin': 0})
             current_doc = record_ref.get() 

        current_data = current_doc.to_dict()
        new_water = max(0, current_data.get('waterMl', 0) + update_data.get('addWater', 0))
        new_exercise = max(0, current_data.get('exerciseMin', 0) + update_data.get('addExercise', 0))
        record_ref.set({'waterMl': new_water, 'exerciseMin': new_exercise, 'updatedAt': firestore.SERVER_TIMESTAMP}, merge=True)
        return jsonify({'waterMl': new_water, 'exerciseMin': new_exercise}) 
    if request.method == 'DELETE':
        try:
             record_ref.delete()
             return jsonify({'message': f'紀錄 {date_str} 已刪除'}), 200
        except Exception as e:
             return jsonify({'error': f'刪除失敗: {str(e)}'}), 500

@app.route('/api/achievement-history', methods=['GET'])
def get_achievement_history():
    decoded_token, error = verify_token(request)
    if error: return error
    uid = decoded_token['uid']

    limit = int(request.args.get('limit', 7)) 
    docs = db.collection('users').document(uid).collection('achievementRecords').order_by("date", direction=Query.DESCENDING).limit(limit).stream()
    return jsonify([{'id': doc.id, **doc.to_dict()} for doc in docs]), 200


@app.route('/api/badges', methods=['GET', 'POST'])
def handle_badges():
    decoded_token, error = verify_token(request)
    if error: return error
    uid = decoded_token['uid']

    badges_ref = db.collection('users').document(uid).collection('badges')

    if request.method == 'GET':
        return jsonify({doc.id: doc.to_dict() for doc in badges_ref.stream()})

    if request.method == 'POST':
        user_doc = db.collection('users').document(uid).get()
        if not user_doc.exists:
            return jsonify({'error': 'User not found'}), 404

        user_data = user_doc.to_dict()
        gw = user_data.get('goalWater', 2000) 
        ge = user_data.get('goalExercise', 30) 

        date_str = request.json.get('date', datetime.now().strftime('%Y-%m-%d'))
        today_rec_doc = db.collection('users').document(uid).collection('achievementRecords').document(date_str).get()

        water_ok, ex_ok = False, False
        if today_rec_doc.exists:
            rec = today_rec_doc.to_dict()
            water_ok = rec.get('waterMl', 0) >= gw
            ex_ok = rec.get('exerciseMin', 0) >= ge

        badges_ref.document('water_2l_day').set({'unlocked': water_ok, 'at': date_str if water_ok else None}, merge=True)
        badges_ref.document('exercise_30m_day').set({'unlocked': ex_ok, 'at': date_str if ex_ok else None}, merge=True)
        badges_ref.document('double_goal_day').set({'unlocked': (water_ok and ex_ok), 'at': date_str if (water_ok and ex_ok) else None}, merge=True)

        streak_ok = True
        streak_date = None
        for i in range(3):
            check_date = (datetime.strptime(date_str, '%Y-%m-%d') - timedelta(days=i)).strftime('%Y-%m-%d')
            r_doc = db.collection('users').document(uid).collection('achievementRecords').document(check_date).get()
            if not r_doc.exists or r_doc.to_dict().get('waterMl', 0) < gw or r_doc.to_dict().get('exerciseMin', 0) < ge:
                streak_ok = False
                break
            if i == 0: 
                streak_date = check_date

        badges_ref.document('streak_3').set({'unlocked': streak_ok, 'at': streak_date if streak_ok else None}, merge=True)

        return jsonify({'message': '徽章評估完成'}), 200


# --- ★ (新) AI 飲食評價 API ---
@app.route('/api/evaluate-diet', methods=['POST'])
def api_evaluate_diet():
    decoded_token, error = verify_token(request)
    if error: return error
    uid = decoded_token['uid']
    
    # 1. 獲取前端傳來的飲食數據
    diet_data = request.json
    if not diet_data or 'total_calories' not in diet_data:
        return jsonify({'error': '缺少飲食數據'}), 400

    # 2. 獲取使用者的個人資料
    try:
        user_doc = db.collection('users').document(uid).get()
        if not user_doc.exists:
            return jsonify({'error': '找不到使用者資料'}), 404
        user_profile = user_doc.to_dict()

        # 3. 將兩份資料交給 AI 處理
        # --- ★★★ 關鍵修改 ★★★ ---
        # 呼叫新的 Gemini 函式，而不是舊的 Llama 函式
        evaluation_text = generate_gemini_evaluation(user_profile, diet_data)
        # ----------------------------
        
        return jsonify({'evaluation': evaluation_text}), 200

    except Exception as e:
        print(f"評價 API 發生錯誤: {e}")
        return jsonify({'error': str(e)}), 500
    

# --- ★ (新) 營養歷史紀錄 API (保持不變) ---
@app.route('/api/nutrition-records', methods=['GET', 'POST'])
def api_nutrition_records():
    decoded_token, error = verify_token(request)
    if error: return error
    uid = decoded_token['uid']
    
    records_ref = db.collection('users').document(uid).collection('nutritionRecords')

    if request.method == 'GET':
        try:
            docs = records_ref.order_by("timestamp", direction=Query.DESCENDING).stream()
            records = [{'id': doc.id, **doc.to_dict()} for doc in docs]
            return jsonify(records), 200
        except Exception as e:
            return jsonify({'error': f'讀取紀錄失敗: {str(e)}'}), 500

    if request.method == 'POST':
        try:
            data = request.json
            if not data.get('foods_list') or 'total_calories' not in data:
                 return jsonify({'error': '缺少食物列表或總熱量'}), 400
            
            data['timestamp'] = firestore.SERVER_TIMESTAMP 
            doc_ref = records_ref.add(data)
            return jsonify({'message': '營養紀錄已儲存', 'doc_id': doc_ref[1].id}), 201
        except Exception as e:
            return jsonify({'error': f'儲存失敗: {str(e)}'}), 500

# --- ★ (新) 營養歷史紀錄刪除 API (保持不變) ---
@app.route('/api/nutrition-records/<record_id>', methods=['DELETE'])
def delete_nutrition_record(record_id):
    decoded_token, error = verify_token(request)
    if error: return error
    uid = decoded_token['uid']

    try:
        db.collection('users').document(uid).collection('nutritionRecords').document(record_id).delete()
        return jsonify({'message': '紀錄已刪除'}), 200
    except Exception as e: 
        return jsonify({'error': f'刪除失敗: {str(e)}'}), 500


# ------------------ 啟動伺服器 ------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)