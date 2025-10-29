# app.py (方法二：同時載入兩個模型)
# 包含使用者後台、AI 問答、BMI、成就、YOLOv8辨識 (雙模型)、FatSecret營養查詢

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
# *** 請將您第一個模型的字典內容填入這裡 ***
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

# *** 這是您第二次提供的大型字典，用於模型 B ***
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
    # (省略...)
    id_token = request.headers.get('Authorization', '').split('Bearer ')[-1]
    if not id_token:
        return None, (jsonify({'error': '缺少驗證資訊'}), 401)
    try:
        decoded_token = auth.verify_id_token(id_token)
        return decoded_token, None
    except Exception as e:
        return None, (jsonify({'error': f'Token 無效或過期: {e}'}), 401)


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
            food_name_cn = translate_text(food_item.get("food_name"))
            desc_cn = translate_text(food_item.get("food_description"))
            index_data = parse_index(food_item.get("food_description", ""))
            return {"food_name": food_name_cn, "food_description": desc_cn, "index": index_data}
    except requests.exceptions.RequestException as e: print(f"查詢 API 時發生錯誤 ({food_name}): {e}")
    return None
# ---------------------------------


# ===============================================================
# 路由 (修改 /predict 以處理雙模型)
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

# --- YOLOv8 辨識與營養查詢路由 (*** 已修改為處理雙模型 ***) ---
@app.route("/predict", methods=["POST"])
def predict():
    if 'image' not in request.files and 'file' not in request.files:
        return render_template("nutrition.html", error="未上傳圖片")

    file = request.files.get('image') or request.files.get('file')
    if not file or file.filename == '':
        return render_template("nutrition.html", error="未選擇圖片")

    try:
        # --- 影像處理 ---
        image_bytes = file.read()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # --- 儲存原始圖片 ---
        filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
        img_path = os.path.join(UPLOAD_FOLDER, filename)
        pil_image.save(img_path) 
        print(f"✅ 上傳圖片儲存於: {img_path}")
        uploaded_image_web = img_path.replace("\\", "/")

        # --- 初始化 OpenCV 圖像用於繪圖 ---
        # *** 從 PIL 轉換一次即可，後續在其上疊加繪圖 ***
        cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        # --- 初始化結果列表 ---
        all_detected_foods = [] # 合併後的偵測清單
        all_food_infos = []     # 合併後的營養資訊
        seen_foods_eng_names = set() # 跨模型追蹤已查詢的英文名

        # --- 執行模型 A 預測 ---
        print("執行模型 A 預測...")
        results_A = model_A(pil_image)
        if results_A and results_A[0].boxes:
            print(f"模型 A 找到 {len(results_A[0].boxes)} 個潛在物件")
            for box in results_A[0].boxes:
                conf = round(float(box.conf[0]), 2)
                if conf < 0.5: continue

                cls = int(box.cls[0])
                eng_name_A = model_A.names[cls].lower().replace("_", " ").replace("-", " ")
                # *** 使用字典 A 進行翻譯 ***
                chi_name_A = item_translation_A.get(eng_name_A, model_A.names[cls].capitalize())

                print(f"  模型 A: {eng_name_A} -> {chi_name_A} (信心度: {conf})")

                # 加入偵測清單
                all_detected_foods.append({'name': chi_name_A, 'confidence': f"{conf:.2f}", 'source': 'A'}) # 標示來源

                # 繪製模型 A 的框 (藍色 BGR: 255, 0, 0)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label_A = f"{chi_name_A} {conf} (A)"
                cv2.rectangle(cv_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(cv_image, label_A, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                # 查詢營養資訊 (如果未查詢過)
                if eng_name_A not in seen_foods_eng_names:
                    seen_foods_eng_names.add(eng_name_A)
                    index_data_A = search_food_index(eng_name_A)
                    if index_data_A:
                         all_food_infos.append({
                            'food_name': chi_name_A, 'confidence': f"{conf:.2f}",
                            'food_description': index_data_A['food_description'],
                            'index': index_data_A['index'], 'source': 'A' # 標示來源
                        })
                    else:
                        all_food_infos.append({
                            'food_name': chi_name_A, 'confidence': f"{conf:.2f}",
                            'food_description': "查無此食物的詳細營養資訊。",
                            'index': None, 'source': 'A' # 標示來源
                        })

        # --- 執行模型 B 預測 ---
        print("執行模型 B 預測...")
        results_B = model_B(pil_image)
        if results_B and results_B[0].boxes:
            print(f"模型 B 找到 {len(results_B[0].boxes)} 個潛在物件")
            # print(f"  [Debug B] results_B[0].boxes 的內容: {results_B[0].boxes}")
            for box in results_B[0].boxes:
                conf = round(float(box.conf[0]), 2)
                if conf < 0.2: continue

                cls = int(box.cls[0])
                eng_name_B = model_B.names[cls].lower().replace("_", " ").replace("-", " ")
                # *** 使用字典 B 進行翻譯 ***
                chi_name_B = item_translation_B.get(eng_name_B, model_B.names[cls].capitalize())

                print(f"  模型 B: {eng_name_B} -> {chi_name_B} (信心度: {conf})")

                # 加入偵測清單
                all_detected_foods.append({'name': chi_name_B, 'confidence': f"{conf:.2f}", 'source': 'B'}) # 標示來源

                # 繪製模型 B 的框 (綠色 BGR: 0, 255, 0) - *** 在已繪製 A 的圖上繼續畫 ***
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label_B = f"{chi_name_B} {conf} (B)"
                # 稍微調整 Y 座標以避免與模型 A 的標籤重疊
                text_y = y1 - 30 if y1 > 30 else y1 + 15
                cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(cv_image, label_B, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # 查詢營養資訊 (如果未查詢過)
                if eng_name_B not in seen_foods_eng_names:
                    seen_foods_eng_names.add(eng_name_B)
                    index_data_B = search_food_index(eng_name_B)
                    if index_data_B:
                         all_food_infos.append({
                            'food_name': chi_name_B, 'confidence': f"{conf:.2f}",
                            'food_description': index_data_B['food_description'],
                            'index': index_data_B['index'], 'source': 'B' # 標示來源
                        })
                    else:
                        all_food_infos.append({
                            'food_name': chi_name_B, 'confidence': f"{conf:.2f}",
                            'food_description': "查無此食物的詳細營養資訊。",
                            'index': None, 'source': 'B' # 標示來源
                        })

        # --- 如果兩個模型都沒偵測到 ---
        if not all_detected_foods:
             all_detected_foods = [{'name': '未偵測到任何食物', 'confidence': 'N/A', 'source': '-'}]

        # --- 儲存最終繪製結果圖片 ---
        result_filename = f"result_combined_{uuid.uuid4()}.jpg"
        result_img_path = os.path.join(RESULT_FOLDER, result_filename)
        # *** 使用 cv_image (已包含兩個模型的框) 儲存 ***
        cv2.imwrite(result_img_path, cv_image)
        print(f"✅ 合併偵測結果儲存於: {result_img_path}")
        result_image_web = result_img_path.replace("\\", "/")

        # --- 回傳所有合併後的資料到前端 ---
        return render_template("nutrition.html",
                               uploaded_image=uploaded_image_web,
                               result_image=result_image_web,
                               detected_foods=all_detected_foods, # 合併後的列表
                               food_infos=all_food_infos)        # 合併後的列表

    except Exception as e:
        print(f"辨識過程中發生錯誤: {e}")
        return render_template("nutrition.html", error=f"處理過程中發生錯誤: {e}")
# ---------------------------------


# --- API 路由 (使用者後台) ---
# (以下所有 /api/... 路由保持不變，省略以節省空間)
# ...
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
        update_data.pop('email', None) # 不允許透過此 API 更新 Email
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
        # 可以在這裡加入 BMI 計算邏輯
        try:
            height_m = float(data['height']) / 100
            weight_kg = float(data['weight'])
            bmi_value = round(weight_kg / (height_m ** 2), 2)
            data['bmi'] = bmi_value # 將計算出的 BMI 加入紀錄
        except (ValueError, KeyError, ZeroDivisionError):
            data['bmi'] = None # 如果無法計算，設為 None
        data['timestamp'] = firestore.SERVER_TIMESTAMP # 加入時間戳
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
    # (省略...)
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
    # (省略...)
    decoded_token, error = verify_token(request)
    if error: return error
    uid = decoded_token['uid']

    record_ref = db.collection('users').document(uid).collection('achievementRecords').document(date_str)
    if request.method == 'GET':
        doc = record_ref.get()
        if doc.exists: return jsonify(doc.to_dict())
        default_record = {'date': date_str, 'waterMl': 0, 'exerciseMin': 0}
        record_ref.set(default_record) # 如果當天記錄不存在，創建一個
        return jsonify(default_record)
    if request.method == 'POST':
        update_data, current_doc = request.json, record_ref.get()
        if not current_doc.exists: # 如果記錄不存在先創建
             record_ref.set({'date': date_str, 'waterMl': 0, 'exerciseMin': 0})
             current_doc = record_ref.get() # 重新獲取

        current_data = current_doc.to_dict()
        new_water = max(0, current_data.get('waterMl', 0) + update_data.get('addWater', 0))
        new_exercise = max(0, current_data.get('exerciseMin', 0) + update_data.get('addExercise', 0))
        record_ref.set({'waterMl': new_water, 'exerciseMin': new_exercise, 'updatedAt': firestore.SERVER_TIMESTAMP}, merge=True)
        return jsonify({'waterMl': new_water, 'exerciseMin': new_exercise}) # 回傳更新後的值
    if request.method == 'DELETE':
        try:
             record_ref.delete()
             return jsonify({'message': f'紀錄 {date_str} 已刪除'}), 200
        except Exception as e:
             return jsonify({'error': f'刪除失敗: {str(e)}'}), 500

@app.route('/api/achievement-history', methods=['GET'])
def get_achievement_history():
    # (省略...)
    decoded_token, error = verify_token(request)
    if error: return error
    uid = decoded_token['uid']

    limit = int(request.args.get('limit', 7)) # 從查詢參數獲取 limit
    docs = db.collection('users').document(uid).collection('achievementRecords').order_by("date", direction=Query.DESCENDING).limit(limit).stream()
    # 將 Firestore DocumentSnapshot 轉為字典列表
    return jsonify([{'id': doc.id, **doc.to_dict()} for doc in docs]), 200


@app.route('/api/badges', methods=['GET', 'POST'])
def handle_badges():
    # (省略...)
    decoded_token, error = verify_token(request)
    if error: return error
    uid = decoded_token['uid']

    badges_ref = db.collection('users').document(uid).collection('badges')

    if request.method == 'GET':
        # 回傳所有徽章及其狀態
        return jsonify({doc.id: doc.to_dict() for doc in badges_ref.stream()})

    if request.method == 'POST':
        # --- 重新評估徽章解鎖狀態 ---
        user_doc = db.collection('users').document(uid).get()
        if not user_doc.exists:
            return jsonify({'error': 'User not found'}), 404

        user_data = user_doc.to_dict()
        gw = user_data.get('goalWater', 2000) # 水目標
        ge = user_data.get('goalExercise', 30) # 運動目標

        # 檢查 "今天" (或指定日期) 是否達標
        date_str = request.json.get('date', datetime.now().strftime('%Y-%m-%d'))
        today_rec_doc = db.collection('users').document(uid).collection('achievementRecords').document(date_str).get()

        water_ok, ex_ok = False, False
        if today_rec_doc.exists:
            rec = today_rec_doc.to_dict()
            water_ok = rec.get('waterMl', 0) >= gw
            ex_ok = rec.get('exerciseMin', 0) >= ge

        # 更新單日徽章
        badges_ref.document('water_2l_day').set({'unlocked': water_ok, 'at': date_str if water_ok else None}, merge=True)
        badges_ref.document('exercise_30m_day').set({'unlocked': ex_ok, 'at': date_str if ex_ok else None}, merge=True)
        badges_ref.document('double_goal_day').set({'unlocked': (water_ok and ex_ok), 'at': date_str if (water_ok and ex_ok) else None}, merge=True)

        # 檢查連續 3 天達標 (包含今天)
        streak_ok = True
        streak_date = None
        for i in range(3):
            check_date = (datetime.strptime(date_str, '%Y-%m-%d') - timedelta(days=i)).strftime('%Y-%m-%d')
            r_doc = db.collection('users').document(uid).collection('achievementRecords').document(check_date).get()
            if not r_doc.exists or r_doc.to_dict().get('waterMl', 0) < gw or r_doc.to_dict().get('exerciseMin', 0) < ge:
                streak_ok = False
                break
            if i == 0: # 如果第一天就檢查通過，記錄當天日期
                streak_date = check_date

        badges_ref.document('streak_3').set({'unlocked': streak_ok, 'at': streak_date if streak_ok else None}, merge=True)

        return jsonify({'message': '徽章評估完成'}), 200
# ------------------ 啟動伺服器 ------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)