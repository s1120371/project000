# app.py (支援所有前端 API 的最終版本 - 已修正 Token 驗證與 Email 回傳問題)
# *** 並已整合 YOLOv8 食物辨識與 FatSecret 營養查詢功能 ***

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

# --- 辨識功能所需套件 (新加入) ---
from ultralytics import YOLO
from PIL import Image
import uuid
import re
# ---------------------------------

# ------------------ 初始化設定 ------------------
app = Flask(__name__)

# --- 辨識功能資料夾設定 (新加入) ---
UPLOAD_FOLDER = os.path.join("static", "uploads")
RESULT_FOLDER = os.path.join("static", "results")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
# ---------------------------------

# --- Firebase 初始化 (來自您的檔案) ---
if not os.path.exists('serviceAccountKey.json'):
    print("錯誤：找不到 'serviceAccountKey.json' 檔案！")
    exit()
cred = credentials.Certificate('serviceAccountKey.json')
firebase_admin.initialize_app(cred)
db = firestore.client()

# --- GGUF AI 模型載入 (來自您的檔案) ---
print("正在載入 GGUF AI 模型...")
model_filename = "Meta-Llama-3-8B-Instruct.Q4_K_M.gguf"
if not os.path.exists(model_filename):
    print(f"錯誤：找不到模型檔案 '{model_filename}'！")
    exit()
llm = Llama(model_path=model_filename, n_ctx=4096, n_gpu_layers=0, verbose=True)
print(f"GGUF AI 模型 '{model_filename}' 載入完成！")

# --- YOLOv8 辨識模型載入 (新加入) ---
print("正在載入 YOLOv8 辨識模型...")
model_path = os.path.join(os.getcwd(), "best.pt")
if not os.path.exists(model_path):
    print(f"錯誤：找不到模型檔案 'best.pt'！")
    exit()
model = YOLO(model_path)
print("YOLOv8 辨識模型 'best.pt' 載入完成！")
# ---------------------------------

# --- API 金鑰設定 (合併) ---
# Firebase Web API Key (來自您的檔案)
FIREBASE_WEB_API_KEY = "AIzaSyCCNPhST7sScxFdSZJ6-NbxKgqrSYOzes4"

# FatSecret API 金鑰 (新加入)
CONSUMER_KEY = "ba46d91448844c4ba3aa81ff09e605df"
CONSUMER_SECRET = "60302dd6c9c240d1a1118a75677e3967"
API_BASE = "https://platform.fatsecret.com/rest/server.api"
# ---------------------------------

# --- 食物名稱中英文對照 (新加入) ---
# (這是來自您 YOLOv8 app.py 的字典)
item_translation = {
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
# ---------------------------------


# ===============================================================
# 輔助函式 (合併)
# ===============================================================

# --- 輔助函式：驗證 Token (來自您的檔案) ---
def verify_token(request):
    """從請求標頭中驗證 idToken 並返回整個解碼後的 token 物件"""
    id_token = request.headers.get('Authorization', '').split('Bearer ')[-1]
    if not id_token:
        return None, (jsonify({'error': '缺少驗證資訊'}), 401)
    try:
        decoded_token = auth.verify_id_token(id_token)
        return decoded_token, None # <-- 改成回傳整個 decoded_token
    except Exception as e:
        return None, (jsonify({'error': f'Token 無效或過期: {e}'}), 401)

# --- 核心AI功能函式 (來自您的檔案) ---
def generate_llama_advice(user_query, user_profile, history_messages=None):
    # (此處省略您檔案中已有的 AI 相關程式碼，保持不變)
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
    # 這裡的 profile_text 仍然是從資料庫讀取的原始資料
    profile_text = f"""
- 健康目標: {goal_map.get(user_profile.get('goal'), '未設定')}
- 飲食習慣: {diet_map.get(user_profile.get('diet'), '未設定')}
- 已知過敏原: {', '.join(user_profile.get('allergens', [])) or '無'}
"""
    # 組合 messages 列表
    messages = [{"role": "system", "content": system_prompt + "\n這是使用者的基本資料（請記住，對話紀錄優先級更高）：\n" + profile_text}]
    
    # 如果有歷史訊息，就把它們加進來
    if history_messages:
        messages.extend(history_messages)
    
    # 最後加上使用者本次的問題
    messages.append({"role": "user", "content": user_query})

    response = llm.create_chat_completion(
        messages=messages, max_tokens=512, temperature=0.7
    )
    return response['choices'][0]['message']['content']
    
# --- 營養查詢輔助函式 (新加入) ---
def translate_text(text, target='zh-TW'):
    """翻譯文字"""
    if not text:
        return ""
    try:
        return GoogleTranslator(source='auto', target=target).translate(text)
    except Exception:
        return text

def parse_nutrition(description):
    """從 FatSecret 的描述文字中解析營養成分"""
    nutrition = {}
    cal_match = re.search(r"Calories:\s*([\d.]+)kcal", description)
    fat_match = re.search(r"Fat:\s*([\d.]+)g", description)
    carb_match = re.search(r"Carbs:\s*([\d.]+)g", description)
    protein_match = re.search(r"Protein:\s*([\d.]+)g", description)

    if cal_match: nutrition["熱量 (kcal)"] = f"{float(cal_match.group(1)):.2f}"
    if fat_match: nutrition["脂肪 (g)"] = f"{float(fat_match.group(1)):.2f}"
    if carb_match: nutrition["碳水化合物 (g)"] = f"{float(carb_match.group(1)):.2f}"
    if protein_match: nutrition["蛋白質 (g)"] = f"{float(protein_match.group(1)):.2f}"
    return nutrition

def search_food_nutrition(food_name):
    """查詢單一食物的營養資訊"""
    auth = OAuth1(CONSUMER_KEY, CONSUMER_SECRET)
    params = {
        "method": "foods.search",
        "search_expression": food_name, # 使用英文名稱查詢
        "format": "json",
        "max_results": 1
    }
    try:
        res = requests.get(API_BASE, params=params, auth=auth)
        res.raise_for_status() 
        data = res.json()

        if "foods" in data and "food" in data["foods"] and data["foods"]["food"]:
            food_item = data["foods"]["food"]
            if isinstance(food_item, list):
                food_item = food_item[0]

            food_name_cn = translate_text(food_item.get("food_name"))
            desc_cn = translate_text(food_item.get("food_description"))
            nutrition = parse_nutrition(food_item.get("food_description", ""))
            
            return {
                "food_name": food_name_cn,
                "food_description": desc_cn,
                "nutrition": nutrition
            }
    except requests.exceptions.RequestException as e:
        print(f"查詢 API 時發生錯誤 ({food_name}): {e}")
    
    return None
# ---------------------------------


# ===============================================================
# 路由 (合併)
# ===============================================================

# --- 靜態網頁路由 (來自您的檔案) ---
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/home')
def home():
    return render_template("home.html")

@app.route('/login')
def login():
    return render_template("login.html")

@app.route('/nutrition', methods=['GET']) # <-- 只保留 GET
def nutrition():
    # 這個路由只負責 "顯示" 上傳頁面
    return render_template("nutrition.html")

@app.route('/edit-profile')
def edit_profile():
    return render_template("edit-profile.html")

@app.route('/bmi')
def bmi():
    return render_template("bmi.html")

@app.route('/achievements')
def achievements():
    return render_template("achievements.html")

# --- YOLOv8 辨識路由 (新加入) ---
# 您的 nutrition.html 會將 POST 請求發送到這個 /predict 路由
@app.route("/predict", methods=["POST"])
def predict():
    if 'image' not in request.files:
        return render_template("nutrition.html", error="未上傳圖片")

    file = request.files['image']
    if file.filename == '':
        return render_template("nutrition.html", error="未選擇圖片")

    # 圖片儲存
    filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
    img_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(img_path)
    print(f"✅ 上傳圖片儲存於: {img_path}")

    # YOLO 偵測
    results = model(img_path)
    result = results[0]

    # 繪製結果圖像
    result_array = result.plot()
    result_image = Image.fromarray(result_array)

    # 儲存結果圖片
    result_filename = f"result_{uuid.uuid4()}.jpg"
    result_img_path = os.path.join(RESULT_FOLDER, result_filename)
    result_image.save(result_img_path)
    print(f"✅ 偵測結果儲存於: {result_img_path}")

    # 網頁路徑轉換
    uploaded_image_web = img_path.replace("\\", "/")
    result_image_web = result_img_path.replace("\\", "/")

    # 處理偵測結果與營養查詢
    detected_foods = []
    food_infos = []    
    seen_foods = set() 

    if result.boxes:
        for cls, conf in zip(result.boxes.cls, result.boxes.conf):
            raw_name = model.names[int(cls)].strip()
            eng_name = raw_name.lower().replace("_", " ").replace("-", " ")
            confidence = f"{float(conf):.2f}"

            zh_name = item_translation.get(eng_name)
            
            if zh_name is None:
                 for key in item_translation.keys():
                    if key.replace("_", " ").replace("-", " ").lower() == eng_name:
                        zh_name = item_translation[key]
                        break
            
            if zh_name is None:
                zh_name = raw_name.capitalize()

            print(f"偵測到: {eng_name} -> 對應中文: {zh_name}")

            # 1. 加入偵測清單
            detected_foods.append({'name': zh_name, 'confidence': confidence})

            # 2. 查詢營養資訊 (避免重複)
            if eng_name not in seen_foods:
                seen_foods.add(eng_name)
                nutrition_data = search_food_nutrition(eng_name)
                
                if nutrition_data:
                    food_infos.append({
                        'food_name': zh_name,
                        'confidence': confidence,
                        'food_description': nutrition_data['food_description'],
                        'nutrition': nutrition_data['nutrition']
                    })
                else:
                    food_infos.append({
                        'food_name': zh_name,
                        'confidence': confidence,
                        'food_description': "查無此食物的詳細營養資訊。",
                        'nutrition': None
                    })
    
    if not detected_foods:
        detected_foods = [{'name': '未偵測到任何食物', 'confidence': 'N/A'}]

    # 回傳所有資料到前端
    return render_template("nutrition.html",
                           uploaded_image=uploaded_image_web,
                           result_image=result_image_web,
                           detected_foods=detected_foods,
                           food_infos=food_infos)
# ---------------------------------


# --- API 路由 (來自您的檔案，保持不變) ---

# --- 註冊與登入 API ---
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

# --- 問答 API ---
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

# --- 聊天歷史紀錄 API ---
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

# --- 使用者資料 API ---
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
        update_data['updatedAt'] = firestore.SERVER_TIMESTAMP
        user_ref.set(update_data, merge=True)
        return jsonify({'message': '資料更新成功'}), 200

# --- 修改密碼 API ---
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

# --- BMI 紀錄 API ---
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

# --- 成就系統 API ---
@app.route('/api/achievement-goals', methods=['GET', 'POST'])
def achievement_goals():
    # (此處省略您檔案中已有的成就 API 程式碼，保持不變)
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
    # (此處省略您檔案中已有的成就 API 程式碼，保持不變)
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
        if current_doc.exists:
            current_data = current_doc.to_dict()
            new_water = max(0, current_data.get('waterMl', 0) + update_data.get('addWater', 0))
            new_exercise = max(0, current_data.get('exerciseMin', 0) + update_data.get('addExercise', 0))
            record_ref.set({'waterMl': new_water, 'exerciseMin': new_exercise, 'updatedAt': firestore.SERVER_TIMESTAMP}, merge=True)
            return jsonify({'waterMl': new_water, 'exerciseMin': new_exercise})
        return jsonify({'error':'Record not found'}), 404
    if request.method == 'DELETE':
        record_ref.delete()
        return jsonify({'message': f'紀錄 {date_str} 已刪除'}), 200


@app.route('/api/achievement-history', methods=['GET'])
def get_achievement_history():
    # (此處省略您檔案中已有的成就 API 程式碼，保持不變)
    decoded_token, error = verify_token(request)
    if error: return error
    uid = decoded_token['uid']
    
    limit = int(request.args.get('limit', 7))
    docs = db.collection('users').document(uid).collection('achievementRecords').order_by("date", direction=Query.DESCENDING).limit(limit).stream()
    return jsonify([{'id': doc.id, **doc.to_dict()} for doc in docs]), 200


@app.route('/api/badges', methods=['GET', 'POST'])
def handle_badges():
    # (此處省略您檔案中已有的成就 API 程式碼，保持不變)
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
            water_ok, ex_ok = rec.get('waterMl', 0) >= gw, rec.get('exerciseMin', 0) >= ge
        
        badges_ref.document('water_2l_day').set({'unlocked': water_ok, 'at': date_str}, merge=True)
        badges_ref.document('exercise_30m_day').set({'unlocked': ex_ok, 'at': date_str}, merge=True)
        badges_ref.document('double_goal_day').set({'unlocked': (water_ok and ex_ok), 'at': date_str}, merge=True)
        
        streak_ok = True
        for i in range(3):
            key = (datetime.strptime(date_str, '%Y-%m-%d') - timedelta(days=i)).strftime('%Y-%m-%d')
            r_doc = db.collection('users').document(uid).collection('achievementRecords').document(key).get()
            if not r_doc.exists or r_doc.to_dict().get('waterMl', 0) < gw or r_doc.to_dict().get('exerciseMin', 0) < ge:
                streak_ok = False
                break
        
        badges_ref.document('streak_3').set({'unlocked': streak_ok, 'at': date_str}, merge=True)
        
        return jsonify({'message': '徽章評估完成'}), 200

# ------------------ 啟動伺服器 ------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)