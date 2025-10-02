# app.py (支援所有前端 API 的最終版本 - 已修正 Token 驗證與 Email 回傳問題)

from flask import Flask, request, jsonify, send_from_directory
import firebase_admin
from firebase_admin import credentials, auth, firestore
from firebase_admin.firestore import Query
import os
import requests
from llama_cpp import Llama
from datetime import datetime, timedelta
from flask import render_template
import torch, cv2, numpy as np, base64, re
from PIL import Image
from requests_oauthlib import OAuth1
from deep_translator import GoogleTranslator

# ------------------ 初始化設定 ------------------
#app = Flask(__name__, static_folder='templates', static_url_path='')
app = Flask(__name__)

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)


# 確保 serviceAccountKey.json 檔案存在
if not os.path.exists('serviceAccountKey.json'):
    print("錯誤：找不到 'serviceAccountKey.json' 檔案！")
    exit()
    
cred = credentials.Certificate('serviceAccountKey.json')
firebase_admin.initialize_app(cred)
db = firestore.client()

print("正在載入 GGUF AI 模型...")
model_filename = "Meta-Llama-3-8B-Instruct.Q4_K_M.gguf"
if not os.path.exists(model_filename):
    print(f"錯誤：找不到模型檔案 '{model_filename}'！")
    exit()
llm = Llama(model_path=model_filename, n_ctx=4096, n_gpu_layers=0, verbose=True)
print(f"GGUF AI 模型 '{model_filename}' 載入完成！")

# 這個金鑰來自您前端 firebaseConfig 物件，它是公開的，用於識別專案
# 請確保這個金鑰是您 Firebase 專案的 Web API Key
FIREBASE_WEB_API_KEY = "AIzaSyCCNPhST7sScxFdSZJ6-NbxKgqrSYOzes4"

# ------------------ 輔助函式：驗證 Token (已修改) ------------------
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

# ------------------ 核心AI功能函式 (維持不變) ------------------
def generate_llama_advice(user_query, user_profile):
    system_prompt = """你是一個專業又親切的「健康管家 AI」。

你必須嚴格遵守以下規則：
1.  **首要規則：你的唯一輸出語言是繁體中文。絕對禁止使用英文或其他任何語言作答。**
2.  你的任務是根據使用者的個人健康資料和他們提出的問題，提供準確、個人化且安全的健康飲食建議。
3.  你的回答應該要自然地**整合**使用者的個人資料，而不是分開條列。
4.  如果使用者的問題涉及到他的過敏原，請**務必**在回答中提出明確的安全警告。
5.  保持回覆簡潔、溫暖、易於理解。
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
    messages = [
        {"role": "system", "content": system_prompt + "\n這是正在跟你對話的使用者的個人資料：\n" + profile_text},
        {"role": "user", "content": user_query},
    ]
    response = llm.create_chat_completion(
        messages=messages, max_tokens=512, temperature=0.7
    )
    return response['choices'][0]['message']['content']

# ------------------ 靜態網頁路由 (維持不變) ------------------
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/home')
def home():
    return render_template("home.html")

@app.route('/login')
def login():
    return render_template("login.html")

@app.route('/nutrition', methods=['GET', 'POST'])
def nutrition():
    detected_foods, food_infos, img_base64 = [], [], None

    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template("nutrition.html", error="請上傳圖片")

        file = request.files['file']
        img = Image.open(file.stream).convert('RGB')
        img_array = np.array(img)

        # YOLOv5 偵測
        results = model(img_array)
        labels = results.names
        pred = results.pred[0]

        detected_foods_conf = []
        for *box, conf, cls in pred:
            conf_score = conf.item()
            if conf_score >= CONF_THRESHOLD:
                detected_foods_conf.append({
                    "name": labels[int(cls)],
                    "confidence": round(conf_score, 2)
                })

        detected_foods = detected_foods_conf

        # ---------------- FatSecret 查詢 (避免重複) ----------------
        seen = set()
        for f in detected_foods:
            food_name = f["name"]
            if food_name in seen:  # 已經查過就跳過
                continue
            seen.add(food_name)

            infos = search_food(food_name)
            if infos:
                info = infos[0]
                info["confidence"] = f["confidence"]
                food_infos.append(info)
            else:
                food_infos.append({
                    "food_name": food_name,
                    "food_description": "查無資料",
                    "confidence": f["confidence"],
                    "nutrition": {}
                })

        # ---------------- 繪製結果圖片 ----------------
        result_img = results.render()[0]
        result_img = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode(".jpg", result_img)
        img_base64 = base64.b64encode(buffer).decode("utf-8")

    return render_template("nutrition.html",
                           detected_foods=detected_foods,
                           food_infos=food_infos,
                           result_img=img_base64)



@app.route('/edit-profile')
def edit_profile():
    return render_template("edit-profile.html")

@app.route('/bmi')
def bmi():
    return render_template("bmi.html")

@app.route('/achievements')
def achievements():
    return render_template("achievements.html")

# ------------------ API 路由 (全新擴充與修正) ------------------

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

# --- 問答 API (已修正為新驗證方式) ---
@app.route('/ask', methods=['POST'])
def ask():
    decoded_token, error = verify_token(request)
    if error: return error
    uid = decoded_token['uid']
    
    user_data = request.json
    user_question = user_data.get('question')
    if not user_question: return jsonify({'answer': '你沒有問問題喔！'})
    
    try:
        user_doc = db.collection('users').document(uid).get()
        if not user_doc.exists: return jsonify({'answer': '錯誤：找不到您的使用者設定檔。'})
        
        answer = generate_llama_advice(user_question, user_doc.to_dict())
        return jsonify({'answer': answer})
    except Exception as e:
        return jsonify({'answer': f'伺服器發生未預期的錯誤: {str(e)}'}), 500

# --- 使用者資料 API (已修正為新驗證方式並確保回傳 email) ---
@app.route('/api/user-profile', methods=['GET', 'POST'])
def user_profile():
    decoded_token, error = verify_token(request)
    if error: return error
    uid = decoded_token['uid']
    user_ref = db.collection('users').document(uid)

    if request.method == 'GET':
        user_doc = user_ref.get()
        profile_data = user_doc.to_dict() if user_doc.exists else {}

        # 關鍵修正：將 token 中的 email 附加到回傳資料中，確保前端一定能取到
        profile_data['email'] = decoded_token.get('email', '') 
        
        return jsonify(profile_data)
        
    if request.method == 'POST':
        update_data = request.json
        update_data['updatedAt'] = firestore.SERVER_TIMESTAMP
        user_ref.set(update_data, merge=True)
        return jsonify({'message': '資料更新成功'}), 200


# --- 修改密碼 API (已修正為新驗證方式) ---
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

# --- BMI 紀錄 API (已修正為新驗證方式) ---
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

# --- 成就系統 API (已修正為新驗證方式) ---
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
API_BASE2 = "https://platform.fatsecret.com/rest/server.api"
CONSUMER_KEY = "ba46d91448844c4ba3aa81ff09e605df"
CONSUMER_SECRET = "60302dd6c9c240d1a1118a75677e3967"

CONF_THRESHOLD = 0.5  # YOLOv5 信心閾值

# 載入 YOLOv5 模型 (用 Ultralytics hub 或本地模型)
model = torch.hub.load("ultralytics/yolov5", "yolov5s")  # 可以換成 yolov5n, yolov5m, yolov5l

# ---------------- 功能函式 ----------------
def translate_text(text, target='zh-TW'):
    try:
        return GoogleTranslator(source='auto', target=target).translate(text)
    except:
        return text

def parse_nutrition(description):
    """從 FatSecret food_description 裡解析營養成分"""
    nutrition = {}
    cal_match = re.search(r"Calories:\s*([\d.]+)kcal", description)
    fat_match = re.search(r"Fat:\s*([\d.]+)g", description)
    carb_match = re.search(r"Carbs:\s*([\d.]+)g", description)
    protein_match = re.search(r"Protein:\s*([\d.]+)g", description)

    if cal_match: nutrition["熱量 (kcal)"] = cal_match.group(1)
    if fat_match: nutrition["脂肪 (g)"] = fat_match.group(1)
    if carb_match: nutrition["碳水化合物 (g)"] = carb_match.group(1)
    if protein_match: nutrition["蛋白質 (g)"] = protein_match.group(1)

    return nutrition

def search_food(food_name):
    """查詢 FatSecret 並翻譯結果"""
    auth = OAuth1(CONSUMER_KEY, CONSUMER_SECRET)
    params = {
        "method": "foods.search",
        "search_expression": food_name,
        "format": "json"
    }
    res = requests.get(API_BASE2, params=params, auth=auth)
    food_list = []
    if res.status_code == 200:
        data = res.json()
        if "foods" in data and "food" in data["foods"]:
            for food_item in data["foods"]["food"]:
                food_name_cn = translate_text(food_item["food_name"])
                desc_cn = translate_text(food_item["food_description"])
                nutrition = parse_nutrition(food_item["food_description"])
                food_list.append({
                    "food_name": food_name_cn,
                    "food_description": desc_cn,
                    "nutrition": nutrition
                })
    return food_list

# ---------------- Flask Route ----------------



# ------------------ 啟動伺服器 ------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)