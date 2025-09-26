# app.py (支援所有前端 API 的最終版本 - 已修正 Token 驗證與 Email 回傳問題)

from flask import Flask, request, jsonify, send_from_directory
import firebase_admin
from firebase_admin import credentials, auth, firestore
from firebase_admin.firestore import Query
import os
import requests
from llama_cpp import Llama
from datetime import datetime, timedelta

# ------------------ 初始化設定 ------------------
app = Flask(__name__, static_folder='templates', static_url_path='')

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
def serve_index(): return send_from_directory(app.static_folder, 'index.html')
@app.route('/login')
def serve_login(): return send_from_directory(app.static_folder, 'login.html')
@app.route('/home')
def serve_home(): return send_from_directory(app.static_folder, 'home.html')
@app.route('/edit-profile')
def serve_edit_profile(): return send_from_directory(app.static_folder, 'edit-profile.html')
@app.route('/achievements')
def serve_achievements(): return send_from_directory(app.static_folder, 'achievements.html')
@app.route('/bmi')
def serve_bmi(): return send_from_directory(app.static_folder, 'bmi.html')
@app.route('/nutrition')
def serve_nutrition(): return send_from_directory(app.static_folder, 'nutrition.html')

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

# ------------------ 啟動伺服器 ------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)