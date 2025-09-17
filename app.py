# 載入我們需要的函式庫
from flask import Flask, request, jsonify, render_template, send_from_directory
from sentence_transformers import SentenceTransformer, util
import json
import torch
import firebase_admin
from firebase_admin import credentials, auth, firestore

# ------------------ 初始化設定 ------------------

# 建立一個 Flask 應用
app = Flask(__name__)

# --- Firebase Admin SDK 初始化 ---
cred = credentials.Certificate('serviceAccountKey.json')
firebase_admin.initialize_app(cred)
db = firestore.client()

# ... (AI模型載入和知識庫向量化的程式碼維持不變) ...
print("正在載入AI模型...")
model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
print("AI模型載入完成！")
with open('knowledge_base.json', 'r', encoding='utf-8') as f:
    knowledge_base = json.load(f)
kb_questions = [item['question'] for item in knowledge_base]
print("正在將知識庫轉換為向量...")
kb_embeddings = model.encode(kb_questions, convert_to_tensor=True)
print("知識庫向量轉換完成！")
# ... (AI核心函式 generate_personalized_advice 維持不變) ...
def generate_personalized_advice(user_query, user_profile):
    # ...函式內容不變...
    query_embedding = model.encode(user_query, convert_to_tensor=True)
    cosine_scores = util.cos_sim(query_embedding, kb_embeddings)
    best_match_idx = torch.argmax(cosine_scores)
    best_score = cosine_scores[0][best_match_idx]
    base_answer = ""
    if best_score > 0.5:
        base_answer = knowledge_base[best_match_idx]['answer']
    else:
        base_answer = "抱歉，我對這個問題還不太了解。但我可以根據你的個人資料給一些通用建議。"
    personal_prompt = f"基本答案：{base_answer}\n\n"
    personal_prompt += "--- 以下是針對您的個人化建議 ---\n"
    if user_profile.get('goal') == 'weight-loss' and '吃' in user_query:
        personal_prompt += "因為您的目標是減重，建議選擇低熱量、高纖維的食物，並注意烹調方式，多用蒸、煮代替油炸。\n"
    elif user_profile.get('goal') == 'muscle-gain' and ('吃' in user_query or '運動' in user_query):
        personal_prompt += "為了幫助您增肌，請確保攝取足夠的優質蛋白質，尤其是在運動後。\n"
    allergens = user_profile.get('allergens', [])
    if '海鮮' in allergens and '海鮮' in base_answer:
         personal_prompt += "提醒您：您的資料顯示對海鮮過敏，請避免食用相關料理。\n"
    if '花生' in allergens and '花生' in base_answer:
         personal_prompt += "提醒您：您的資料顯示對花生過敏，請注意食物成分。\n"
    diet = user_profile.get('diet')
    if diet == 'vegan' and ('肉' in base_answer or '奶' in base_answer):
        personal_prompt += "由於您是全素者，建議將食譜中的動物性成分替換為植物性蛋白質來源，如豆腐、鷹嘴豆或扁豆。\n"
    elif diet == 'lacto-ovo' and '肉' in base_answer:
        personal_prompt += "身為蛋奶素的您，可以將肉類替換成雞蛋、乳製品或植物性蛋白。\n"
    if personal_prompt.endswith("---\n"):
        return base_answer + "\n\n希望這個資訊對您有幫助！"
    return personal_prompt

# ------------------ 網頁服務路由 (Routes) ------------------

# AI 問答頁 (根目錄)
@app.route('/')
def index():
    return render_template('index.html')

# 提供登入頁面
@app.route('/login')
def login_page():
    return send_from_directory('static', 'login.html')

# 提供主頁
@app.route('/home')
def home_page():
    return send_from_directory('static', 'home.html')

# 提供編輯個人資料頁面
@app.route('/edit-profile')
def edit_profile_page():
    return send_from_directory('static', 'edit-profile.html')

# 提供 BMI 計算頁面
@app.route('/bmi')
def bmi_page():
    return send_from_directory('static', 'bmi.html')

# 問答 API 路由
@app.route('/ask', methods=['POST'])
def ask():
    # ... (ask 函式內容維持不變) ...
    user_data = request.json
    user_question = user_data.get('question')
    id_token = user_data.get('idToken')
    if not user_question:
        return jsonify({'answer': '你沒有問問題喔！'})
    if not id_token:
        return jsonify({'answer': '錯誤：請求中缺少使用者驗證資訊，請先登入。'})
    try:
        decoded_token = auth.verify_id_token(id_token)
        uid = decoded_token['uid']
        user_ref = db.collection('users').document(uid)
        user_doc = user_ref.get()
        if user_doc.exists:
            user_profile = user_doc.to_dict()
        else:
            return jsonify({'answer': '錯誤：找不到您的使用者設定檔。'})
        answer = generate_personalized_advice(user_question, user_profile)
        return jsonify({'answer': answer})
    except auth.InvalidIdTokenError:
        return jsonify({'answer': '錯誤：無效的驗證憑證，請重新登入。'}), 401
    except Exception as e:
        print(f"發生錯誤: {e}")
        return jsonify({'answer': '伺服器發生未預期的錯誤。'}), 500

# ------------------ 啟動伺服器 ------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)