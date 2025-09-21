# 載入我們需要的函式庫
from flask import Flask, request, jsonify, render_template, send_from_directory
from sentence_transformers import SentenceTransformer, util
import json
import torch
import firebase_admin
from firebase_admin import credentials, auth, firestore

# ------------------ 初始化設定 ------------------

app = Flask(__name__)

# --- Firebase Admin SDK 初始化 ---
cred = credentials.Certificate('serviceAccountKey.json')
firebase_admin.initialize_app(cred)
db = firestore.client()

# --- AI 模型與知識庫載入 ---
print("正在載入AI模型...")
model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
print("AI模型載入完成！")
with open('knowledge_base.json', 'r', encoding='utf-8') as f:
    knowledge_base = json.load(f)
kb_questions = [item['question'] for item in knowledge_base]
print("正在將知識庫轉換為向量...")
kb_embeddings = model.encode(kb_questions, convert_to_tensor=True)
print("知識庫向量轉換完成！")

# ------------------ 核心AI功能函式 ------------------

def generate_personalized_advice(user_query, user_profile):
    query_embedding = model.encode(user_query, convert_to_tensor=True)
    cosine_scores = util.cos_sim(query_embedding, kb_embeddings)
    best_match_idx = torch.argmax(cosine_scores)
    best_score = cosine_scores[0][best_match_idx]

    base_answer = ""
    if best_score > 0.5:
        base_answer = knowledge_base[best_match_idx]['answer']
    else:
        base_answer = "抱歉，我對這個問題還不太了解。但我可以根據你的個人資料給一些通用建議。"

    personal_advice_list = []
    goal = user_profile.get('goal')
    diet = user_profile.get('diet')
    allergens = user_profile.get('allergens', [])

    if goal == 'weight-loss':
        if '吃' in user_query or '外食' in user_query:
            personal_advice_list.append("因為您的目標是減重，建議優先選擇低熱量、高纖維的食物，並多用蒸、煮代替油炸。")
        if '新陳代謝' in user_query:
            personal_advice_list.append("為了配合您的減重目標，增加肌肉量是提升基礎代謝率的好方法，可以考慮加入適度的重量訓練。")
    elif goal == 'muscle-gain':
        if '吃' in user_query or '運動' in user_query or '健身' in user_query:
            if diet == 'vegan':
                personal_advice_list.append("為了幫助您增肌，身為全素者的您可以多攝取豆腐、天貝、鷹嘴豆和各式豆類來補充優質植物性蛋白。")
            elif diet == 'lacto-ovo':
                personal_advice_list.append("為了幫助您增肌，身為蛋奶素的您可以多攝取雞蛋、牛奶、希臘優格等優質蛋白質。")
            else:
                personal_advice_list.append("為了幫助您增肌，請確保攝取足夠的優質蛋白質，尤其是在運動後。雞胸肉、魚肉和雞蛋都是很好的選擇。")
    elif goal == 'control-sugar':
        if '吃' in user_query or '高血壓' in user_query:
            personal_advice_list.append("考量到您控制血糖的目標，建議選擇低GI值的食物，如糙米、燕麥和綠色蔬菜，並遵循少油、少鹽、少糖的原則。")

    if diet == 'vegan' and any(x in base_answer for x in ['肉','奶','蛋','海鮮']):
        personal_advice_list.append("提醒您是全素者，可以將答案中的動物性成分替換為植物性來源，如豆漿、豆腐或植物肉。")
    elif diet == 'lacto-ovo' and any(x in base_answer for x in ['肉','海鮮']):
        personal_advice_list.append("提醒您是蛋奶素者，可以將答案中的肉類替換成雞蛋、乳製品或植物性蛋白。")

    for allergen in allergens:
        if allergen in base_answer:
            personal_advice_list.append(f"⚠️ 安全提醒：您的資料顯示對「{allergen}」過敏，請務必避免食用答案中提到的相關料理。")

    if not personal_advice_list:
        return base_answer + "\n\n希望這個資訊對您有幫助！"
    else:
        final_response = f"基本答案：{base_answer}\n\n"
        final_response += "--- 針對您的個人化建議 ---\n"
        final_response += "\n".join(f"• {item}" for item in personal_advice_list)
        return final_response

# ------------------ 網頁服務路由 ------------------

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login')
def login_page():
    return send_from_directory('static', 'login.html')

@app.route('/home')
def home_page():
    return send_from_directory('static', 'home.html')

@app.route('/edit-profile')
def edit_profile_page():
    return send_from_directory('static', 'edit-profile.html')

@app.route('/achievements')
def achievements_page():
    return send_from_directory('static', 'achievements.html')

@app.route('/bmi')
def bmi_page():
    return send_from_directory('static', 'bmi.html')

@app.route('/nutrition')
def nutrition_page():
    return send_from_directory('static', 'nutrition.html')

# ------------------ API: 取得使用者資料 ------------------

@app.route('/api/profile', methods=['POST'])
def get_profile():
    data = request.json
    id_token = data.get('idToken')
    if not id_token:
        return jsonify({'error': '缺少 ID Token'}), 400

    try:
        decoded = auth.verify_id_token(id_token)
        uid = decoded['uid']
        user_ref = db.collection('users').document(uid)
        doc = user_ref.get()
        if not doc.exists:
            return jsonify({'error': '使用者資料不存在'}), 404
        return jsonify({'uid': uid, 'profile': doc.to_dict()})
    except Exception as e:
        return jsonify({'error': f'驗證失敗: {e}'}), 401

# ------------------ API: 更新使用者資料 ------------------

@app.route('/api/saveProfile', methods=['POST'])
def save_profile():
    data = request.json
    id_token = data.get('idToken')
    profile_data = data.get('profile')
    
    if not id_token or not profile_data:
        return jsonify({'error': '缺少必要參數'}), 400

    try:
        decoded = auth.verify_id_token(id_token)
        uid = decoded['uid']
        user_ref = db.collection('users').document(uid)
        user_ref.set(profile_data, merge=True)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': f'更新失敗: {e}'}), 500

# ------------------ AI 問答 API ------------------

@app.route('/ask', methods=['POST'])
def ask():
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
        if not user_doc.exists:
            return jsonify({'answer': '錯誤：找不到您的使用者設定檔。'})
        user_profile = user_doc.to_dict()
        answer = generate_personalized_advice(user_question, user_profile)
        return jsonify({'answer': answer})
    except Exception as e:
        return jsonify({'answer': f'伺服器發生未預期的錯誤: {e}'}), 500

# ------------------ 啟動伺服器 ------------------

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
