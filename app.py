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

# --- AI 模型與知識庫載入 (維持不變) ---
print("正在載入AI模型...")
model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
print("AI模型載入完成！")
with open('knowledge_base.json', 'r', encoding='utf-8') as f:
    knowledge_base = json.load(f)
kb_questions = [item['question'] for item in knowledge_base]
print("正在將知識庫轉換為向量...")
kb_embeddings = model.encode(kb_questions, convert_to_tensor=True)
print("知識庫向量轉換完成！")


# ------------------ 核心AI功能函式 (已大幅升級) ------------------

def generate_personalized_advice(user_query, user_profile):
    """
    這個函式負責找出與使用者問題最相近的答案，並結合使用者資料生成個人化建議。
    (V2.0 - 增強版邏輯)
    """
    # 1. 取得基礎答案 (邏輯不變)
    query_embedding = model.encode(user_query, convert_to_tensor=True)
    cosine_scores = util.cos_sim(query_embedding, kb_embeddings)
    best_match_idx = torch.argmax(cosine_scores)
    best_score = cosine_scores[0][best_match_idx]

    base_answer = ""
    if best_score > 0.5:
        base_answer = knowledge_base[best_match_idx]['answer']
    else:
        base_answer = "抱歉，我對這個問題還不太了解。但我可以根據你的個人資料給一些通用建議。"

    # --- 關鍵：開始生成個人化建議 (全新邏輯) ---
    
    # 使用一個列表來收集所有個人化建議，更清晰好管理
    personal_advice_list = []
    goal = user_profile.get('goal')
    diet = user_profile.get('diet')
    allergens = user_profile.get('allergens', [])

    # === 判斷 1: 根據健康目標 ===
    if goal == 'weight-loss':
        if '吃' in user_query or '外食' in user_query:
            personal_advice_list.append("因為您的目標是減重，建議優先選擇低熱量、高纖維的食物，並多用蒸、煮代替油炸。")
        if '新陳代謝' in user_query:
            personal_advice_list.append("為了配合您的減重目標，增加肌肉量是提升基礎代謝率的好方法，可以考慮加入適度的重量訓練。")
            
    elif goal == 'muscle-gain':
        if '吃' in user_query or '運動' in user_query or '健身' in user_query:
            # ✨ 多重條件判斷：目標(增肌) + 飲食習慣(素食)
            if diet == 'vegan':
                personal_advice_list.append("為了幫助您增肌，身為全素者的您可以多攝取豆腐、天貝、鷹嘴豆和各式豆類來補充優質植物性蛋白。")
            elif diet == 'lacto-ovo':
                personal_advice_list.append("為了幫助您增肌，身為蛋奶素的您可以多攝取雞蛋、牛奶、希臘優格等優質蛋白質。")
            else:
                personal_advice_list.append("為了幫助您增肌，請確保攝取足夠的優質蛋白質，尤其是在運動後。雞胸肉、魚肉和雞蛋都是很好的選擇。")

    elif goal == 'control-sugar':
        if '吃' in user_query or '高血壓' in user_query:
            personal_advice_list.append("考量到您控制血糖的目標，建議選擇低GI值的食物，如糙米、燕麥和綠色蔬菜，並遵循少油、少鹽、少糖的原則。")

    # === 判斷 2: 根據飲食習慣 (提供替代方案) ===
    # 這個判斷可以在目標判斷之外獨立存在
    if diet == 'vegan' and ('肉' in base_answer or '奶' in base_answer or '蛋' in base_answer or '海鮮' in base_answer):
        personal_advice_list.append("提醒您是全素者，可以將答案中的動物性成分替換為植物性來源，如豆漿、豆腐或植物肉。")
    elif diet == 'lacto-ovo' and ('肉' in base_answer or '海鮮' in base_answer):
        personal_advice_list.append("提醒您是蛋奶素者，可以將答案中的肉類替換成雞蛋、乳製品或植物性蛋白。")

    # === 判斷 3: 根據過敏源 (提供警告) ===
    # 這個判斷的優先級最高，因為關乎安全
    for allergen in allergens:
        if allergen in base_answer:
            # 使用 f-string 讓訊息更具體
            personal_advice_list.append(f"⚠️ 安全提醒：您的資料顯示對「{allergen}」過敏，請務必避免食用答案中提到的相關料理。")

    # === 組合最終回覆 ===
    if not personal_advice_list:
        # 如果沒有觸發任何個人化建議，就回傳基本答案和一句通用結語
        return base_answer + "\n\n希望這個資訊對您有幫助！"
    else:
        # 如果有個人化建議，就把它們組合起來
        final_response = f"基本答案：{base_answer}\n\n"
        final_response += "--- 針對您的個人化建議 ---\n"
        # 用換行符號連接所有建議
        final_response += "\n".join(f"• {item}" for item in personal_advice_list)
        return final_response

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
    # --- 開始偵錯 ---
    print("\n[DEBUG] 收到一個新的 /ask 請求")
    
    user_data = request.json
    user_question = user_data.get('question')
    id_token = user_data.get('idToken')
    
    if not user_question:
        print("[DEBUG] 請求失敗：沒有問題內容")
        return jsonify({'answer': '你沒有問問題喔！'})
        
    if not id_token:
        print("[DEBUG] 請求失敗：缺少 ID Token")
        return jsonify({'answer': '錯誤：請求中缺少使用者驗證資訊，請先登入。'})

    try:
        print(f"[DEBUG] 步驟 1: 正在驗證 ID Token...")
        decoded_token = auth.verify_id_token(id_token)
        uid = decoded_token['uid']
        print(f"[DEBUG] 步驟 1: 驗證成功！使用者 UID: {uid}")

        print(f"[DEBUG] 步驟 2: 正在從 Firestore 獲取使用者資料...")
        user_ref = db.collection('users').document(uid)
        user_doc = user_ref.get()
        
        user_profile = {}
        if user_doc.exists:
            user_profile = user_doc.to_dict()
            print(f"[DEBUG] 步驟 2: 成功獲取使用者資料！目標為: {user_profile.get('goal')}")
        else:
            print("[DEBUG] 請求失敗：在 Firestore 中找不到該使用者")
            return jsonify({'answer': '錯誤：找不到您的使用者設定檔。'})

        print(f"[DEBUG] 步驟 3: 準備呼叫 AI 模型產生個人化建議...")
        answer = generate_personalized_advice(user_question, user_profile)
        print("[DEBUG] 步驟 3: AI 模型已成功回傳答案！")

        print("[DEBUG] 步驟 4: 準備將最終答案傳回前端")
        return jsonify({'answer': answer})

    except auth.InvalidIdTokenError as e:
        print(f"[ERROR] ID Token 無效: {e}")
        return jsonify({'answer': '錯誤：無效的驗證憑證，請重新登入。'}), 401
    except Exception as e:
        # 這個 Exception 會捕捉到所有其他類型的錯誤
        print(f"[CRITICAL ERROR] 發生未預期的嚴重錯誤: {e}")
        import traceback
        traceback.print_exc() # 這會印出更詳細的錯誤堆疊
        return jsonify({'answer': f'伺服器發生未預期的錯誤: {e}'}), 500

# ------------------ 啟動伺服器 ------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)