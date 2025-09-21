# 載入我們需要的函式庫
from flask import Flask, request, jsonify, render_template, send_from_directory
# 舊的 SentenceTransformer 不再需要了
# from sentence_transformers import SentenceTransformer, util 
# import json # 知識庫也不再需要了
import torch
import firebase_admin
from firebase_admin import credentials, auth, firestore

# --- 新增：載入 Hugging Face 的 pipeline ---
from transformers import pipeline

# ------------------ 初始化設定 ------------------

# 建立一個 Flask 應用
app = Flask(__name__)

# --- Firebase Admin SDK 初始化 (維持不變) ---
cred = credentials.Certificate('serviceAccountKey.json')
firebase_admin.initialize_app(cred)
db = firestore.client()

# --- AI 模型與知識庫載入 (*** 大幅修改 ***) ---
print("正在載入 Llama 3 AI 模型...")
# 選擇 Llama 3 8B 指令微調模型
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

# 建立一個文字生成的 pipeline
# device_map="auto" 會自動偵測是否有可用的 GPU
# torch_dtype=torch.bfloat16 可以節省記憶體並加速
llama_pipeline = pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)
print("Llama 3 AI 模型載入完成！")

# 舊的知識庫相關程式碼可以全部刪除
# with open('knowledge_base.json', 'r', encoding='utf-8') as f:
#     knowledge_base = json.load(f)
# kb_questions = [item['question'] for item in knowledge_base]
# print("正在將知識庫轉換為向量...")
# kb_embeddings = model.encode(kb_questions, convert_to_tensor=True)
# print("知識庫向量轉換完成！")


# ------------------ 核心AI功能函式 (*** 全新重寫 ***) ------------------

def generate_llama_advice(user_query, user_profile):
    """
    這個函式負責將使用者問題和個人資料打包成提示，
    並呼叫 Llama 3 模型來生成個人化建議。
    (V3.0 - Llama 生成式 AI)
    """
    # 1. --- 建立系統提示 (System Prompt) ---
    # 告訴 AI 它的角色和行為準則
    system_prompt = """你是一個專業又親切的「健康管家 AI」，你的任務是根據使用者的個人健康資料和他們提出的問題，提供準確、個人化且安全的健康飲食建議。

請遵循以下規則：
1.  **使用繁體中文** 回答所有問題。
2.  你的回答應該要**整合**使用者的個人資料，而不是分開條列。
3.  如果使用者的問題涉及到他的過敏原，請**務必**在回答中提出明確的安全警告。
4.  保持回覆簡潔、溫暖、易於理解。
"""

    # 2. --- 整理使用者個人資料 ---
    # 將字典格式的 profile 轉換為人類易讀的文字
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

    # 3. --- 組合完整的對話式 Prompt ---
    # Llama 3 使用特定的對話模板，我們用 list of dicts 來建構
    messages = [
        {
            "role": "system", 
            "content": system_prompt + "\n這是正在跟你對話的使用者的個人資料：\n" + profile_text
        },
        {"role": "user", "content": user_query},
    ]

    # 4. --- 呼叫模型生成答案 ---
    # `max_new_tokens` 限制回答的長度，避免過長
    # `do_sample=True`, `temperature`, `top_p` 讓回答更有創意，不死板
    outputs = llama_pipeline(
        messages,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
    )

    # 5. --- 清理並回傳結果 ---
    # pipeline 的輸出會包含你給的 prompt，我們需要把它切掉，只留下 AI 生成的部分
    full_response = outputs[0]["generated_text"]
    # 這是從 Llama 3 回應中提取助理回答的標準方法
    assistant_response = full_response[-1]['content']
    
    return assistant_response


# ------------------ 網頁服務路由 (Routes) ------------------

# AI 問答頁 (根目錄)
@app.route('/')
def index():
    return render_template('index.html')

# (其他路由維持不變)
# ...
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

# 提供 成就頁面
@app.route('/achievements')
def achievements_page():
    return send_from_directory('static', 'achievements.html')

# 提供 BMI 計算頁面
@app.route('/bmi')
def bmi_page():
    return send_from_directory('static', 'bmi.html')

# 提供 營養分析頁面
@app.route('/nutrition')
def nutrition_page():
    return send_from_directory('static', 'nutrition.html')


# 問答 API 路由 (*** 只需修改呼叫的函式 ***)
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

        print(f"[DEBUG] 步驟 3: 準備呼叫 Llama 3 模型產生個人化建議...")
        # *** 關鍵修改：呼叫新的函式 ***
        answer = generate_llama_advice(user_question, user_profile)
        print("[DEBUG] 步驟 3: Llama 3 模型已成功回傳答案！")

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