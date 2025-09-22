# app.py (GGUF + llama-cpp-python 加速版，所有已知問題已修正)

from flask import Flask, request, jsonify, send_from_directory
import firebase_admin
from firebase_admin import credentials, auth, firestore
import os

# --- 新增：載入 llama-cpp-python ---
from llama_cpp import Llama

# ------------------ 初始化設定 ------------------

# 建立一個 Flask 應用
app = Flask(__name__)


# --- Firebase Admin SDK 初始化 (維持不變) ---
cred = credentials.Certificate('serviceAccountKey.json')
firebase_admin.initialize_app(cred)
db = firestore.client()

# --- AI 模型載入 (*** 使用 Llama CPP ***) ---
print("正在載入 GGUF AI 模型...")

# 確保這個檔名與您下載的檔案完全一致
model_filename = "Meta-Llama-3-8B-Instruct.Q4_K_M.gguf"

# 檢查模型檔案是否存在
if not os.path.exists(model_filename):
    print(f"錯誤：找不到模型檔案 '{model_filename}'！請確認檔案是否在同一個資料夾中。")
    exit()

llm = Llama(
    model_path=model_filename,
    n_ctx=4096,       # 模型可以處理的上下文長度
    n_gpu_layers=0,   # 設為 0 強制使用 CPU
    verbose=True      # 在終端機印出詳細載入資訊
)
print(f"GGUF AI 模型 '{model_filename}' 載入完成！")


# ------------------ 核心AI功能函式 ------------------

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
        messages=messages,
        max_tokens=512,
        temperature=0.7,
    )
    answer = response['choices'][0]['message']['content']
    return answer


# ------------------ 網頁服務路由 (指向 templates 資料夾) ------------------

@app.route('/')
def serve_index():
    return send_from_directory('templates', 'index.html')

@app.route('/login')
def serve_login():
    return send_from_directory('templates', 'login.html')
    
@app.route('/home')
def serve_home():
    return send_from_directory('templates', 'home.html')

@app.route('/edit-profile')
def serve_edit_profile():
    return send_from_directory('templates', 'edit-profile.html')

@app.route('/achievements')
def serve_achievements():
    return send_from_directory('templates', 'achievements.html')
    
@app.route('/bmi')
def serve_bmi():
    return send_from_directory('templates', 'bmi.html')

@app.route('/nutrition')
def serve_nutrition():
    return send_from_directory('templates', 'nutrition.html')


# ------------------ 問答 API 路由 (*** 變數名稱已修正 ***) ------------------

@app.route('/ask', methods=['POST'])
def ask():
    print("\n[DEBUG] 收到一個新的 /ask 請求")
    user_data = request.json
    user_question = user_data.get('question')
    id_token = user_data.get('idToken')

    if not user_question:
        return jsonify({'answer': '你沒有問問題喔！'})
    if not id_token:
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
            return jsonify({'answer': '錯誤：找不到您的使用者設定檔。'})

        print(f"[DEBUG] 步驟 3: 準備呼叫 GGUF 模型產生個人化建議...")
        # *** 修正點：將 user_query 改為 user_question ***
        answer = generate_llama_advice(user_question, user_profile)
        print("[DEBUG] 步驟 3: GGUF 模型已成功回傳答案！")

        print("[DEBUG] 步驟 4: 準備將最終答案傳回前端")
        return jsonify({'answer': answer})

    except auth.InvalidIdTokenError as e:
        print(f"[ERROR] ID Token 無效: {e}")
        return jsonify({'answer': '錯誤：無效的驗證憑證，請重新登入。'}), 401
    except Exception as e:
        print(f"[CRITICAL ERROR] 發生未預期的嚴重錯誤: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'answer': f'伺服器發生未預期的錯誤: {e}'}), 500


# ------------------ 啟動伺服器 ------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

