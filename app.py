# 載入我們需要的函式庫
from flask import Flask, request, jsonify, render_template
from sentence_transformers import SentenceTransformer, util
import json
import torch

# ------------------ 初始化設定 ------------------

# 建立一個 Flask 應用
app = Flask(__name__)

# 載入一個預訓練好的句向量模型
# 'distiluse-base-multilingual-cased-v1' 是一個效能不錯且支援多國語言（包含中文）的模型
print("正在載入AI模型...")
model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
print("AI模型載入完成！")

# 從 JSON 檔案中讀取我們的知識庫
with open('knowledge_base.json', 'r', encoding='utf-8') as f:
    knowledge_base = json.load(f)

# 從知識庫中，只取出所有的「問題」
# 我們將用這些問題來跟使用者的輸入做比對
kb_questions = [item['question'] for item in knowledge_base]

# 將知識庫中的所有問題預先轉換成向量(Embeddings)，並存起來
# 這樣未來在比對時，就不用每次都重新計算，可以大大加速
print("正在將知識庫轉換為向量...")
kb_embeddings = model.encode(kb_questions, convert_to_tensor=True)
print("知識庫向量轉換完成！")


# ------------------ 核心AI功能函式 ------------------

def find_best_match(user_query):
    """
    這個函式負責找出與使用者問題最相近的答案
    """
    # 1. 將使用者的問題也轉換成向量
    query_embedding = model.encode(user_query, convert_to_tensor=True)

    # 2. 計算使用者問題的向量 與 知識庫中所有問題向量 的「餘弦相似度」
    # 餘弦相似度的分數介於 -1 到 1 之間，分數越高代表語意上越相近
    cosine_scores = util.cos_sim(query_embedding, kb_embeddings)

    # 3. 找出分數最高的那個問題是在第幾個位置 (index)
    best_match_idx = torch.argmax(cosine_scores)

    # 4. 檢查最高分是否高於一個門檻值，避免回答完全不相關的問題
    if cosine_scores[0][best_match_idx] > 0.5: # 0.5 是一個可以自己調整的信心分數
        # 5. 如果夠高分，就回傳對應的答案
        return knowledge_base[best_match_idx]['answer']
    else:
        # 6. 如果分數太低，就表示我們的知識庫沒有相關答案
        return "抱歉，我還不知道這個問題的答案。我會繼續學習的！"


# ------------------ 網頁服務路由 (Routes) ------------------

# 定義網站首頁的路由
@app.route('/')
def index():
    # 當使用者打開網站首頁時，回傳 index.html 這個網頁
    return render_template('index.html')

# 定義處理問答請求的 API 路由
@app.route('/ask', methods=['POST'])
def ask():
    # 取得前端網頁用 POST 方法傳來的 JSON 資料
    user_data = request.json
    user_question = user_data.get('question')

    if not user_question:
        return jsonify({'answer': '你沒有問問題喔！'})

    # 呼叫我們的 AI 核心函式來找答案
    answer = find_best_match(user_question)

    # 將找到的答案用 JSON 格式回傳給前端網頁
    return jsonify({'answer': answer})


# ------------------ 啟動伺服器 ------------------

# 如果這個檔案是直接被執行 (而不是被其他檔案引用)
if __name__ == '__main__':
    # 就啟動 Flask 伺服器，並開放讓同個網路下的裝置連線
    app.run(host='0.0.0.0', port=5000, debug=True)
