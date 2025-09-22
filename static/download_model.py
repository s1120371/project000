# download_model.py (修正版)
from huggingface_hub import hf_hub_download
import os

# --- 設定 ---
repo_id = "TheBloke/Meta-Llama-3-8B-Instruct-GGUF"
# 錯誤點：原來的檔名用的是小寫 'm'
# 正確的檔名開頭應該是大寫 'M'
filename = "Meta-Llama-3-8B-Instruct.Q4_K_M.gguf" # <<<--- 就是這裡！'m' 已修正為 'M'
# 下載到當前資料夾
local_dir = "."

# --- 開始下載 ---
print(f"正在從 {repo_id} 下載 {filename}...")

# 檢查檔案是否已存在，避免重複下載
target_path = os.path.join(local_dir, filename)
if os.path.exists(target_path):
    print(f"檔案 '{target_path}' 已存在，跳過下載。")
else:
    hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=local_dir,
        local_dir_use_symlinks=False # 在 Windows 上建議設為 False
    )
    print(f"下載完成！檔案已儲存至 '{target_path}'")