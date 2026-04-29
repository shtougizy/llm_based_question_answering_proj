import sys, json, subprocess, re, os

MODEL_DIR   = "/home/zsy/workspace/250606/code/work/260221_gd/MOSS-TTS-Nano/models"
PROJECT_DIR = "/home/zsy/workspace/250606/code/work/260221_gd/MOSS-TTS-Nano"
TTS_PYTHON  = "/home/zsy/anaconda3/envs/tts/bin/python3"  # 明确指定 tts 环境

def clean_text(text: str) -> str:
    text = re.sub(r'\\\[[\s\S]*?\\\]', '，', text)
    text = re.sub(r'\\\([\s\S]*?\\\)', '', text)
    text = re.sub(r'\$\$[\s\S]*?\$\$', '，', text)
    text = re.sub(r'\$[^$\n]{1,80}\$', '', text)
    text = re.sub(r'```[\s\S]*?```', '，代码略，', text)
    text = re.sub(r'`[^`\n]{1,100}`', '', text)
    text = re.sub(r'<[^>]{1,50}>', '', text)
    text = re.sub(r'\b[a-zA-Z_]\w*(?:\[[\w,\s]*\])+\s*[=<>!]+[^\n]*', '', text)
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\*{1,3}([^*\n]+)\*{1,3}', r'\1', text)
    text = re.sub(
        r'[^\u4e00-\u9fff\u3000-\u303f\uff00-\uffef'
        r'a-zA-Z0-9\s，。！？、：；""'
        r"''（）【】…—,.!?;:'\"\-\n]",
        ' ', text
    )
    text = re.sub(r'\n{2,}', '。', text)
    text = re.sub(r'\s+', ' ', text).strip()
    if len(text) > 400:
        cut = text.rfind('。', 0, 400)
        text = text[:cut+1] if cut > 200 else text[:400] + '。'
    return text

def synthesize(text, output):
    # 用 tts 环境的 python 运行 infer_onnx.py，隔离依赖
    cmd = [
        TTS_PYTHON,
        f"{PROJECT_DIR}/infer_onnx.py",
        "--model-dir", MODEL_DIR,
        "--text", text,
        "--output-audio-path", output,
        "--voice", "Junhao"
    ]
    # 清除可能继承的 conda 环境变量，避免污染
    env = os.environ.copy()
    env["CONDA_DEFAULT_ENV"] = "tts"
    env["CONDA_PREFIX"] = "/home/zsy/anaconda3/envs/tts"

    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        raise RuntimeError(result.stderr[-500:])
    return output

def main():
    print(json.dumps({"ready": True}), flush=True)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            data   = json.loads(line)
            raw    = data.get("text", "")
            output = data.get("output", "/tmp/out.wav")
            text   = clean_text(raw)
            if not text:
                print(json.dumps({"ok": False, "error": "文本清洗后为空"}), flush=True)
                continue
            synthesize(text, output)
            print(json.dumps({"ok": True, "output": output}), flush=True)
        except Exception as e:
            print(json.dumps({"ok": False, "error": str(e)}), flush=True)

if __name__ == "__main__":
    main()
