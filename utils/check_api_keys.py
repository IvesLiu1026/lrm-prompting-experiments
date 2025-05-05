import os
import re
from together import Together
from dotenv import dotenv_values

def check_key(key_name, api_key):
    os.environ["TOGETHER_API_KEY"] = api_key
    try:
        client = Together(api_key=api_key)
        response = client.chat.completions.create(
            model="deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free",
            messages=[{"role": "user", "content": "ping"}],
            max_new_tokens=5,
        )
        return True, None
    except Exception as e:
        return False, (key_name, api_key, str(e))

def main():
    env_vars = dotenv_values(".env")
    pattern = re.compile(r"^together_\d+$")

    valid_keys = []
    invalid_keys = []

    print("🔍 Checking Together API keys...\n")
    for key_name, key_value in env_vars.items():
        if pattern.match(key_name):
            result, data = check_key(key_name, key_value)
            if result:
                print(f"{key_name:<15} ✅ VALID")
                valid_keys.append((key_name, key_value))
            else:
                name, key, reason = data
                print(f"{name:<15} ❌ INVALID — {reason}")
                invalid_keys.append((name, key, reason))

    if invalid_keys:
        with open("invalid_keys.txt", "w", encoding="utf-8") as f:
            for name, key, reason in invalid_keys:
                f.write(f"{name}={key}  # {reason}\n")
        print(f"\n📝 Saved {len(invalid_keys)} invalid keys to invalid_keys.txt")
    if valid_keys:
        with open("valid_keys.txt", "w", encoding="utf-8") as f:
            for name, key in valid_keys:
                f.write(f"{name}={key}\n")
        print(f"\n📝 Saved {len(valid_keys)} valid keys to valid_keys.txt")

if __name__ == "__main__":
    main()
