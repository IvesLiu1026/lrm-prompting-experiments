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
        return False, str(e)

def main():
    env_vars = dotenv_values(".env")  # Only loads uncommented key-value pairs
    pattern = re.compile(r"^together_\d+$")

    invalid_keys = []

    print("🔍 Checking Together API keys...\n")
    for key_name, key_value in env_vars.items():
        if pattern.match(key_name):
            valid, error = check_key(key_name, key_value)
            if valid:
                print(f"{key_name:<15} ✅ VALID")
            else:
                print(f"{key_name:<15} ❌ INVALID — {error}")
                invalid_keys.append((key_name, key_value, error))

    if invalid_keys:
        with open("invalid_keys.txt", "w", encoding="utf-8") as f:
            for name, key, reason in invalid_keys:
                f.write(f"{name}={key}  # {reason}\n")
        print(f"\n📝 Saved {len(invalid_keys)} invalid keys to invalid_keys.txt")
    else:
        print("\n🎉 All keys are valid!")

if __name__ == "__main__":
    main()
