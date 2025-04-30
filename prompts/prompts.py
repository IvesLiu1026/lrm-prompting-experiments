# prompts.py

def standard(question, choices):
    instruction = '''Please answer the following question and write your answer after "The answer is" in the format: The answer is [(choice)] [choice content].'''
    choice_str = f"(a) {choices[0]} (b) {choices[1]} (c) {choices[2]} (d) {choices[3]}"
    return instruction + f"\nQuestion: {question}\nChoices: {choice_str}"

def slow(question, choices):
    instruction = '''Please think more slowly and thoroughly. Then, answer the following question and write your answer after "The answer is" in the format: The answer is [(choice)] [choice content]'''
    choice_str = f"(a) {choices[0]} (b) {choices[1]} (c) {choices[2]} (d) {choices[3]}"
    return instruction + f"\nQuestion: {question}\nChoices: {choice_str}"

def quick(question, choices):
    instruction = '''Please think quickly and efficiently. Then, answer the following question and write your answer after "The answer is" in the format: The answer is [(choice)] [choice content].'''
    choice_str = f"(a) {choices[0]} (b) {choices[1]} (c) {choices[2]} (d) {choices[3]}"
    return instruction + f"\nQuestion: {question}\nChoices: {choice_str}"

def adaptive(question, choices):
    instruction = '''Please answer the following question and write your answer after \"The answer is\" in the format: The answer is [(choice)] [choice content]. 
If the question seems difficult to you, slow down and think carefully. If it seems easy, think quickly and respond promptly.'''
    choice_str = f"(a) {choices[0]} (b) {choices[1]} (c) {choices[2]} (d) {choices[3]}"
    return instruction + f"\nQuestion: {question}\nChoices: {choice_str}"

def without_wait(question, choices):
    instruction = '''Please answer the following question but "cannot use "Wait"" in your <think></think> section. Then, write your answer after "The answer is" in the format: The answer is [(choice)] [choice content].'''
    choice_str = f"(a) {choices[0]} (b) {choices[1]} (c) {choices[2]} (d) {choices[3]}"
    return instruction + f"\nQuestion: {question}\nChoices: {choice_str}"

def smart(question, choices):
    instruction = '''Please answer the following question and think smartly about the answer. Write your answer after "The answer is" in the format: The answer is [(choice)] [choice content].'''
    choice_str = f"(a) {choices[0]} (b) {choices[1]} (c) {choices[2]} (d) {choices[3]}"
    return instruction + f"\nQuestion: {question}\nChoices: {choice_str}"

def stupid(question, choices):
    instruction = '''Please answer the following question and think stupidly about the answer. Write your answer after "The answer is" in the format: The answer is [(choice)] [choice content].'''
    choice_str = f"(a) {choices[0]} (b) {choices[1]} (c) {choices[2]} (d) {choices[3]}"
    return instruction + f"\nQuestion: {question}\nChoices: {choice_str}"

def difficulty_aware(question, choices):
    instruction = '''Before answering, briefly assess how difficult the question is (easy / medium / hard).

    - If easy, answer directly.
    - If medium, provide 1-2 sentences of explanation.
    - If hard, provide detailed reasoning before answering.

    Then write your answer in the format:
    The answer is [(choice)] [choice content].'''
    choice_str = f"(a) {choices[0]} (b) {choices[1]} (c) {choices[2]} (d) {choices[3]}"
    return instruction + f"\nQuestion: {question}\nChoices: {choice_str}"

def fast_thinking(question, choices):
    instruction = '''
    You are a fast-thinking expert.

    Start thinking immediately with your first reasoning step. 
    Be decisive and direct.

    Respond in this format:
    <think>
    [Your quick reasoning here.]
    </think>

    Then write your answer in the format:
    The answer is [(choice)] [choice content].
    '''
    choice_str = f"(a) {choices[0]} (b) {choices[1]} (c) {choices[2]} (d) {choices[3]}"
    return instruction + f"\nQuestion: {question}\nChoices: {choice_str}"

def minimalist(question, choices):
    instruction = '''Answer the following question using the fewest words possible. Avoid explanations unless necessary.

    Format: The answer is [(choice)] [choice content].'''
    choice_str = f"(a) {choices[0]} (b) {choices[1]} (c) {choices[2]} (d) {choices[3]}"
    return instruction + f"\nQuestion: {question}\nChoices: {choice_str}"

def fast_confident(question, choices):
    instruction = '''
You are a confident expert.

Answer with speed and clarity.
Do not use filler words like "Wait", "Hmm", or "Let me think."
Start immediately with your reasoning, and keep it concise.

Respond in this format:
<think>
[One or two decisive sentences]
</think>

Then write your answer in this format:
The answer is [(choice)] [choice content].
    '''
    choice_str = f"(a) {choices[0]} (b) {choices[1]} (c) {choices[2]} (d) {choices[3]}"
    return instruction.strip() + f"\nQuestion: {question}\nChoices: {choice_str}"

def no_explanation(question, choices):
    instruction = '''
You must answer the question without providing any explanation or justification. Just give the answer directly in this format:
The answer is [(choice)] [choice content].
'''
    choice_str = f"(a) {choices[0]} (b) {choices[1]} (c) {choices[2]} (d) {choices[3]}"
    return instruction.strip() + f"\nQuestion: {question}\nChoices: {choice_str}"

def meta_reasoning(question, choices):
    instruction = '''
You are tasked with answering the following question.

- First, determine if the question requires reasoning.
    - If reasoning is required:
        - If the reasoning seems difficult, think slowly and carefully, making sure every step is solid before answering.
        - If the reasoning seems easy, think quickly and confidently like an expert, answering with speed and clarity, without hesitation or unnecessary words and keep it "concise" (e.g., NEVER say "wait", "hmm" or "let me think").
    - If no reasoning is required (the answer can be directly recalled or inferred based on facts), immediately answer based on known information, NEVER overthinking or unnecessary reasoning.

Format your calculations and reasoning steps clearly if needed. Otherwise, provide the final answer directly.

Then write your answer in this format:
The answer is [(choice)] [choice content].
'''
    choice_str = f"(a) {choices[0]} (b) {choices[1]} (c) {choices[2]} (d) {choices[3]}"
    return instruction.strip() + f"\nQuestion: {question}\nChoices: {choice_str}"

prompt_map = {
    "standard": standard,
    "slow": slow,
    "quick": quick,
    "adaptive": adaptive,
    "without_wait": without_wait,
    "smart": smart,
    "stupid": stupid,
    "difficulty_aware": difficulty_aware,
    "fast_thinking": fast_thinking,
    "minimalist": minimalist,
    "fast_confident": fast_confident,
    "no_explanation": no_explanation,
    "meta_reasoning": meta_reasoning,
}
