import anthropic
import json
import itertools
import time
from dotenv import load_dotenv

load_dotenv()
client = anthropic.Anthropic()

def generate_prompt_variants(prompt_file):
    with open(prompt_file) as f:
        data = json.load(f)
    variants = []
    for template, group in itertools.product(data["templates"], data["groups"]):
        variants.append({
            "template": template,
            "group": group,
            "prompt": template.replace("{group}", group)
        })
    return variants

def run_prompt(prompt):
    try:
        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text
    except Exception as e:
        print(f"  Error: {e}")
        return ""

def run_all(prompt_file, bias_type):
    variants = generate_prompt_variants(prompt_file)
    results = []
    total = len(variants)
    for i, v in enumerate(variants):
        print(f"  [{i+1}/{total}] Group: {v['group']}")
        response = run_prompt(v["prompt"])
        results.append({
            "bias_type": bias_type,
            "template": v["template"],
            "group": v["group"],
            "prompt": v["prompt"],
            "response": response
        })
        time.sleep(0.3)  # avoid rate limits
    return results