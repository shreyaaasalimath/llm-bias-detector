from pipeline.runner import run_all
from pipeline.analyzer import analyze
from pipeline.db import init_db, save_results

def main():
    print("=== Initializing database ===")
    init_db()

    print("\n=== Running demographic bias prompts ===")
    demo_results = run_all("prompts/demographic.json", "demographic")

    print("\n=== Running political bias prompts ===")
    political_results = run_all("prompts/political.json", "political")

    all_results = demo_results + political_results

    print(f"\n=== Analyzing {len(all_results)} responses ===")
    analyzed = []
    for i, r in enumerate(all_results):
        print(f"  Analyzing {i+1}/{len(all_results)}...")
        analyzed.append(analyze(r))

    print("\n=== Saving results ===")
    save_results(analyzed)
    print("Done! Now run: streamlit run dashboard/app.py")

if __name__ == "__main__":
    main()