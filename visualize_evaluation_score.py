import json
import matplotlib.pyplot as plt

# Load the data
file_path = "single_question_lora_test_result.json"
with open(file_path, "r") as f:
    data = json.load(f)

# Prepare data for scatter plot
points = []
for trait, levels in data.items():
    for label, entries in levels.items():
        for entry in entries:
            logit = entry["logit"]
            points.append({
                "trait": trait,
                "label": label,
                "logit": logit
            })

# Create scatter plot
plt.figure(figsize=(10, 6))
colors = {"high": "green", "low": "red"}
for point in points:
    plt.scatter(point["trait"], point["logit"], marker='x', color=colors.get(point["label"].lower(), "gray"), label=point["label"].capitalize())

# Avoid duplicate labels in legend
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())

plt.title("Logit Score Scatter Plot by Trait")
plt.xlabel("Trait")
plt.ylabel("Logit Score")
plt.grid(True)
plt.tight_layout()
plt.savefig("single_question_lora_test_result.png", dpi=300)
