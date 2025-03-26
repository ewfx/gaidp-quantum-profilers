import pandas as pd
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset
import re

# Load initial regulatory rules
rules_dict = {
    "customer_id": {
        "Field No.": 1,
        "MDRM": "CLCOM047",
        "Description": "Report the unique internal identifier",
        "Allowable Values": "Must not contain special characters and no duplicates",
        "Pattern": r"^[a-zA-Z0-9]+$",
        "Unique": True,
        "Risk": "Medium"
    },
    "transaction_amount": {
        "Field No.": 2,
        "MDRM": "TRN002",
        "Description": "Report the transaction amount",
        "Allowable Values": "Must be a positive number",
        "MinValue": 0,
        "Risk": "High"
    },
    "transaction_type": {
        "Field No.": 3,
        "MDRM": "TRN003",
        "Description": "Report the type of transaction",
        "Allowable Values": "Allowed values: Deposit, Transfer, Withdrawal",
        "AllowedValues": ["Deposit", "Transfer", "Withdrawal"],
        "Risk": "Medium"
    },
    "account_balance": {
        "Field No.": 4,
        "MDRM": "ACC004",
        "Description": "Report the account balance",
        "Allowable Values": "Must be a non-negative number",
        "MinValue": 0,
        "Risk": "High"
    },
    "transaction_date": {
        "Field No.": 5,
        "MDRM": "TRN005",
        "Description": "Report the transaction date",
        "Allowable Values": "Must be in yyyy-mm-dd format",
        "Pattern": r"^\d{4}-\d{2}-\d{2}$",
        "Risk": "Medium"
    }
}
# Load your actual dataset
input_path = "banking_data.csv"
df_input = pd.read_csv(input_path)

# Dynamic validation to identify violations in real data
def validate_data(df, rules):
    violations = []
    for field, rule in rules.items():
        if field not in df.columns:
            continue
        if rule.get("Unique", False):
            duplicate_values = df[df.duplicated(subset=field, keep=False)][field].unique()
            for idx, value in enumerate(df[field]):
                if value in duplicate_values:
                    violations.append((field, value, f"Duplicate {field}: {value}", rule['Risk']))
        for idx, value in enumerate(df[field]):
            if "Pattern" in rule and not re.match(rule["Pattern"], str(value)):
                violations.append((field, value, f"Invalid format in {field}: {value}", rule['Risk']))
            elif "AllowedValues" in rule and value not in rule["AllowedValues"]:
                violations.append((field, value, f"Invalid {field}: {value}", rule['Risk']))
            elif "MinValue" in rule and (not isinstance(value, (int, float)) or value < rule["MinValue"]):
                violations.append((field, value, f"Invalid {field}: {value} below minimum", rule['Risk']))
    return violations

# Generate training samples from real data violations
def generate_training_samples(df, rules):
    samples = []
    violations = validate_data(df, rules)
    
    # Add examples from real violations (reduced to balance with rule generation)
    for field, value, issue, risk in violations[:len(violations)//2]:  # Use half to balance
        samples.append((
            f"Validate field: {field} with value: {value}",
            f"Invalid: {issue}. Risk: {risk}"
        ))

        remediation_action = f"Correct {field} to meet {rules[field]['Allowable Values']}. Review {rules[field]['Description']}"
        if "Pattern" in rules[field]:
            remediation_action = f"Correct {field} to match the format {rules[field]['Pattern']}. Review {rules[field]['Description']}"
        elif "MinValue" in rules[field]:
            remediation_action = f"Ensure {field} is at least {rules[field]['MinValue']}. Review {rules[field]['Description']}"
        samples.append((
            f"Suggest remediation for {field} with value {value}",
            remediation_action
        ))
    
    # Add valid examples from the dataset (reduced to balance)
    for field in rules.keys():
        if field in df.columns:
            valid_values = df[field].head(3).tolist()  # Reduced from 5 to 3
            for value in valid_values:
                if not any(v[1] == value and v[0] == field for v in violations):
                    samples.append((
                        f"Validate field: {field} with value: {value}",
                        f"Valid: {rules[field]['Description']}"
                    ))
    
    # Add diverse rule generation examples (increased to improve learning)
    for field, details in rules.items():
        for _ in range(3):  # Repeat each type 3 times for emphasis
            # Basic rule
            samples.append((
                f"Generate a profiling rule for field {field}",
                f"Ensure {field} follows: {details['Allowable Values']}. Description: {details['Description']}"
            ))
            # More specific rule variations
            if "Pattern" in details:
                samples.append((
                    f"Generate a profiling rule for field {field}",
                    f"Ensure {field} matches the format {details['Pattern']}. Description: {details['Description']}"
                ))
            if "MinValue" in details:
                samples.append((
                    f"Generate a profiling rule for field {field}",
                    f"Ensure {field} is at least {details['MinValue']}. Description: {details['Description']}"
                ))
            if "AllowedValues" in details:
                samples.append((
                    f"Generate a profiling rule for field {field}",
                    f"Ensure {field} is one of {details['AllowedValues']}. Description: {details['Description']}"
                ))
            if details.get("Unique", False):
                samples.append((
                    f"Generate a profiling rule for field {field}",
                    f"Ensure {field} is unique with no duplicates. Description: {details['Description']}"
                ))
            # Generic rule for creativity
            samples.append((
                f"Generate a profiling rule for field {field}",
                f"Field {field} must comply with {details['Allowable Values']} to ensure data integrity. Description: {details['Description']}"
            ))
    
    return samples

training_data = generate_training_samples(df_input, rules_dict)

# Create training dataset
df_train = pd.DataFrame(training_data, columns=["prompt", "completion"])
train_dataset = Dataset.from_pandas(df_train)

# Load model and tokenizer
model_name = "distilgpt2"
# model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Tokenize dataset
def tokenize_function(examples):
    combined_texts = [p + " " + c for p, c in zip(examples["prompt"], examples["completion"])]
    tokens = tokenizer(combined_texts, truncation=True, padding="max_length", max_length=128)
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized_dataset = train_dataset.map(tokenize_function, batched=True)


# Set up fine-tuning arguments with more epochs
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="no",
    per_device_train_batch_size=4,
    num_train_epochs=10,  # Increased to 10 for better learning
    save_strategy="no",
    logging_dir="./logs",
    logging_steps=10,
    learning_rate=1e-5  # Lowered learning rate for better convergence
)

# Fine-tune the model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset
)

print("🚀 Fine-tuning the model...")
trainer.train()

# Save the fine-tuned model only after training completes
model.save_pretrained("./distilgpt2-rules-validator")  # Or "D:/distilgpt2-rules-validator"
tokenizer.save_pretrained("./distilgpt2-rules-validator")
print("✅ Model fine-tuned and saved!")