import streamlit as st
import pandas as pd
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from sklearn.ensemble import IsolationForest

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

# Load the fine-tuned model
# Load the fine-tuned model
model_path = "./distilgpt2-rules-validator"  # Update to new path
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)
generator = pipeline("text-generation", model=model, tokenizer=tokenizer, max_length=100)

# Load input dataset
input_path = "banking_data.csv"
df_input = pd.read_csv(input_path)

# Unsupervised learning for anomaly detection
numerical_cols = [col for col in df_input.columns if col in ["transaction_amount", "account_balance"]]
if numerical_cols:
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    anomalies = iso_forest.fit_predict(df_input[numerical_cols].fillna(0))
    anomaly_indices = df_input.index[anomalies == -1].tolist()
else:
    anomaly_indices = []

# Validation with GenAI integration
def validate_data(df, rules, generator, anomaly_indices):
    violations = []
    for field, rule in rules.items():
        if field not in df.columns:
            continue
        
        # Rule-based checks for duplicates
        if rule.get("Unique", False):
            duplicate_values = df[df.duplicated(subset=field, keep=False)][field].unique()
            for idx, value in enumerate(df[field]):
                if value in duplicate_values:
                    prompt = f"Validate field: {field} with value: {value}"
                    validation_output = generator(prompt)[0]['generated_text'].split("Invalid: ")[-1].strip() if "Invalid: " in generator(prompt)[0]['generated_text'] else generator(prompt)[0]['generated_text'].strip()
                    remediation_prompt = f"Suggest remediation for {field} with value {value}"
                    remediation_output = generator(remediation_prompt)[0]['generated_text'].split("Correct ")[-1].strip() if "Correct " in generator(remediation_prompt)[0]['generated_text'] else generator(remediation_prompt)[0]['generated_text'].strip()
                    violations.append((idx + 1, f"Duplicate {field}: {value}. {validation_output}", remediation_output))
        
        # Other validation checks
        for idx, value in enumerate(df[field]):
            issue = None
            if "Pattern" in rule and not re.match(rule["Pattern"], str(value)):
                issue = f"Invalid format in {field}: {value}"
            elif "AllowedValues" in rule and value not in rule["AllowedValues"]:
                issue = f"Invalid {field}: {value}"
            elif "MinValue" in rule and (not isinstance(value, (int, float)) or value < rule["MinValue"]):
                issue = f"Invalid {field}: {value} below minimum"
            
            if issue or idx in anomaly_indices:
                # Use GenAI for validation
                prompt = f"Validate field: {field} with value: {value}"
                validation_output = generator(prompt)[0]['generated_text'].split("Invalid: ")[-1].strip() if "Invalid: " in generator(prompt)[0]['generated_text'] else generator(prompt)[0]['generated_text'].strip()
                
                # Use GenAI for remediation
                remediation_prompt = f"Suggest remediation for {field} with value {value}"
                remediation_output = generator(remediation_prompt)[0]['generated_text'].split("Correct ")[-1].strip() if "Correct " in generator(remediation_prompt)[0]['generated_text'] else generator(remediation_prompt)[0]['generated_text'].strip()
                
                # Add anomaly flag if detected
                if idx in anomaly_indices:
                    validation_output += " (Anomaly Detected)"
                
                violations.append((idx + 1, f"Field: {field} - Value: {value}. {validation_output}", remediation_output))
    
    return violations

# Perform validation
validation_issues = validate_data(df_input, rules_dict, generator, anomaly_indices)

# Streamlit Dashboard
st.title("🚀 AI-Powered Data Compliance Dashboard")
if not validation_issues:
    st.success("✅ All records are valid!")
else:
    st.subheader("Validation Issues")
    for issue in validation_issues:
        row, validation, remediation = issue
        st.error(f"Row {row}: {validation}")
        if remediation:
            st.write(f"**Remediation Suggestion**: {remediation}")

# Generate new rules
st.subheader("Generate New Profiling Rules")
field_input = st.text_input("Enter a field name to generate a rule for:")
if field_input:
    # rule_prompt = f"Generate a profiling rule for field {field_input}"
    rule_prompt = f"Generate a profiling rule for the field {field_input} to ensure compliance with regulatory requirements, without referencing any specific value"
    generated_text = generator(rule_prompt)[0]['generated_text']
    
    # Debug: Print raw model output
    print(f"Raw model output for rule generation: {generated_text}")
    
    # Improved output parsing
    if "Ensure " in generated_text:
        new_rule = generated_text.split("Ensure ")[-1].strip()
    elif "Field " in generated_text:
        new_rule = generated_text.split("Field ")[-1].strip()
    elif "Description: " in generated_text:
        new_rule = generated_text.split("Description: ")[0].strip()
    else:
        new_rule = generated_text.strip()
    
    # Clean up the output to remove the prompt if it’s repeated
    prompt_prefix = f"Generate a profiling rule for field {field_input}"
    if new_rule.startswith(prompt_prefix):
        new_rule = new_rule[len(prompt_prefix):].strip()
    if new_rule.startswith(":"):
        new_rule = new_rule[1:].strip()
    
    st.write(f"**Generated Rule**: {new_rule}")