import os
import json
import ast
import io
import contextlib
import pandas as pd

from dotenv import load_dotenv
from google import genai


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY not found. Put it inside your .env file."
    )

client = genai.Client(api_key=API_KEY)

MODEL = "gemini-3.8-flash"


# ============================================================
# DATA DESCRIPTION
# ============================================================

def describe_tables(tables):

    description = []

    for name, df in tables.items():

        info = {
            "name": name,
            "rows": int(len(df)),
            "columns": [str(c) for c in df.columns],
            "missing_values": {
                str(k): int(v)
                for k, v in df.isnull().sum().items()
                if int(v) > 0
            },
            "duplicate_rows": int(df.duplicated().sum()),
            "sample": df.head(10).to_dict(orient="records")
        }

        description.append(info)

    return json.dumps(description, indent=2, default=str)


# ============================================================
# SAFE CODE VALIDATION
# ============================================================

ALLOWED_NAMES = {
    "df",
    "tables",
    "pd",
    "answer",
    "result",
    "filtered",
    "grouped",
    "total",
    "average",
    "maximum",
    "minimum",
    "count"
}


ALLOWED_ATTRIBUTES = {
    "sum",
    "mean",
    "median",
    "min",
    "max",
    "count",
    "nunique",
    "idxmax",
    "idxmin",
    "sort_values",
    "groupby",
    "reset_index",
    "head",
    "tail",
    "loc",
    "iloc",
    "astype",
    "fillna",
    "dropna",
    "duplicated",
    "isnull",
    "isna",
    "value_counts",
    "unique",
    "tolist",
    "to_dict",
    "round",
    "str",
    "dt",
    "year",
    "month",
    "day"
}


def validate_code(code):

    """
    Only allow pandas-style data analysis code.

    This prevents generated code from importing modules,
    opening files, running commands, etc.
    """

    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as e:
        return False, f"Invalid Python syntax: {e}"

    for node in ast.walk(tree):

        # ----------------------------------------------------
        # BLOCK IMPORTS
        # ----------------------------------------------------

        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return False, "Imports are not allowed."

        # ----------------------------------------------------
        # BLOCK DANGEROUS FUNCTIONS
        # ----------------------------------------------------

        if isinstance(node, ast.Call):

            if isinstance(node.func, ast.Name):

                dangerous = {
                    "eval",
                    "exec",
                    "compile",
                    "open",
                    "input",
                    "__import__",
                    "globals",
                    "locals",
                    "getattr",
                    "setattr",
                    "delattr"
                }

                if node.func.id in dangerous:
                    return False, f"Function '{node.func.id}' is not allowed."

        # ----------------------------------------------------
        # BLOCK DANGEROUS ATTRIBUTES
        # ----------------------------------------------------

        if isinstance(node, ast.Attribute):

            if node.attr.startswith("__"):
                return False, "Private attributes are not allowed."

            if node.attr not in ALLOWED_ATTRIBUTES:
                return False, f"Attribute '{node.attr}' is not allowed."

        # ----------------------------------------------------
        # CHECK VARIABLE NAMES
        # ----------------------------------------------------

        if isinstance(node, ast.Name):

            if isinstance(node.ctx, ast.Load):

                if node.id not in ALLOWED_NAMES:

                    # Allow normal temporary variables
                    # but block suspicious names.
                    if node.id.startswith("__"):
                        return False, f"Name '{node.id}' is not allowed."

        # ----------------------------------------------------
        # BLOCK LAMBDA
        # ----------------------------------------------------

        if isinstance(node, ast.Lambda):
            return False, "Lambda functions are not allowed."

    return True, ""


# ============================================================
# EXECUTE GENERATED CODE
# ============================================================

def execute_generated_code(code, tables):

    valid, error = validate_code(code)

    if not valid:
        return None, error

    local_vars = {
        "pd": pd,
        "tables": tables
    }

    # If only one file was uploaded,
    # expose it as df.
    if len(tables) == 1:
        local_vars["df"] = list(tables.values())[0]

    output = io.StringIO()

    try:

        with contextlib.redirect_stdout(output):

            exec(
                code,
                {
                    "__builtins__": {
                        "str": str,
                        "int": int,
                        "float": float,
                        "len": len,
                        "round": round,
                        "bool": bool
                    }
                },
                local_vars
            )

        answer = local_vars.get("answer")

        if answer is None:

            printed = output.getvalue().strip()

            if printed:
                answer = printed
            else:
                return None, "Generated code did not produce an answer."

        return answer, ""

    except Exception as e:

        return None, f"Generated code failed: {e}"


# ============================================================
# FORMAT ANSWER
# ============================================================

def format_answer(answer):

    if isinstance(answer, pd.DataFrame):
        return answer.to_string(index=False)

    if isinstance(answer, pd.Series):
        return answer.to_string()

    if hasattr(answer, "item"):

        try:
            return str(answer.item())
        except Exception:
            pass

    return str(answer)


# ============================================================
# MAIN ANALYSIS FUNCTION
# ============================================================

def analyze(tables, question):

    if not tables:
        return {
            "can_determine": False,
            "answer": "",
            "code": "",
            "evidence": "",
            "reason": "No data was uploaded."
        }

    if not question or not question.strip():
        return {
            "can_determine": False,
            "answer": "",
            "code": "",
            "evidence": "",
            "reason": "No question was provided."
        }

    # --------------------------------------------------------
    # Describe uploaded data
    # --------------------------------------------------------

    data_description = describe_tables(tables)

    # --------------------------------------------------------
    # AI PROMPT
    # --------------------------------------------------------

    prompt = f"""
You are a STRICT proof-carrying AI data analyst.

Your job is to answer the user's question using ONLY the uploaded
tables.

UPLOADED DATA:

{data_description}


USER QUESTION:

{question}


============================================================
CORE RULES
============================================================

1. NEVER invent or guess data.

2. Every numerical answer MUST come from executable Python code.

3. The Python code MUST store the final result in:

answer = ...

4. Use pandas.

5. If there is only one uploaded table, use:

df

6. If multiple tables are required, use:

tables["filename"]

7. Do NOT create fake data.

8. Do NOT use external information.

9. If the requested information is missing, return:

can_determine = false


============================================================
SEMANTIC RULES
============================================================

"total", "total revenue", "overall revenue", "sum"

-> use SUM.

Example:

answer = df["Revenue"].sum()


"highest", "maximum", "largest"

-> use MAX.

Example:

answer = df["Revenue"].max()


"lowest", "minimum", "smallest"

-> use MIN.

Example:

answer = df["Revenue"].min()


"average", "mean"

-> use MEAN.

Example:

answer = df["Revenue"].mean()


"how many", "count", "number of"

-> use COUNT or len appropriately.


"which product has the highest revenue"

-> calculate revenue by product and select the product with
the highest total.

Example:

answer = df.groupby("Product")["Revenue"].sum().idxmax()


============================================================
IMPORTANT
============================================================

DO NOT confuse:

TOTAL revenue
with
HIGHEST revenue.

For:

"What is the total revenue?"

you MUST use:

answer = df["Revenue"].sum()

NOT:

df["Revenue"].max()


============================================================
MESSY DATA
============================================================

Look for:

- missing values
- duplicate rows
- ambiguous dates
- missing columns
- insufficient information
- inconsistent values
- contradictory tables

If the question cannot be answered reliably:

can_determine = false


============================================================
OUTPUT
============================================================

Return ONLY valid JSON.

For an answerable question:

{{
    "can_determine": true,
    "code": "answer = df['Revenue'].sum()",
    "reason": "",
    "explanation": "The Revenue column was summed to calculate total revenue."
}}


For an unanswerable question:

{{
    "can_determine": false,
    "code": "",
    "reason": "The uploaded data does not contain the information required.",
    "explanation": ""
}}
"""

    # --------------------------------------------------------
    # CALL GEMINI
    # --------------------------------------------------------

    try:

        response = client.models.generate_content(
            model=MODEL,
            contents=prompt
        )

        text = response.text.strip()

        # Remove markdown fences
        if text.startswith("```"):

            text = text.replace("```json", "")
            text = text.replace("```JSON", "")
            text = text.replace("```", "")
            text = text.strip()

        result = json.loads(text)

    except Exception as e:

        return {
            "can_determine": False,
            "answer": "",
            "code": "",
            "evidence": "",
            "reason": "AI response error: " + str(e)
        }

    # --------------------------------------------------------
    # HANDLE CANNOT DETERMINE
    # --------------------------------------------------------

    if not result.get("can_determine", False):

        reason = result.get(
            "reason",
            "The answer cannot be determined from the uploaded data."
        )

        return {
            "can_determine": False,
            "answer": "",
            "code": "",
            "evidence": reason,
            "reason": reason
        }

    # --------------------------------------------------------
    # GET GENERATED CODE
    # --------------------------------------------------------

    generated_code = result.get("code", "").strip()

    generated_code = generated_code.replace(
        "```python", ""
    )

    generated_code = generated_code.replace(
        "```", ""
    ).strip()

    if not generated_code:

        return {
            "can_determine": False,
            "answer": "",
            "code": "",
            "evidence": "",
            "reason": "AI did not generate executable proof code."
        }

    # --------------------------------------------------------
    # EXECUTE CODE
    # --------------------------------------------------------

    answer, execution_error = execute_generated_code(
        generated_code,
        tables
    )

    if execution_error:

        return {
            "can_determine": False,
            "answer": "",
            "code": generated_code,
            "evidence": "",
            "reason": execution_error
        }

    # --------------------------------------------------------
    # FORMAT RESULT
    # --------------------------------------------------------

    formatted_answer = format_answer(answer)

    explanation = result.get(
        "explanation",
        "The answer was calculated from the uploaded data."
    )

    evidence = (
        "PROOF:\n"
        "The displayed answer was produced by executing the "
        "generated Python code on the uploaded dataset.\n\n"
        "EXPLANATION:\n"
        + explanation
    )

    # --------------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------------

    return {
        "can_determine": True,
        "answer": formatted_answer,
        "code": generated_code,
        "evidence": evidence,
        "reason": ""
    }