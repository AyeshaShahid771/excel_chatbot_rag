import os
import tempfile

from agent.tools import query_db, session

sql = """SELECT DISTINCT
    TRIM("Name") AS contact_name
FROM
    "brand-contacts"
WHERE
    "Channel" ILIKE '%PPC%'
ORDER BY
    contact_name;
"""

print("Running query via agent.tools.query_db...")
res = query_db(sql)

print("--- Result (first 2000 chars) ---")
print(res[:2000])

# session.df may be set by the tool
try:
    df = session.df
    if df is not None:
        print(f"session.df shape: {df.shape}")
    else:
        print("session.df is None")
except Exception as e:
    print(f"Error reading session.df: {e}")

# Show debug file path
out_path = os.path.abspath(os.path.join("output", "query_debug.txt"))
print(f"Expected debug file path: {out_path}")
if os.path.exists(out_path):
    print("Found debug file. Showing head:")
    with open(out_path, "r", encoding="utf-8") as f:
        data = f.read()
        print(data[:4000])
else:
    print("Debug file not found in workspace output folder.")

# Check temp error file
err_path = os.path.join(tempfile.gettempdir(), "query_debug_errors.txt")
print(f"Temp error file path: {err_path}")
if os.path.exists(err_path):
    print("Found temp error file. Showing contents:")
    with open(err_path, "r", encoding="utf-8") as f:
        print(f.read())
else:
    print("No temp error file found.")
    print("No temp error file found.")
