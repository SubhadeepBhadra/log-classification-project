import sys
import requests
import json
import pandas as pd
from io import BytesIO

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Test the API endpoint
BASE_URL = "http://127.0.0.1:8000"

# Create a test CSV file in memory
test_data = {
    "source": [
        "LegacyCRM",
        "LegacyCRM",
        "ModernCRM",
        "BillingSystem"
    ],
    "log_message": [
        "Case escalation for ticket ID 7324 failed because the assigned support agent is no longer active.",
        "The 'ReportGenerator' module will be retired in version 4.0. Please migrate to the 'AdvancedAnalyticsSuite' by Dec 2025",
        "IP 192.168.133.114 blocked due to potential attack",
        "User 12345 logged in successfully."
    ]
}

df = pd.DataFrame(test_data)
csv_buffer = BytesIO()
df.to_csv(csv_buffer, index=False)
csv_buffer.seek(0)

print("=" * 60)
print("Testing Log Classification API")
print("=" * 60)
print("\nTest Data:")
print(df.to_string())

# Test the /classify/ endpoint
print("\n" + "=" * 60)
print("Making POST request to /classify/ endpoint...")
print("=" * 60)

try:
    files = {'file': ('test_logs.csv', csv_buffer, 'text/csv')}
    response = requests.post(f"{BASE_URL}/classify/", files=files)
    
    print(f"\nStatus Code: {response.status_code}")
    
    if response.status_code == 200:
        print("✅ Classification successful!")
        print("\nClassified Output:")
        
        # Read the response CSV
        result_df = pd.read_csv(BytesIO(response.content))
        print(result_df.to_string())
        
        # Save the output to file
        output_file = "api_test_output.csv"
        result_df.to_csv(output_file, index=False)
        print(f"\n✅ Results saved to {output_file}")
        
    else:
        print(f"❌ Error: {response.status_code}")
        print(f"Response: {response.text}")
        
except Exception as e:
    print(f"❌ Error making request: {e}")
    print("\nMake sure the server is running on http://127.0.0.1:8000")

print("\n" + "=" * 60)
print("Test Complete")
print("=" * 60)
