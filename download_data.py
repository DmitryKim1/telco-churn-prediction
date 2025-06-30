import requests
import os

def download_dataset():
    url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    save_dir = "data/raw"
    save_path = f"{save_dir}/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    
    os.makedirs(save_dir, exist_ok=True)
    
    response = requests.get(url)
    response.raise_for_status()
    
    with open(save_path, "wb") as f:
        f.write(response.content)
    
    print(f"Dataset downloaded successfully to {save_path}")
    print(f"File size: {len(response.content) / 1024:.2f} KB")

if __name__ == "__main__":
    download_dataset()