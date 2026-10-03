from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_FILE = PROJECT_ROOT / "data" / "raw" / "flipkart" / "flipkart_com-ecommerce_sample.csv"

df = pd.read_csv(RAW_FILE)

duplicate_ids = [
    "ACCE54YZ3PUHU2RM",
    "ACCEDCYTFNHAVRKN",
    "BRAE3CYXV4JGWEDA",
    "BRAE3UG9NGHDQZZE",
    "BRAE3UG9P8GJXZ2X",
    "BRAE3WVQGWYEUEZ4",
    "BRAE3WVQSRGWGHZV",
    "BRAE6AV3FZNGNTXH",
    "BRAE6AV3HNZX3TWR",
    "BRAE6AV3PYGAVCYZ",
    "BRAE6AV3YVDFJZH3",
    "BRAE6AV3ZZQHKDCN",
    "BRAEAZE23UATU5MQ",
    "BRAEAZEFT8RSHGJJ",
    "BRAEAZEFXGVCTJJN",
    "BRAEAZEFXHCKE3RG",
    "BRAEAZEFZYER5JHJ",
    "CAGE5GYKTZWZSHXN",
    "SAREGTVEX3ZNBJNY",
    "SAREGTVFFTURURNH",
    "SAREGTVFGRYRDCNH",
    "SAREGTVFZG6KEGHB",
    "SFFEHUAUGA5GBTYZ",
    "SHOE8APHN5RKHCDY",
    "SHOEFZFHGGMUSMDK",
    "SHOEGAJFAU2N5RUR",
    "SHOEGAJGF5CSZNHA",
    "SHOEGAJGGTZD6HSY",
    "SHOEGHFQKGQVSGHH",
    "SHOEGJK85Z3TSXZE",
    "SHOEJDYYY77DZF6Z",
    "SHOEJKQNFKGHKDWS",
    "SKIE8GB5GXGYSGGR",
    "SNDEHG3DNVBFGAT5",
    "SNDEHZKGKGHDFG57",
    "SNDEHZKGZHHHQVYX",
    "SNDEJMK8HWAQEPQY",
    "SNDEJMK8J2ZEZJKX",
    "SNDEJMK8PGUGVXZQ",
    "TMLEEVFEGZRYJTAY",
    "TMLEEVFEU66VGFGC",
    "USGE8R7KG4TYZPV5",
    "USGEBHHGHGUYVCZ8",
    "USGEBZZ3SVFAFF5E",
    "USGEBZZ3Y567FZKP",
    "USGEJPQHFM7CVCYH",
    "USGEJPQHT8YPZ7D8",
    "USGEJPQHZHYZQHBU",
    "USGEJPQKBNS5KDTE",
]

result = df[df["pid"].astype(str).isin(duplicate_ids)]

columns = [
    "pid",
    "product_name",
    "product_category_tree",
    "description",
    "brand",
]

print(f"Found {len(result)} overlapping products.\n")
print("=" * 100)

for _, row in result[columns].iterrows():
    print("ID:", row["pid"])
    print("PRODUCT:", row["product_name"])
    print("CATEGORY:", row["product_category_tree"])
    print("BRAND:", row["brand"])
    print("-" * 100)