import sys
import pandas as pd


def trans(csv_path, output_path):
    csv_file = pd.read_csv(csv_path, sep=",", header=0, encoding="utf-8")
    # 储存格式为一个长列表，元素为每一行的字典
    csv_file.to_json(output_path, orient="records", force_ascii=False, lines=True)
    print("Finished!")


if __name__ == "__main__":
    # python csv_to_json.py movie_comments.csv movie_comments.json
    csv_path = sys.argv[1]
    output_path = sys.argv[2]
    trans(csv_path, output_path)