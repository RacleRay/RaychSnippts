import json
import threading
import multiprocessing as mp
import tqdm
from raych.preprocess.nlp.stringAlgo import text_segmentate


def json2txt(from_json_dir, to_txt_dir, thread_idx, num_threads):
    with open(from_json_dir, "r", encoding="utf-8") as f_in, \
            open(to_txt_dir, "w", encoding="utf-8", buffering=20480) as f_out:
        for i, line in tqdm.tqdm(enumerate(f_in)):
            if i % num_threads != thread_idx:
                continue

            line = line.strip()
            if not line: continue

            line = json.loads(line)

            title_ = line["title"].strip()
            content_ = line["content"].strip()

            if title_:
                f_out.write(title_ + "\n")
            if content_:
                list_sents = text_segmentate(content_, 256, seps=['\n', '。', '？', '！', '?', '!']))
                if list_sents:
                    for sent in list_sents:
                        sent = sent.strip()
                        f_out.write(sent + "\n")

            f_out.write("\n")


if __name__ == "__main__":
    from_json_dir = "data.json"
    num_threads = 16
    num_process = 4

    # processing
    pool = mp.pool.Pool(num_process)
    with pool as p:
        p.map(json2txt, [1, 2, 3, 4])

    # # threading
    # for i in range(num_threads):
    #     to_txt_dir_ = "%d.txt" % i

    #     p_ = threading.Thread(
    #         target=json2txt,
    #         args=(from_json_dir, to_txt_dir_, i, num_threads),
    #     )
    #     p_.start()