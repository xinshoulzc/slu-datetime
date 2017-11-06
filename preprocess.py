
# coding = utf-8

import json

def checkin(word, tag):
    arr = list()
    length = len(word)
    for i in xrange(length):
        if tag[i] != "PUBTIME":
            continue
        else:
            # print word[i].encode('utf-8'), tag[i]
            arr.append(word[i])
    if len(arr) > 1: return arr
    else: return None

if __name__ == "__main__":

    source_file_path = "data/train/train.seq.in"
    target_file_path = "data/train/train.seq.out"
    pos_file_path = "data/train/train.seq.pos"
    output_file_path = "data/results.json"
    print source_file_path
    arr = list()
    output_file = open(output_file_path, "w")
    hashtable = dict({})
    with open(source_file_path, "r") as source_file:
        with open(target_file_path, "r") as target_file:
            with open(pos_file_path, "r") as pos_file:
                source, target, pos = source_file.readline(), target_file.readline(), pos_file.readline()
                while source and target and pos:
                    word = source.split()
                    tag = target.split()
                    # print word, tag
                    tmp = checkin(word, tag)
                    if tmp:
                        arr.append(tmp)
                        line = json.dumps(tmp, encoding="utf-8", ensure_ascii=False)
                        if not hashtable.get(line, None):
                            output_file.write(line + '\n')
                        hashtable[line] = 1
                    source, target, pos = source_file.readline(), target_file.readline(), pos_file.readline()

    output_file.close()