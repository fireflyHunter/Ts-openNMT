import argparse
import os
import numpy as np
RAW_TRAIN_DATA="./data/aligned-good-0.67.txt"
RAW_PARTIAL='./data/aligned-good_partial-0.53.txt'
MANUAL_DATA='./data/manual_data.txt'
validation_size = 5000
test_size_backup = 1000



def write_to_file(filename,data):
    with open(filename, 'w', encoding='utf8') as f:
        f.writelines(data)


def split_data():
    rows = []
    raw_data = []
    lines = 0
    scores = []
    with open(RAW_TRAIN_DATA, 'r', encoding='utf8') as f:
        for line in f.readlines():
            lines += 1
            raw_data.append(line)
            data = line.split("\t")
            if len(data) != 3:
                print(len(data))
                continue
            rows.append(data)
            scores.append(float(data[-1]))
    
    print("{} lines of data in total with avg score {}".format(lines, sum(scores) / lines))
    raw_data = np.asarray(raw_data)
    np.random.shuffle(raw_data)
    train, val, test = raw_data[:-6000], raw_data[-6000:-1000], raw_data[-1000]
    write_to_file("./data/good_train.txt", train)
    write_to_file("./data/good_val.txt", val)
    write_to_file("./data/good_test_backup.txt", test)




def main():
    parser = argparse.ArgumentParser(description='process from parlai format to'
                                                 ' opennmt-py format')
    parser.add_argument('-o', '--output_dir_name', help='output directory')
    parser.add_argument('-i', '--input_dir_name', help='input files')
    parser.add_argument('-t', '--test', help='test')
    args = parser.parse_args()
    input_names = []
    for root, dirs, files in os.walk(args.input_dir_name):
        input_names.extend(files)
    for input_file_name in input_names:
        input_file_path = os.path.join(args.input_dir_name, input_file_name)
        num_of_lines = 0
        source_utterances = []
        target_utterances = []
        scores = []
        with open(input_file_path, 'r', encoding='utf8') as f:
            for line in f.readlines():
                data = line.split("\t")
                if len(data) != 3:
                    
                    continue
                num_of_lines += 1
                
                source_utterances.append(data[0])
                target_utterances.append(data[1])
                scores.append(float(data[2]))

        input_file_root = \
            os.path.basename(os.path.splitext(input_file_name)[0])
        source_output_file_name = os.path.join(args.output_dir_name,
                                        input_file_root + '_source.txt')
        with open(source_output_file_name, 'w', encoding='utf8') as out_file:
            out_file.write('\n'.join(source_utterances))
        target_output_file_name = os.path.join(args.output_dir_name,
                                        input_file_root + '_target.txt')
        with open(target_output_file_name, 'w', encoding='utf8') as out_file:
            out_file.write('\n'.join(target_utterances))


if __name__ == '__main__':
    main()