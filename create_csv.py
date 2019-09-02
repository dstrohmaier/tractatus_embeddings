import csv
import random


def create_base_csv(source="tractatus-body.txt", destination="tractatus.csv"):
    with open(source, "r") as infile, open(destination, "w", newline="") as outfile:
        fieldnames = ["count", "depth", "number", "text"]
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        count = 0
        start = True
        for line in infile:
            if line == "\n":
                start = True
                continue

            if start:
                if count != 0:
                    writer.writerow({"count": count, "depth": depth,
                                     "number": number.replace(".", ";"), "text": text[:-1]})
                    # the semicolon is dirty a bug fix
                    # it avoids for the number to be interpreted as a float

                first_space_ix = line.find(" ")
                number = line[:first_space_ix]
                text = line[first_space_ix + 1:]
                depth = max(1, len(number) - 1)  # eliminates dot
                count += 1
                start = False
            else:
                text += line

        writer.writerow({"count": count, "depth": depth,
                         "number": number.replace(".", ";") , "text": text[:-1]})

        return count


def add_splits(source="tractatus.csv", destination="tractatus_with_splits.csv", row_count=None):
    with open(source, "r") as infile, open(destination, "w", newline="") as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        if row_count is None: # only if length of csv is unknown
            row_count = -1 # -1 because we discount the header row
            for row in reader:
                row_count += 1
            infile.seek(0)
            reader = csv.reader(infile)


        train_count = row_count // 100 * 80
        test_count = row_count // 100 * 10
        validate_count = row_count // 100 * 10
        sum_count = train_count + test_count + validate_count

        if sum_count < row_count:
            train_count += row_count - sum_count
        elif sum_count > row_count:
            train_count -= sum_count - row_count

        labels = ["train" for _ in range(train_count)] + \
                 ["test" for _ in range(test_count)] + \
                 ["validation" for _ in range(validate_count)]
        random.shuffle(labels)

        assert len(labels) == row_count, "Number of split labels and number of rows differ"

        header_row = next(reader) + ["split"]
        #print(header_row)

        content_rows = []
        for row,  label in zip(reader, labels):
            #print(row)
            row.append(label)
            content_rows.append(row)

        writer.writerows([header_row]+content_rows)

if __name__ == "__main__":
    random.seed(42)
    row_count = create_base_csv()
    add_splits(row_count=row_count)
