import csv

with open("tractatus-body.txt", "r") as infile, open("tractatus.csv", "w", newline="") as outfile:
    fieldnames=["count", "depth", "number", "text"]
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
                                 "number": number, "text": text[:-1]})
            
            first_space_ix = line.find(" ")
            number = line[:first_space_ix]
            text = line[first_space_ix+1:]
            depth = max(1, len(number)-1) # eliminates dot
            count += 1
            start = False
        else:
            text += line
        

        
        


