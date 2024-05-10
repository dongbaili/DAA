import argparse
sources = ["30", "60", "120", "150"]
parser = argparse.ArgumentParser(description='Domain generalization')
parser.add_argument("--n", type=int)
args = parser.parse_args()

for source in sources:
    with open(f"domainbed/txtlist/RMnist/{source}.txt", 'r') as f:
        lines = f.readlines()[:args.n]
    with open(f"domainbed/txtlist/RMnist/{source}.txt", 'w') as f:
        for line in lines:
            f.write(line)