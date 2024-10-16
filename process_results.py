import argparse
import datetime
import numpy as np
import os
import re 

parser = argparse.ArgumentParser()
parser.add_argument("--agent", type=str, required=True, choices=["reinforce", "sac", "hac", "PGMORL"], help="Type of agent")
parser.add_argument("--env-id", type=str, required=False, default="sardine/SlateTopK-Bored-v0",help="Environment name")
parser.add_argument("--seeds", type=int, nargs="+", default = [2705, 3688, 3751, 4685, 6383], help="Seeds")
parser.add_argument("--ranker", type=str, required=True, choices=["gems", "topk"], help="Type of ranker")
parser.add_argument("--slate-size", type=int, required=True, help="Size of the slate")
parser.add_argument("--num-items", type=int, required=True, help="Number of items")
parser.add_argument("--reward-type", type=str, required=True, choices=["click", "diversity"], help="Type of reward")


args = parser.parse_args()

clicks = []
diversities = []
catalog_coverages = []
runtime = []
for seed in args.seeds:
    file_format = f"{args.agent}_{args.ranker}_env{args.env_id}_slatesize{args.slate_size}-numitems{args.num_items}-seed{seed}-reward{args.reward_type}"
    # file_format = f"{args.agent}-{args.ranker}_slatesize{args.slate_size}-num-items{args.num_items}-seed{seed}-test"
    file_format = re.sub(r"[^a-zA-Z0-9]+", '-', file_format)
    # input(file_format)
    # file_format_test = re.sub(r"[^a-zA-Z0-9]+", '_', file_format_test)
    # check for files that contain the file_format
    for file in os.listdir("logs/das6"):
        if "1682" in file:
            print(file)
            print(file_format)
        if file_format in file:
            print("yes")
            if "test" in file:
                with open(os.path.join("logs/das6",file), "r") as f:
                    text = f.read()
                    # find the line that contains the test reward
                    test_reward = float(re.search(r"Average Return: (\d+\.\d+)", text).group(1))
                    print(test_reward)
                    clicks.append(test_reward)

                    test_diversity = float(re.search(r"Average Diversity: (\d+\.\d+)", text).group(1))
                    diversities.append(test_diversity)

                    test_catalog_coverage = float(re.search(r"Average Catalog Coverage: (\d+\.\d+)", text).group(1))
                    catalog_coverages.append(test_catalog_coverage)
            if "train" in file:
                with open(os.path.join("logs/das6",file), "r") as f:
                    text = f.read()
                    elapsed_time = re.search(r"Elapsed time: (\d+:\d+:\d+.\d+)", text).group(1)
                    # convert elapsed time to timedelta
                    elapsed_time = datetime.datetime.strptime(elapsed_time, "%H:%M:%S.%f") - datetime.datetime(1900, 1, 1)
                    # convert timedelta to seconds
                    elapsed_time = elapsed_time.total_seconds()
                    runtime.append(elapsed_time)

if clicks != []:
    try:
        print(f"Average Test Reward: {(sum(clicks)/len(clicks)):.2f}")
        mean_clicks = sum(clicks)/len(clicks)
        std_clicks = np.std(clicks)
    except:
        print("No results found for clicks")
        mean_clicks = ""
        std_clicks = ""
    try:
        print(f"Average Test Diversity: {(sum(diversities)/len(diversities)):.2f}")
        mean_diversities = sum(diversities)/len(diversities)
        std_diversities = np.std(diversities)
    except:
        print("No results found for diversities")
        mean_diversities = ""
        std_diversities = ""
    try:
        print(f"Average Test Catalog Coverage: {(sum(catalog_coverages)/len(catalog_coverages)):.2f}")
        mean_catalog_coverages = sum(catalog_coverages)/len(catalog_coverages)
        std_catalog_coverages = np.std(catalog_coverages)
    except:
        print("No results found for catalog coverages")
        mean_catalog_coverages = ""
    try:
        print(f"Average Elapsed Time: {(sum(runtime)/len(runtime)):.2f} s")
        mean_runtime = (sum(runtime)/len(runtime))
        print(mean_runtime)
        std_runtime = np.std(runtime)
    except:
        print("No results found for runtime")
        mean_runtime = ""
        std_runtime = ""

    # save results to results/results.csv
    with open(os.path.join("results", "results.csv"), "a") as f:
        f.write(f"\n{args.agent},{args.ranker},{args.env_id},{args.slate_size},{args.num_items},{args.reward_type},{mean_clicks:.2f},{std_clicks:.2f},{mean_diversities:.2f},{std_diversities:.2f},{mean_catalog_coverages:.2f},{std_catalog_coverages:.2f},{mean_runtime},{std_runtime}")
else:
    print("No results found")