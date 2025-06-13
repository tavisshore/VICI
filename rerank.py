

query_dict = {}
q_to_idx = {}


query_characteristics = {}
# Get query characteristics
with open('query.txt', 'r') as f:
    lines = f.readlines()
    for il, l in enumerate(lines):
        l = l.strip()
        query_id = l.split(' ')[0]
        print(query_id)
        chars = ' '.join(l.split(' ')[1:])
        chars = chars.split(' - ')
        chars[-1] = chars[-1].split(' ')[0]  # Remove the last part which is not a characteristic
        query_characteristics[query_id] = chars

print(query_characteristics['ptxLnXi8hQp8QXyI'])

ref_characteristics = {}
# Get ref characteristics
with open('ref.txt', 'r') as f:
    lines = f.readlines()
    for il, l in enumerate(lines):
        l = l.strip()
        query_id = l.split(' ')[0]
        chars = ' '.join(l.split(' ')[1:])
        chars = chars.split(' - ')
        chars[-1] = chars[-1].split(' ')[0]  # Remove the last part which is not a characteristic

        ref_characteristics[query_id] = chars

# Now get id to query id
q_to_idx = {}
idx_to_q = {}
with open('src/data/query_street_name.txt', 'r') as f:
    lines = f.readlines()
    for il, l in enumerate(lines):
        l = l.split('.')[0]
        q_to_idx[l] = il
        idx_to_q[il] = l


new_lines = []

# Get original retrievals
with open('results/answer.txt', 'r') as f:
    lines = f.readlines()
    for il, l in enumerate(lines):
        # tab separated
        l = l.strip().split('\t')
        # print(l)
        # print()
        qq = idx_to_q[il]

        # Get chars for both
        qq_chars = query_characteristics[qq]
        rr_chars = [ref_characteristics[i] for i in l]
        # print([len(x) for x in rr_chars])
        # print()
        scores = []

        for r in rr_chars:
            score = 0
            for ic, c in enumerate(qq_chars):
                try:
                    if c == r[ic]:
                        score += 1
                except:
                    print(c)
                    # print(ic)
                    # print(r)
                    breakpoint()
            scores.append(score)
        # Get ordered indices of scores in descending order
        ordered_indices = sorted(range(len(scores)), key=lambda k: scores[k], reverse=True)
        # Reorder l 
        new_l = [l[i] for i in ordered_indices]
        new_line = '\t'.join(new_l)
        new_lines.append(new_line)        


with open('answer_reranked.txt', 'w') as f:
    for l in new_lines:
        f.write(l + '\n')