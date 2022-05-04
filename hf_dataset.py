import torch
import numpy as np
from datasets import  load_dataset
from sklearn.model_selection import train_test_split

dataset = load_dataset('daily_dialog')
# daily_dialog: daily chat. 11118 dialogs, 7.9 utterance per dialog. We also manually label the developed dataset with communication intention and emotion information
# empathetic_dialogues: daily chat with empathetic tone. 76673 lines
#woz_dialogue: finding restaruant.
#curiosity_dialogs: human-assistant dialog, seaching geographic information
#deal_or_no_dialog: human-human negotiations on a multi-issue bargaining task
dataloader = torch.utils.data.DataLoader(dataset["train"], batch_size=32)
SPLIT_TOKEN = "<&&&>"
EA_TOKEN = " <%%%> "
src_dataset = []
tgt_dataset = []
all_ea = []
for i, data in enumerate(dataloader.dataset):
    # if i > 200:
    #     break
    dialog = data['dialog']
    act_info = data['act']
    emo_info = data['emotion']
    for i in range(1, len(dialog)):
        src = SPLIT_TOKEN.join(dialog[:i])
        src_dataset.append(src)
        ea_onehot = int(act_info[i]-1) + 4 * int(emo_info[i])
        all_ea.append(ea_onehot)
        tgt_dataset.append(dialog[i] + EA_TOKEN + str(ea_onehot))

src_dataset = np.array(src_dataset)
tgt_dataset = np.array(tgt_dataset)
def write_row(list, filename):
    list = [x+"\n" for x in list]
    with open(filename, 'w', encoding='utf-8') as f:
        f.writelines(list)


run_label = "EA_small"
src_train, src_val, tgt_train, tgt_val = train_test_split(src_dataset, tgt_dataset, test_size=0.08, random_state=42)
# write_row(src_train[:5000], 'EA_small/src-train.txt')
# write_row(src_val[:1000], 'EA_small/src-val.txt')
# write_row(tgt_train[:5000], 'EA_small/tgt-train.txt')
# write_row(tgt_val[:1000], 'EA_small/tgt-val.txt')
write_row(src_train, 'OpenNMT-py/EA_small/src-train.txt')
write_row(src_val, 'OpenNMT-py/EA_small/src-val.txt')
write_row(tgt_train, 'OpenNMT-py/EA_small/tgt-train.txt')
write_row(tgt_val, 'OpenNMT-py/EA_small/tgt-val.txt')