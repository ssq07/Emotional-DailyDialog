import os
import warnings
warnings.filterwarnings("ignore")
out_file = 'demo_out'
in_file = 'src_demo'

def read_output():
    with open(out_file, 'r', encoding='utf8') as f:
        data = f.readline()
    return data.strip()
def print_file():
    with open(in_file, 'r', encoding='utf8') as f:
        data = f.readline()
    dialogues = data.split(" <&&&> ")
    cur_speaker = "A"
    for line in dialogues:
        print("{}: {}".format(cur_speaker, line))
        cur_speaker = "B" if cur_speaker == "A" else "A"
    return cur_speaker

def append_src(out):
    with open(in_file, 'r', encoding='utf8') as f:
        data = f.readline()
    data = data.split(" <&&&> ")[-5:] + [out]
    data = " <&&&> ".join(data)

    with open(in_file, 'w', encoding='utf8') as f:
        f.write(data)


print("*"*50)
print("Initialing Dialogue:")
print("*"*50)
cur_speaker = print_file()
user_input = "n"
while user_input != 'q':
    user_input = input("Press 'N' to continue the dialogue with Transformer, or input your own response\n")
    if user_input in ['n','N']:
        # emotion = input("Please select an emotion by inputting a number from 0 to 5. (e.g. 0 is joy)\n")
        os.system(
            'python translate.py -model ./daily_dialog/run116/best_trans_model.pt --fp32 -src ./src_demo -output ./demo_out')
        system_out = read_output()
        print("Transformer({}): {}".format(cur_speaker, system_out))
        append_src(system_out)
    elif user_input == 'q':
        print("Program exits.")
    else:
        print("User({}): {}".format(cur_speaker, user_input))
        append_src(user_input)
        cur_speaker = "B" if cur_speaker == "A" else "A"
        os.system(
            'python translate.py -model ./daily_dialog/run116/best_trans_model.pt --fp32 -src ./src_demo -output ./demo_out' )
        system_out = read_output()
        print("Transformer({}): {}".format(cur_speaker, system_out))
        append_src(system_out)
    cur_speaker = "B" if cur_speaker == "A" else "A"

# os.system('python translate.py -model ./daily_dialog/run116/best_trans_model.pt --fp32 -src ./src_demo -output ./demo_out')
# ./daily_dialog/run116/trans_model/model_step_20000.pt --src ./ --tgt ./
# with open("tgt-train.txt", 'r') as f:
#     data = f.readlines()
#     print(data[0])
# python translate.py -model ./daily_dialog/run116/model_step_20000.pt --fp32 -src ./src_demo -output ./demo_out
