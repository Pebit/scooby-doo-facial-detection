import os
l=[]
for i in range(10):
    l.append(i)
save_path = os.path.join("parameters.DIR_POS_EXAMPLES", "character_name", "save_name")
print(save_path)
print(l[1:-1])