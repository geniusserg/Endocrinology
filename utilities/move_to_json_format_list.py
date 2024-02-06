f_path = "clist2.txt"
f = open(f_path, "r")
g = f.read().split("\n")
f.close()
e = ",\n".join(['"'+i+'"' for i in g])
f = open(f_path, "w")
f.write(e)
f.close()